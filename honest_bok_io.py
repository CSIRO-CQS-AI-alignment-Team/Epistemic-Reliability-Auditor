"""Shared stdlib-only contracts for the honest best-of-K pipeline.

``debate-bok.py`` and ``honest_select.py`` reuse this module for paths, manifests,
resume keys, JSONL durability, request seeds, verifier identity, and publication. It
imports only stdlib code plus dependency-light candidate helpers.

Key safeguards:

1. ``Workspace`` derives production paths from the dataset selector. ``--dry-run`` uses
   an isolated subtree and cannot publish into the canonical dataset.
2. Every candidate carries the digest of a run identity covering inputs, ``K``, rounds,
   model endpoint, prompt, seed schedule, and transcript budget. Scoring, selection, and
   publication require the same manifest identity.
3. Append-only JSONL writes one complete, locked, fsynced line at a time. Resume may
   discard only a torn final line; downstream readers require fully valid artifacts.
4. ``publish_transaction`` stages and validates both dataset outputs, commits them under
   a transaction marker, and restores both targets if either replacement fails.
5. Base-verifier checks bind selection to an allowed unfine-tuned checkpoint basename and
   full content fingerprint.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from adversarial_transcript import common  # noqa: E402

DATASETS = (
    "QuALITY-H", "QuALITY-H-50", "QuALITY-H-250",
    "BoolQ", "GPQA", "TruthfulQA",
)
# The two ablation datasets are directory-level copies of QuALITY-H.  Their files
# intentionally retain the QuALITY-H stem, while their run directories retain the
# variant name so downstream artifacts stay isolated.
DATASET_FILE_STEMS = {
    "QuALITY-H-50": "QuALITY-H",
    "QuALITY-H-250": "QuALITY-H",
}
DEFAULT_TRANSCRIPT_BUDGET = {
    "unit": "whitespace_words",
    "hard_limit": 150,
    "soft_limit": 100,
    "truncation_policy": "legacy_space_split_content_cap_marker_outside",
    "config_path": "config/debate-default.yaml",
}
TRANSCRIPT_BUDGETS = {
    "QuALITY-H-50": {
        "unit": "whitespace_words",
        "hard_limit": 50,
        "soft_limit": 50,
        "truncation_policy": "strict_stored_words_including_marker",
        "config_path": "config/debate-50.yaml",
    },
    "QuALITY-H-250": {
        "unit": "whitespace_words",
        "hard_limit": 250,
        "soft_limit": 200,
        "truncation_policy": "strict_stored_words_including_marker",
        "config_path": "config/debate-250.yaml",
    },
}
HONEST_CONDITION = "honest"

# Mirrors debate.NUM_ROUNDS. Duplicated as a plain constant because `debate` imports
# tqdm + reads a YAML config at import time and must stay a LAZY import here.
DEFAULT_NUM_ROUNDS = 3
DEFAULT_K = 8

# The ONLY endpoint a production run may use. vLLM, local, OpenAI-compatible.
PRODUCTION_BASE_URL = "http://localhost:18888/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_GENERATOR_MODEL = "google/gemma-4-31B-it"
DEFAULT_VERIFIER_MODEL = "checkpoints/gpt-oss-20b-bf16-base"
# The scorer name score_verifier.py stamps on a REAL forced-choice readout.
PRODUCTION_SCORER = "base-forced-choice-logits"

MANIFEST_SCHEMA_VERSION = "aisi.honest_bok.run_manifest.v1"
SELECTION_SCHEMA_VERSION = "aisi.honest_bok.selection.v1"
TXN_SCHEMA_VERSION = "aisi.honest_bok.publish_txn.v1"
VERIFIER_MANIFEST_SCHEMA_VERSION = "aisi.honest_bok.verifier_manifest.v1"

# The generation prompt contract lives here so selection can validate it without
# importing the generator. Score/select/write reject pools with a different prompt
# version; regenerate such a pool with ``debate-bok.py --overwrite``.
GENERATION_PROMPT_VERSION = "honest-bok-v1"

# Base-verifier identity. The selector must be the UNFINE-TUNED base model; accepting a
# fine-tuned checkpoint would silently change what "V_base selected it" means and break
# comparability with the adversarial arm's V_base provenance.
#
# Identity is by directory BASENAME plus a full-content fingerprint -- never an
# absolute path, which changes between HPC filesystems. `config/base-verifier.json` may
# extend the allowlist of accepted IDs and is auditable in-repo; it CANNOT relax
# FINETUNED_MARKERS, which is checked first and is deliberately not overridable.
BASE_VERIFIER_IDS = ("gpt-oss-20b-bf16-base",)
BASE_VERIFIER_ALLOWLIST_FILE = os.path.join("config", "base-verifier.json")
# Substrings that mark a fine-tuned/adapted checkpoint. Fail-closed denylist for the
# exact confusion this guards against (see checkpoints/gpt-oss-20b-verifier-fullft-*).
FINETUNED_MARKERS = ("fullft", "-ft", "_ft", "finetune", "fine-tune", "fted", "lora",
                     "adapter", "sft", "verifier-fullft", "adversarial", "honest")

MODE_PRODUCTION = "production"
MODE_DRYRUN = "dryrun"
MODE_TEST = "test"

STATUS_OK = "ok"
STATUS_FALLBACK = "ok_fallback_no_qy_gate"
STATUS_NONE = "no_honest_survivor"
SELECTION_STATUSES = (STATUS_OK, STATUS_FALLBACK, STATUS_NONE)


class HonestBokError(RuntimeError):
    """Contract violation that must abort the run."""


def dataset_file_stem(dataset):
    return DATASET_FILE_STEMS.get(dataset, dataset)


def transcript_budget_for_dataset(dataset):
    if dataset not in DATASETS:
        raise HonestBokError("unknown dataset %r" % (dataset,))
    return dict(TRANSCRIPT_BUDGETS.get(dataset, DEFAULT_TRANSCRIPT_BUDGET))


# ---- workspace ------------------------------------------------------------
class Workspace:
    """Resolved, mode-scoped file layout. Production paths cannot be overridden.

    production : reads dataset/<DS>/, writes runs/honest/<DS>/, publishes IN PLACE.
                 The QuALITY-H length variants retain the copied ``QuALITY-H-*`` file
                 stem inside their distinct ``QuALITY-H-50/250`` directories.
    dryrun     : reads the real dataset/<DS>/ inputs, but every write (artifacts AND
                 publication) is confined to runs/honest/<DS>/dryrun/.
    test       : everything under an arbitrary root; only reachable from the internal
                 API, never from argv.
    """

    def __init__(self, dataset, mode=MODE_PRODUCTION, root=None):
        if dataset not in DATASETS:
            raise HonestBokError("unknown dataset %r" % (dataset,))
        if mode not in (MODE_PRODUCTION, MODE_DRYRUN, MODE_TEST):
            raise HonestBokError("unknown workspace mode %r" % (mode,))
        self.dataset = dataset
        self.file_stem = dataset_file_stem(dataset)
        self.mode = mode
        self.root = os.path.abspath(root or _REPO_ROOT)
        self.dataset_dir = os.path.join(self.root, "dataset", dataset)
        run_base = os.path.join(self.root, "runs", "honest", dataset)
        if mode == MODE_DRYRUN:
            self.run_dir = os.path.join(run_base, "dryrun")
            self.publish_dir = os.path.join(self.run_dir, "publish")
        else:
            self.run_dir = run_base
            # production publishes in place; test publishes into its own dataset dir
            self.publish_dir = self.dataset_dir

    # --- inputs (always read-only) ---
    @property
    def items(self):
        """Generation input. The ONLY accepted items file."""
        return os.path.join(self.dataset_dir, "%s-no-debate.json" % self.file_stem)

    @property
    def stories(self):
        return os.path.join(self.dataset_dir, "%s-title-story.json" % self.file_stem)

    @property
    def canonical_source(self):
        return os.path.join(self.dataset_dir, "%s.json" % self.file_stem)

    # --- run artifacts ---
    @property
    def candidates(self):
        """SINGULAR `candidate.jsonl` by instruction; adversarial uses the plural."""
        return os.path.join(self.run_dir, "candidate.jsonl")

    @property
    def scores(self):
        return os.path.join(self.run_dir, "verifier_scores.jsonl")

    @property
    def selected(self):
        return os.path.join(self.run_dir, "selected.jsonl")

    @property
    def report(self):
        return os.path.join(self.run_dir, "report.json")

    @property
    def manifest(self):
        return os.path.join(self.run_dir, "run_manifest.json")

    @property
    def txn_marker(self):
        return os.path.join(self.run_dir, "publish_txn.json")

    @property
    def verifier_manifest(self):
        return os.path.join(self.run_dir, "verifier_manifest.json")

    # --- publication targets ---
    @property
    def canonical_out(self):
        return os.path.join(self.publish_dir, "%s.json" % self.file_stem)

    @property
    def with_honest_out(self):
        return os.path.join(self.publish_dir,
                            "%s-with-honest-transcripts.json" % self.file_stem)

    @property
    def publishes_in_place(self):
        return os.path.realpath(self.publish_dir) == os.path.realpath(self.dataset_dir)

    def ensure_dirs(self):
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.publish_dir, exist_ok=True)

    def describe(self):
        return {"dataset": self.dataset, "file_stem": self.file_stem,
                "mode": self.mode, "run_dir": self.run_dir,
                "publish_dir": self.publish_dir}


def assert_production_layout(ws):
    """Re-derive the locked production paths and compare REALPATHS.

    Belt and braces: Workspace already derives these, but this makes the lock an
    explicit, testable assertion rather than an emergent property of the class.
    """
    if ws.mode != MODE_PRODUCTION:
        return
    expect_root = os.path.realpath(_REPO_ROOT)
    if os.path.realpath(ws.root) != expect_root:
        raise HonestBokError("production workspace must be rooted at the repo (%s), got %s"
                             % (expect_root, ws.root))
    expected = {
        "items": os.path.join(expect_root, "dataset", ws.dataset,
                              "%s-no-debate.json" % ws.file_stem),
        "candidates": os.path.join(expect_root, "runs", "honest", ws.dataset,
                                   "candidate.jsonl"),
        "scores": os.path.join(expect_root, "runs", "honest", ws.dataset,
                               "verifier_scores.jsonl"),
        "selected": os.path.join(expect_root, "runs", "honest", ws.dataset,
                                 "selected.jsonl"),
        "canonical_out": os.path.join(expect_root, "dataset", ws.dataset,
                                      "%s.json" % ws.file_stem),
        "with_honest_out": os.path.join(expect_root, "dataset", ws.dataset,
                                        "%s-with-honest-transcripts.json" % ws.file_stem),
    }
    for name, want in expected.items():
        got = os.path.realpath(getattr(ws, name))
        # The target may not exist yet; realpath of a non-existent leaf is stable.
        if got != os.path.realpath(want):
            raise HonestBokError(
                "production path lock violated for %s: expected %s, got %s"
                % (name, want, got))


def assert_production_endpoint(base_url):
    if base_url != PRODUCTION_BASE_URL:
        raise HonestBokError(
            "production generation is locked to %s; refusing %r (an external or "
            "non-local endpoint must never generate canonical transcripts)"
            % (PRODUCTION_BASE_URL, base_url))


# ---- fingerprints / run manifest ------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(obj):
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_identity(ws, k, num_rounds, backend, base_url, model, seed,
                   dry_run, prompt_version, lm_config, transcript_budget=None,
                   debate_config_sha256=None):
    """The fields that MUST be identical across every candidate of one pool.

    `lm_config` (temperature/top_p/max_tokens from config/debate-default.yaml) is
    folded in as a digest: editing the sampling config mid-pool would silently mix two
    transcript distributions, which is the same class of contamination as a model or
    seed change.
    """
    identity = {
        "dataset": ws.dataset,
        "k": int(k),
        "num_rounds": int(num_rounds),
        "backend": backend,
        "base_url": base_url,
        "model": model,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "prompt_version": prompt_version,
        "seed_policy_version": SEED_POLICY_VERSION,
        "lm_config_digest": canonical_digest(lm_config),
        "items_sha256": sha256_file(ws.items),
        "stories_sha256": sha256_file(ws.stories),
    }
    # Transcript-budget fields are recorded when the selected dataset defines an explicit
    # ablation contract; default-pool identities omit them.
    if transcript_budget is not None:
        identity["transcript_budget"] = dict(transcript_budget)
        identity["debate_config_sha256"] = str(debate_config_sha256)
    return identity


def identity_digest(identity):
    return canonical_digest(identity)


def build_manifest(ws, identity, scope, lm_config):
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mode": ws.mode,
        "identity": identity,
        "identity_digest": identity_digest(identity),
        # Recorded for audit; deliberately NOT part of the identity, so a sharded or
        # resumed run is compatible while a model/seed/prompt change is not.
        "scope": scope,
        "lm_config": lm_config,
        "paths": {"items": ws.items, "stories": ws.stories, "candidates": ws.candidates},
    }


def read_manifest(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise HonestBokError("%s is not a %s manifest" % (path, MANIFEST_SCHEMA_VERSION))
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise HonestBokError("%s has no identity block" % path)
    if manifest.get("identity_digest") != identity_digest(identity):
        raise HonestBokError(
            "%s identity_digest does not match its identity block (tampered or "
            "hand-edited manifest)" % path)
    return manifest


def diff_identity(want, got):
    """Human-readable list of the fields that differ (for the refusal message)."""
    keys = sorted(set(want) | set(got))
    return [{"field": key, "expected": want.get(key), "found": got.get(key)}
            for key in keys if want.get(key) != got.get(key)]


def require_manifest_for_read(ws, expect_dry_run, stage):
    """Load the pool manifest and enforce the active generation contract.

    Candidate/manifest self-consistency is insufficient by itself; the recorded identity
    must also match this code's dataset, mode, prompt, seed, and budget contracts.
    """
    manifest = read_manifest(ws.manifest)
    if manifest is None:
        raise HonestBokError(
            "%s: no run manifest at %s. The candidate pool cannot be provenance-checked; "
            "regenerate with debate-bok.py." % (stage, ws.manifest))
    identity = manifest["identity"]

    if bool(identity.get("dry_run")) != bool(expect_dry_run):
        sandbox = Workspace(ws.dataset, mode=MODE_DRYRUN, root=ws.root).run_dir
        raise HonestBokError(
            "%s: candidate pool was generated with dry_run=%s but this is a %s run. "
            "Dry-run pools belong in %s and can never be promoted to production."
            % (stage, identity.get("dry_run"),
               "dry-run" if expect_dry_run else "production", sandbox))

    # --- active contract checks ---
    if identity.get("dataset") != ws.dataset:
        raise HonestBokError(
            "%s: manifest is for dataset %r but this workspace is %r"
            % (stage, identity.get("dataset"), ws.dataset))
    if manifest.get("mode") != ws.mode:
        raise HonestBokError(
            "%s: manifest was written in mode %r but this run is mode %r"
            % (stage, manifest.get("mode"), ws.mode))
    if identity.get("prompt_version") != GENERATION_PROMPT_VERSION:
        raise HonestBokError(
            "%s: candidate pool was generated with prompt_version %r, but this code "
            "expects %r. Regenerate the pool with `debate-bok.py --overwrite`."
            % (stage, identity.get("prompt_version"), GENERATION_PROMPT_VERSION))
    if identity.get("seed_policy_version") != SEED_POLICY_VERSION:
        raise HonestBokError(
            "%s: candidate pool used seed policy %r, but this code expects %r; "
            "regenerate the pool."
            % (stage, identity.get("seed_policy_version"), SEED_POLICY_VERSION))
    if identity.get("lm_config_digest") != canonical_digest(manifest.get("lm_config")):
        raise HonestBokError(
            "%s: manifest lm_config does not hash to identity.lm_config_digest; the "
            "manifest is inconsistent or hand-edited." % stage)
    budget_contract = TRANSCRIPT_BUDGETS.get(ws.dataset)
    if budget_contract is not None:
        if identity.get("transcript_budget") != budget_contract:
            raise HonestBokError(
                "%s: manifest transcript budget %r does not match dataset %s's locked "
                "budget %r; regenerate the pool with the matching --transcript-len."
                % (stage, identity.get("transcript_budget"), ws.dataset, budget_contract))
        config_path = os.path.join(_REPO_ROOT, budget_contract["config_path"])
        if identity.get("debate_config_sha256") != sha256_file(config_path):
            raise HonestBokError(
                "%s: %s changed since this pool was generated; regenerate so the "
                "recorded hard/soft transcript budget matches the live prompt."
                % (stage, config_path))
    if not isinstance(identity.get("k"), int) or identity["k"] < 1:
        raise HonestBokError("%s: manifest k is not a positive integer" % stage)

    if not expect_dry_run:
        if identity.get("backend") != "api":
            raise HonestBokError(
                "%s: candidate pool backend is %r; production requires the vLLM HTTP "
                "backend." % (stage, identity.get("backend")))
        assert_production_endpoint(identity.get("base_url"))
        if ws.mode == MODE_PRODUCTION:
            assert_production_layout(ws)

    # Inputs must not have changed under the pool.
    for key, path in (("items_sha256", ws.items), ("stories_sha256", ws.stories)):
        if identity.get(key) != sha256_file(path):
            raise HonestBokError(
                "%s: %s changed since the candidate pool was generated (%s mismatch); "
                "regenerate the pool." % (stage, path, key))
    return manifest


def check_candidate_provenance(candidate, digest, identity):
    """Every candidate must belong to this exact run. Returns a reason or None.

    Checks every identity-bearing field build_candidate writes, so a row that was
    hand-edited or carried over from another run cannot pass by matching only the
    digest.
    """
    meta = candidate.get("gen_meta")
    if not isinstance(meta, dict):
        return "missing gen_meta"
    if meta.get("manifest_digest") != digest:
        return "manifest_digest %r != run %r" % (meta.get("manifest_digest"), digest)
    if bool(meta.get("dry_run")) != bool(identity.get("dry_run")):
        return "gen_meta.dry_run disagrees with the manifest"
    for field in ("backend", "base_url", "model", "num_rounds", "prompt_version",
                  "seed_policy_version"):
        if meta.get(field) != identity.get(field):
            return "gen_meta.%s %r != manifest %r" % (field, meta.get(field),
                                                      identity.get(field))
    if meta.get("seed") != identity.get("seed"):
        return "gen_meta.seed %r != manifest %r" % (meta.get("seed"), identity.get("seed"))
    if meta.get("prompt_version") != GENERATION_PROMPT_VERSION:
        return "gen_meta.prompt_version %r != current %r" % (
            meta.get("prompt_version"), GENERATION_PROMPT_VERSION)
    if canonical_digest(meta.get("lm_config")) != identity.get("lm_config_digest"):
        return "gen_meta.lm_config does not match the manifest lm_config_digest"
    for field in ("transcript_budget", "debate_config_sha256"):
        if field in identity and meta.get(field) != identity.get(field):
            return "gen_meta.%s does not match the manifest" % field
    budget = identity.get("transcript_budget")
    if isinstance(budget, dict):
        hard_limit = budget.get("hard_limit")
        if (isinstance(hard_limit, bool) or not isinstance(hard_limit, int)
                or hard_limit < 1):
            return "manifest transcript_budget.hard_limit is not a positive integer"
        transcript = candidate.get("transcript") or {}
        for round_index, round_row in enumerate(transcript.get("rounds") or [], 1):
            for debater in common.DEBATER_NAMES:
                argument = round_row.get(debater) if isinstance(round_row, dict) else None
                if not isinstance(argument, str):
                    return "transcript round %d %s argument is not a string" % (
                        round_index, debater)
                observed = len(argument.split())
                if observed > hard_limit:
                    return (
                        "transcript round %d %s has %d whitespace words, exceeding "
                        "manifest hard_limit %d"
                        % (round_index, debater, observed, hard_limit))
    # Fixed markers written by build_candidate; a foreign row rarely reproduces them.
    if meta.get("stance_source") != "stance_rng":
        return "gen_meta.stance_source %r != 'stance_rng'" % (meta.get("stance_source"),)
    if meta.get("selector_arm") != "honest_bok":
        return "gen_meta.selector_arm %r != 'honest_bok'" % (meta.get("selector_arm"),)
    expected_candidate_seed = "%s|%s|%s|%s" % (
        identity.get("seed"), candidate.get("dataset_index"), HONEST_CONDITION,
        candidate.get("candidate_idx"))
    if meta.get("candidate_seed") != expected_candidate_seed:
        return "gen_meta.candidate_seed %r != expected %r" % (
            meta.get("candidate_seed"), expected_candidate_seed)
    # Re-derive the sampling seeds; a stale row from another policy/seed cannot match.
    expected_seeds = [] if identity.get("dry_run") else model_request_seeds(
        identity["seed"], int(candidate["dataset_index"]),
        int(candidate["candidate_idx"]), identity["num_rounds"])
    if list(meta.get("model_request_seeds") or []) != expected_seeds:
        return "gen_meta.model_request_seeds do not match the seed policy"
    return None


# ---- base-verifier identity -------------------------------------------------
def base_verifier_allowlist(root=None):
    """Allowed BASE verifier ids: the built-in set plus an auditable in-repo override."""
    root = _REPO_ROOT if root is None else root
    ids = list(BASE_VERIFIER_IDS)
    path = os.path.join(root, BASE_VERIFIER_ALLOWLIST_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        extra = payload.get("base_verifier_ids")
        if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
            raise HonestBokError("%s: base_verifier_ids must be a list of strings" % path)
        ids.extend(extra)
    return tuple(dict.fromkeys(ids))


def verifier_id_for(model_path):
    """Path-independent identity: the checkpoint directory's basename."""
    return os.path.basename(os.path.normpath(str(model_path)))


# Checkpoint fingerprinting. Names and sizes cannot identify weights rewritten in place,
# so production fingerprints hash the full content of every weight shard. This stdlib
# implementation keeps the honest pipeline dependency-light.
WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "*.pt", "*.pth")
META_PATTERNS = (
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "pytorch_model.bin.index.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja",
)
_HASH_CHUNK = 8 << 20  # 8 MiB


def _matches(name, patterns):
    import fnmatch

    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def scan_checkpoint(path):
    """Sorted (relname, size, mtime_ns, kind) for the weight + metadata files."""
    out = []
    root = os.path.abspath(path)
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            if _matches(name, WEIGHT_PATTERNS):
                kind = "weight"
            elif _matches(name, META_PATTERNS):
                kind = "meta"
            else:
                continue
            stat = os.stat(full)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            out.append((rel, int(stat.st_size), int(stat.st_mtime_ns), kind))
    return sorted(out)


def _index_referenced_shards(path, entries):
    names = {name for name, _s, _m, kind in entries if kind == "meta"}
    referenced = set()
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if index_name not in names:
            continue
        try:
            with open(os.path.join(path, index_name), encoding="utf-8") as f:
                index = json.load(f)
        except (OSError, ValueError) as exc:
            raise HonestBokError("%s: unreadable weight index %s (%s)"
                                 % (path, index_name, exc))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise HonestBokError("%s: %s has no usable weight_map" % (path, index_name))
        referenced |= {str(v) for v in weight_map.values()}
    return referenced


def validate_checkpoint_weights(path, entries):
    """Structural checks. A partially-transferred checkpoint hashes stably but cannot load."""
    problems = []
    weights = {name: size for name, size, _m, kind in entries if kind == "weight"}
    if not weights:
        problems.append("no weight shards found")
    empty = sorted(name for name, size in weights.items() if size <= 0)
    if empty:
        problems.append("zero-byte weight shard(s): %s" % empty[:5])
    referenced = _index_referenced_shards(path, entries)
    if referenced:
        missing = sorted(r for r in referenced if r not in weights)
        if missing:
            problems.append("weight index references absent shard(s): %s" % missing[:5])
        empty_ref = sorted(r for r in referenced if weights.get(r, 0) <= 0)
        if empty_ref:
            problems.append("weight index references empty shard(s): %s" % empty_ref[:5])
    return problems


def _fingerprint_cache_path(cache_dir, path):
    key = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:32]
    return os.path.join(cache_dir, "verifier_fingerprint_%s.json" % key)


def _cache_signature(path, entries):
    """Filesystem change identity for the fingerprint cache.

    (name, size, mtime_ns) alone is forgeable: `os.utime` can restore an mtime after an
    in-place shard rewrite, so a same-name/same-size content swap would reuse a stale
    cached fingerprint. We therefore also record st_ctime_ns (inode change time, which
    userspace cannot set and which any write updates) plus st_dev/st_ino (so replacing
    the file with a different inode, or moving the checkpoint across filesystems, misses
    the cache).

    Residual threat, stated plainly: ctime is not settable through normal APIs but is
    not a cryptographic guarantee -- root can move the system clock, and some network or
    overlay filesystems report coarse or non-standard ctime. The cache is ONLY an
    optimisation; the full-content hash is authoritative and `force=True` (exposed as
    `--rehash-verifier`) always recomputes it.
    """
    signature = []
    for name, _size, _mtime, kind in entries:
        stat = os.stat(os.path.join(path, name))
        signature.append([name, kind, int(stat.st_dev), int(stat.st_ino),
                          int(stat.st_size), int(stat.st_mtime_ns),
                          int(stat.st_ctime_ns)])
    return signature


def verifier_fingerprint(model_path, cache_dir=None, force=False):
    """Full-content fingerprint of a checkpoint dir; None when it is not present.

    Hashes every weight shard end to end plus the metadata files, so an in-place
    fine-tune or a same-size shard swap changes the fingerprint. Structural defects
    (no weights, a zero-byte shard, an index naming an absent shard) abort rather than
    producing a stable fingerprint for an unloadable checkpoint.

    Hashing tens of GB is a one-time cost, so `cache_dir` memoises the result under the
    filesystem change identity in `_cache_signature` (dev/inode/size/mtime/ctime), which
    an mtime-restoring in-place swap cannot forge. `force=True` recomputes regardless;
    the full-content hash is always the authority.
    """
    path = str(model_path)
    if not os.path.isdir(path):
        return None
    entries = scan_checkpoint(path)
    if not entries:
        return None
    problems = validate_checkpoint_weights(path, entries)
    if problems:
        raise HonestBokError(
            "%s is not a usable checkpoint: %s. Refusing to fingerprint it -- a "
            "partially transferred checkpoint would hash stably but cannot be loaded."
            % (path, "; ".join(problems)))

    cache_key = {"path": os.path.abspath(path), "algo": "sha256-full-content-v2",
                 "signature": _cache_signature(path, entries)}
    cache_file = _fingerprint_cache_path(cache_dir, path) if cache_dir else None
    if cache_file and not force and os.path.isfile(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("cache_key") == cache_key and cached.get("fingerprint"):
                return cached["fingerprint"]
        except (OSError, ValueError):
            pass  # a corrupt cache entry is simply recomputed

    digests = []
    for name, size, _mtime, kind in entries:
        digest = hashlib.sha256()
        with open(os.path.join(path, name), "rb") as f:
            while True:
                block = f.read(_HASH_CHUNK)
                if not block:
                    break
                digest.update(block)
        digests.append([name, size, kind, digest.hexdigest()])
    fingerprint = canonical_digest({"algo": "sha256-full-content-v1", "files": digests})

    if cache_file:
        try:
            atomic_write_json(cache_file, {"cache_key": cache_key,
                                           "fingerprint": fingerprint})
        except OSError:
            pass  # caching is an optimisation, never a correctness requirement
    return fingerprint


def check_base_verifier(model_path, stage, root=None):
    """Refuse anything that is not a recognised UNFINE-TUNED base checkpoint.

    The experiment design fixes the selector as the un-fine-tuned base gpt-oss-20b. Accepting a
    fine-tuned checkpoint would silently change what "V_base selected it" means and
    destroy comparability with the adversarial arm's V_base provenance. Identity is by
    basename + content fingerprint, never an absolute path (which moves between HPC
    filesystems).
    """
    verifier_id = verifier_id_for(model_path)
    lowered = verifier_id.lower()
    hits = [marker for marker in FINETUNED_MARKERS if marker in lowered]
    if hits:
        raise HonestBokError(
            "%s: verifier %r looks fine-tuned (matched %s). The honest selector must be "
            "the UNFINE-TUNED base model." % (stage, verifier_id, hits))
    allowed = base_verifier_allowlist(root)
    if verifier_id not in allowed:
        raise HonestBokError(
            "%s: verifier %r is not a recognised base checkpoint (allowed: %s). Add it "
            "to %s deliberately if it really is the un-fine-tuned base model."
            % (stage, verifier_id, list(allowed), BASE_VERIFIER_ALLOWLIST_FILE))
    return verifier_id


def build_verifier_manifest(model_path, option_seed, score_prompt_version, dry_run,
                            root=None, require_fingerprint=False, cache_dir=None,
                            force=False):
    """Record the base-verifier identity the scores will be attributable to.

    `require_fingerprint` is set for production scoring: discovering that the checkpoint
    could not be fingerprinted should abort BEFORE the GPU hours are spent, not at
    select time when the scores already exist.

    `force` must carry the caller's --rehash-verifier intent. This is the fingerprint the
    scores get BOUND to, so trusting a cache here would defeat the flag entirely: the run
    would stamp a cached (possibly stale or poisoned) fingerprint onto every row, spend
    the whole scoring job, and only discover the disagreement when verify_pool re-hashes
    at the end.
    """
    verifier_id = None if dry_run else check_base_verifier(model_path, "score", root)
    fingerprint = None if dry_run else verifier_fingerprint(
        model_path, cache_dir=cache_dir, force=force)
    if require_fingerprint and not dry_run and not fingerprint:
        raise HonestBokError(
            "score: cannot fingerprint the verifier checkpoint at %r (directory absent "
            "or empty), so the scores could not be tied to a specific base checkpoint. "
            "Refusing to score. Run where the checkpoint is present."
            % str(model_path))
    return {
        "schema_version": VERIFIER_MANIFEST_SCHEMA_VERSION,
        "model": str(model_path),
        "verifier_id": verifier_id,
        "fingerprint": fingerprint,
        "fingerprint_available": bool(fingerprint),
        "option_seed": int(option_seed),
        "score_prompt_version": score_prompt_version,
        "dry_run": bool(dry_run),
        # Sealed by seal_verifier_manifest() once the scores artifact is final.
        "complete": False,
        "scores_sha256": None,
        "scores_count": None,
    }


def seal_verifier_manifest(ws, manifest):
    """Bind the manifest to the FINAL scores artifact once scoring has completed.

    Without this, `verifier_scores.jsonl` could drift after the run that produced it
    (rows appended, edited, or partially rewritten) and still look authentic row by row.
    Recording the file's sha256 + row count, and marking the manifest complete only
    here, means select/write can prove they are reading the exact artifact the score
    stage finished with.
    """
    sealed = dict(manifest)
    sealed["scores_sha256"] = sha256_file(ws.scores)
    sealed["scores_count"] = len(read_jsonl_strict(ws.scores, "scores"))
    sealed["complete"] = True
    atomic_write_json(ws.verifier_manifest, sealed)
    return sealed


def require_verifier_manifest(ws, model_path, option_seed, score_prompt_version,
                              dry_run, stage, root=None, require_sealed=True,
                              rehash=False):
    """Re-verify at select/write that the scores came from the declared base verifier."""
    path = ws.verifier_manifest
    if not os.path.exists(path):
        raise HonestBokError(
            "%s: no verifier manifest at %s; re-run --stage score so the base-verifier "
            "identity is recorded." % (stage, path))
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != VERIFIER_MANIFEST_SCHEMA_VERSION:
        raise HonestBokError("%s: %s is not a %s" % (stage, path,
                                                     VERIFIER_MANIFEST_SCHEMA_VERSION))
    if bool(manifest.get("dry_run")) != bool(dry_run):
        raise HonestBokError("%s: verifier manifest dry_run=%r but this run is %r"
                             % (stage, manifest.get("dry_run"), dry_run))
    if manifest.get("model") != str(model_path):
        raise HonestBokError("%s: scores were produced by verifier %r, not %r"
                             % (stage, manifest.get("model"), model_path))
    if int(manifest.get("option_seed", -1)) != int(option_seed):
        raise HonestBokError("%s: verifier manifest option_seed %r != %r"
                             % (stage, manifest.get("option_seed"), option_seed))
    if manifest.get("score_prompt_version") != score_prompt_version:
        raise HonestBokError("%s: scores used prompt_version %r but this code expects %r"
                             % (stage, manifest.get("score_prompt_version"),
                                score_prompt_version))
    if require_sealed:
        if not manifest.get("complete"):
            raise HonestBokError(
                "%s: the verifier manifest is not sealed, so --stage score did not run "
                "to completion. Re-run scoring before selecting or publishing." % stage)
        recorded = manifest.get("scores_sha256")
        if not recorded:
            raise HonestBokError("%s: verifier manifest records no scores_sha256" % stage)
        actual = sha256_file(ws.scores)
        if actual != recorded:
            raise HonestBokError(
                "%s: %s has changed since scoring completed (sha256 %s != recorded %s). "
                "The scores artifact drifted; re-run --stage score."
                % (stage, ws.scores, actual[:16], str(recorded)[:16]))

    if dry_run:
        return manifest

    check_base_verifier(model_path, stage, root)
    if not manifest.get("fingerprint"):
        raise HonestBokError(
            "%s: the verifier manifest carries no checkpoint fingerprint, so the scores "
            "cannot be tied to a specific base checkpoint. Re-run --stage score on a "
            "machine where %r is present." % (stage, str(model_path)))
    # The checkpoint must still be PRESENT and match. Treating "absent" as "nothing to
    # compare" would let a deleted or emptied checkpoint sail through: the recorded
    # fingerprint could then name a model nobody can produce or audit any more, which is
    # exactly the provenance claim select/write are supposed to be able to stand behind.
    current = verifier_fingerprint(model_path, cache_dir=ws.run_dir, force=rehash)
    if current is None:
        raise HonestBokError(
            "%s: the verifier checkpoint %r is missing (or contains no "
            "checkpoint files), so the fingerprint recorded at scoring time cannot be "
            "re-verified. Restore the checkpoint, or re-score against the checkpoint "
            "you intend to attribute these scores to." % (stage, str(model_path)))
    if current != manifest.get("fingerprint"):
        raise HonestBokError(
            "%s: the checkpoint at %r does not match the fingerprint recorded when "
            "the scores were produced; it was swapped or further trained."
            % (stage, str(model_path)))
    return manifest


# ---- join keys ------------------------------------------------------------
def candidate_key(row):
    """(item_id, condition, candidate_idx) -- identical to schema.score_key."""
    return (row["item_id"], row["condition"], int(row["candidate_idx"]))


def candidate_key_from_parts(dataset_index, candidate_idx, condition=HONEST_CONDITION):
    return (common.item_id_for(dataset_index), condition, int(candidate_idx))


def candidate_sort_key(row):
    return (int(row.get("dataset_index", -1)), str(row.get("condition")),
            int(row.get("candidate_idx", -1)))


def score_sort_key(row):
    return (str(row.get("item_id")), str(row.get("condition")),
            int(row.get("candidate_idx", -1)))


# ---- seeds ----------------------------------------------------------------
# Best-of-K requires a distinct sampling seed per candidate. The schedule follows model
# call order—round-major, Debater A then Debater B—and derives every seed from candidate
# identity rather than a shared request stream.
SEED_POLICY_VERSION = "aisi-honest-bok-request-seed-v1"
_SEED_NAMESPACE = "aisi.honest_bok.request"
_SEED_MODULUS = 2 ** 31


def stable_seed(*parts):
    """Deterministic 64-bit int from stringified parts (never Python's salted hash)."""
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def model_request_seeds(base_seed, dataset_index, candidate_idx,
                        num_rounds, condition=HONEST_CONDITION):
    """Seeds in the exact order debate.debate issues model calls.

    debate.debate runs `for _ in range(num_rounds): A.take_turn(); B.take_turn()`, and
    each take_turn issues exactly one model_client.generate() -> 2*num_rounds requests.
    """
    seeds = []
    for round_index in range(int(num_rounds)):
        for debater in common.DEBATER_NAMES:
            seeds.append(
                stable_seed(_SEED_NAMESPACE, int(base_seed), int(dataset_index),
                            str(condition), int(candidate_idx), round_index, debater)
                % _SEED_MODULUS
            )
    return seeds


def seed_policy():
    return {
        "version": SEED_POLICY_VERSION,
        "derivation": "sha256-first64-mod-2**31",
        "namespace": _SEED_NAMESPACE,
        "key_fields": ["base_seed", "dataset_index", "condition", "candidate_idx",
                       "round_index_zero_based", "debater"],
        "call_order": "round-major; Debater A then Debater B",
        "api_parameter": "seed",
    }


# ---- JSONL / JSON IO ------------------------------------------------------
def read_jsonl_resume(path, warn=None):
    """Read an append-only artifact for RESUME. Tolerates exactly one thing.

    A crash mid-append can leave an unterminated final line; that line is dropped. Any
    OTHER defect -- a malformed interior line, a non-object -- aborts, because we only
    ever write complete lines, so interior damage means the file is not ours to trust.
    Downstream stages use `read_jsonl_strict`, which tolerates nothing at all.
    """
    warn = warn or (lambda msg: None)
    stats = {"lines": 0, "rows": 0, "truncated_tail": False}
    if not os.path.exists(path):
        return [], stats

    with open(path, encoding="utf-8") as f:
        raw_lines = f.readlines()

    rows = []
    last_index = len(raw_lines) - 1
    for i, raw in enumerate(raw_lines):
        line = raw.strip()
        if not line:
            continue
        stats["lines"] += 1
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            if i == last_index and not raw.endswith("\n"):
                stats["truncated_tail"] = True
                warn("%s: dropping truncated final line (interrupted append)" % path)
                continue
            raise HonestBokError(
                "%s:%d is not valid JSON (%s) and is not an interrupted final line; "
                "refusing to silently drop it." % (path, i + 1, exc))
        if not isinstance(row, dict):
            raise HonestBokError("%s:%d is not a JSON object" % (path, i + 1))
        rows.append(row)
    stats["rows"] = len(rows)
    return rows, stats


def dedupe_identical_or_fail(rows, key_fn, path):
    """Drop byte-identical crash duplicates; abort on CONFLICTING ones.

    An interrupted append can re-emit a row verbatim, which is harmless. Two rows with
    the same key but different content are not: they make the artifact ambiguous.
    """
    seen = {}
    out = []
    dropped = 0
    for position, row in enumerate(rows, 1):
        try:
            key = key_fn(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise HonestBokError("%s row %d has an unusable join key (%s)"
                                 % (path, position, exc))
        if key in seen:
            if seen[key] == row:
                dropped += 1
                continue
            raise HonestBokError(
                "%s contains CONFLICTING duplicate rows for %r; refusing to guess which "
                "is authoritative. Re-run with --overwrite." % (path, key))
        seen[key] = row
        out.append(row)
    return out, dropped


def read_jsonl_strict(path, kind):
    """Read JSONL with ZERO tolerance -- the loader for every downstream artifact.

    `read_jsonl_resume` exists for GENERATION RESUME, where an interrupted append can
    leave a half-written final line. Everything downstream of generation (score, select,
    write) must instead fail closed: silently skipping a malformed or non-object line
    would let a dirty artifact be "normalised" and then published. A pool that cannot be
    read exactly is not a pool we are willing to select from.
    """
    if not os.path.exists(path):
        raise HonestBokError("%s: %s does not exist" % (kind, path))
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                # ZERO tolerance means zero: we never emit blank lines, so one means the
                # artifact was edited. Normalising it away is exactly the silent repair
                # this reader exists to prevent.
                raise HonestBokError("%s: %s:%d is blank" % (kind, path, line_no))
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise HonestBokError(
                    "%s: %s:%d is not valid JSON (%s). Refusing to normalise a "
                    "damaged artifact; regenerate it." % (kind, path, line_no, exc))
            if not isinstance(row, dict):
                raise HonestBokError("%s: %s:%d is not a JSON object" % (kind, path, line_no))
            rows.append(row)
    if not rows:
        raise HonestBokError("%s: %s is empty" % (kind, path))
    return rows


def load_rows_strict(path, kind, validate, key_fn, expect_condition=HONEST_CONDITION):
    """Strictly load + validate + uniquely key one artifact file.

    Any schema-invalid row, wrong condition, unusable join key, or DUPLICATE key aborts.
    Duplicates are never de-duplicated here: two rows claiming the same
    (item_id, condition, candidate_idx) make the artifact ambiguous, and picking "the
    first" would silently decide which transcript gets published.
    """
    rows = read_jsonl_strict(path, kind)
    by_key = {}
    for position, row in enumerate(rows, 1):
        errors = validate(row)
        if errors:
            raise HonestBokError(
                "%s: %s row %d failed validation: %s" % (kind, path, position, errors[0]))
        if expect_condition is not None and row.get("condition") != expect_condition:
            raise HonestBokError(
                "%s: %s row %d has condition %r, expected %r"
                % (kind, path, position, row.get("condition"), expect_condition))
        try:
            key = key_fn(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise HonestBokError("%s: %s row %d has an unusable join key (%s)"
                                 % (kind, path, position, exc))
        if key in by_key:
            previous = by_key[key]
            same = previous == row
            raise HonestBokError(
                "%s: %s contains a DUPLICATE row for %r (%s). Refusing to guess which "
                "one is authoritative; regenerate the artifact."
                % (kind, path, key,
                   "byte-identical copies" if same else "CONFLICTING contents"))
        by_key[key] = row
    return rows, by_key


class JsonlAppender:
    """Thread-safe append-only JSONL writer with per-record durability."""

    def __init__(self, path, fsync=True):
        self.path = path
        self._fsync = fsync
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(parent, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self.count = 0

    def append(self, row):
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            self.count += 1

    def close(self):
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.flush()
                if self._fsync:
                    os.fsync(self._fh.fileno())
                self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _render_json(obj, indent=2):
    def render(f):
        json.dump(obj, f, ensure_ascii=False, indent=indent)

    return render


def _render_jsonl(rows):
    def render(f):
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    return render


def _atomic_write(path, render):
    """tempfile in the destination dir + os.replace (mirrors common._atomic_write)."""
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            render(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_json(path, obj, indent=2):
    _atomic_write(path, _render_json(obj, indent))


def atomic_write_jsonl(path, rows):
    _atomic_write(path, _render_jsonl(rows))


def stage_file(path, render, suffix=".staged"):
    """Write `path + suffix` durably WITHOUT publishing it. Returns the staged path."""
    staged = path + suffix
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    with open(staged, "w", encoding="utf-8") as f:
        render(f)
        f.flush()
        os.fsync(f.fileno())
    return staged



def canonicalize_jsonl(path, key_fn, sort_key_fn, warn=None):
    """De-duplicate + sort + atomically rewrite an append-only artifact."""
    rows, stats = read_jsonl_resume(path, warn=warn)
    rows, dropped = dedupe_identical_or_fail(rows, key_fn, path)
    rows.sort(key=sort_key_fn)
    atomic_write_jsonl(path, rows)
    stats["duplicates_dropped"] = dropped
    stats["final_rows"] = len(rows)
    return rows, stats


# ---- dataset loading / preflight -------------------------------------------
def load_dataset_items(path):
    return common.load_items(path)


def load_story_map(path):
    return common.load_story_map(path)


def check_items_are_no_debate(items, path):
    """Fail closed unless the items file is the transcript-free generation input.

    The stance-parity argument depends on common.resolve_stance falling back to the
    deterministic stance RNG (which reproduces debate.stance_rng, and therefore the
    stance already baked into every existing adversarial transcript). If the items file
    carried Q_Y.transcript, ``resolve_stance`` would prefer it while ``debate.debate``
    would draw from ``stance_rng``, producing inconsistent stance assignments.
    """
    offenders = []
    for i, ex in enumerate(items):
        q_y = ex.get("Q_Y")
        if isinstance(q_y, dict) and q_y.get("transcript") is not None:
            offenders.append(i)
            if len(offenders) >= 5:
                break
    if offenders:
        raise HonestBokError(
            "%s carries Q_Y.transcript at index %s: honest generation must read the "
            "transcript-free <DS>-no-debate.json so the stance falls back to the "
            "deterministic stance RNG." % (path, offenders))


def expected_stance(example):
    """The stance debate.debate WILL draw for this example, as a plain dict."""
    resolved = common.resolve_stance(example)
    if resolved.get("source") != "stance_rng":
        raise HonestBokError(
            "expected the deterministic stance fallback, got source=%r; the items file "
            "must be <DS>-no-debate.json" % (resolved.get("source"),))
    return {"Debater A": resolved["Debater A"], "Debater B": resolved["Debater B"]}


def transcript_stance(transcript):
    return {"Debater A": transcript.get("Debater A"), "Debater B": transcript.get("Debater B")}


def iter_indices(items, start_index=0, limit=None):
    end = len(items) if limit is None else min(len(items), start_index + limit)
    return range(start_index, end)


# ---- pool completeness ----------------------------------------------------
def check_pool_complete(items, candidates, k, stage):
    """Every item must carry EXACTLY candidate_idx 0..k-1, joined uniquely.

    A K=1 pool, a pool missing an item, or a pool with duplicate/foreign indices is a
    silently biased best-of-K, so it is a hard error rather than a warning.
    """
    by_index = {}
    for candidate in candidates:
        index = candidate.get("dataset_index")
        if not isinstance(index, int) or not 0 <= index < len(items):
            raise HonestBokError("%s: candidate with out-of-range dataset_index %r"
                                 % (stage, index))
        if candidate.get("item_id") != common.item_id_for(index):
            raise HonestBokError(
                "%s: candidate item_id %r disagrees with dataset_index %d"
                % (stage, candidate.get("item_id"), index))
        idx = candidate.get("candidate_idx")
        if not isinstance(idx, int):
            raise HonestBokError("%s: non-integer candidate_idx %r" % (stage, idx))
        bucket = by_index.setdefault(index, set())
        if idx in bucket:
            raise HonestBokError("%s: duplicate candidate_idx %d for item %d"
                                 % (stage, idx, index))
        bucket.add(idx)

    wanted = set(range(int(k)))
    incomplete = []
    for index in range(len(items)):
        got = by_index.get(index, set())
        if got != wanted:
            incomplete.append({"dataset_index": index,
                               "missing": sorted(wanted - got),
                               "unexpected": sorted(got - wanted)})
    if incomplete:
        raise HonestBokError(
            "%s: best-of-K pool is incomplete for %d/%d item(s) at k=%d (first: %s). "
            "Selection over a partial pool is a biased best-of-K; finish generation "
            "or regenerate." % (stage, len(incomplete), len(items), k, incomplete[:5]))


# ---- honest selection record schema ---------------------------------------
_SELECTION_GATE_KEYS = ("total", "no_score", "drop_qy_gate", "survivors",
                        "fallback_used")


def validate_honest_selection(row, k=None):
    """Runtime validator for one selected.jsonl row (see schemas/honest_selection.schema.json)."""
    errs = []
    if not isinstance(row, dict):
        return ["selection: must be an object"]
    for key in ("schema_version", "item_id", "dataset_index", "story_title", "status",
                "selector", "honest_selected", "gates", "audits", "readout"):
        if key not in row:
            errs.append("selection: missing required key %r" % key)
    if row.get("schema_version") != SELECTION_SCHEMA_VERSION:
        errs.append("selection.schema_version: must be %r" % SELECTION_SCHEMA_VERSION)
    if row.get("status") not in SELECTION_STATUSES:
        errs.append("selection.status: must be one of %r" % (SELECTION_STATUSES,))
    if not isinstance(row.get("dataset_index"), int):
        errs.append("selection.dataset_index: must be an integer")
    elif row.get("item_id") != common.item_id_for(row["dataset_index"]):
        errs.append("selection.item_id: disagrees with dataset_index")

    gates = row.get("gates")
    if not isinstance(gates, dict):
        errs.append("selection.gates: must be an object")
    else:
        for key in _SELECTION_GATE_KEYS:
            if key not in gates:
                errs.append("selection.gates: missing %r" % key)
        if k is not None and gates.get("total") != int(k):
            errs.append("selection.gates.total: must equal k=%d, got %r"
                        % (int(k), gates.get("total")))

    audits = row.get("audits")
    if not isinstance(audits, list):
        errs.append("selection.audits: must be a list")
    elif k is not None and len(audits) != int(k):
        errs.append("selection.audits: must carry one entry per candidate (k=%d), got %d"
                    % (int(k), len(audits)))

    readout = row.get("readout")
    if not isinstance(readout, dict):
        errs.append("selection.readout: must be an object")

    selected = row.get("honest_selected")
    status = row.get("status")
    if status in (STATUS_OK, STATUS_FALLBACK):
        if not isinstance(selected, dict):
            errs.append("selection.honest_selected: must be an object when status=%r"
                        % status)
        else:
            errs += _validate_selected_candidate(selected, row)
    elif status == STATUS_NONE and selected is not None:
        errs.append("selection.honest_selected: must be null when status=%r" % status)
    return errs


def _validate_selected_candidate(selected, row):
    from adversarial_transcript import schema as _schema

    errs = _schema.validate_candidate(selected)
    errs = ["selection.honest_selected: %s" % e for e in errs]
    if selected.get("condition") != HONEST_CONDITION:
        errs.append("selection.honest_selected.condition: must be 'honest'")
    if selected.get("dataset_index") != row.get("dataset_index"):
        errs.append("selection.honest_selected.dataset_index: must match the row")
    if selected.get("story_title") != row.get("story_title"):
        errs.append("selection.honest_selected.story_title: must match the row")
    score = selected.get("score")
    if not isinstance(score, dict):
        errs.append("selection.honest_selected.score: must be the joined score object")
        return errs
    errs += ["selection.honest_selected.score: %s" % e
             for e in _schema.validate_score(score)]
    if _schema.score_key(score) != candidate_key(selected):
        errs.append("selection.honest_selected.score: join key does not match the candidate")
    readout = row.get("readout") or {}
    if readout.get("honest_p_htrue") is None:
        errs.append("selection.readout.honest_p_htrue: required when a candidate is selected")
    return errs


# ---- two-file publication transaction --------------------------------------
def _backup_path(path):
    return path + ".honest-bok.bak"


def publication_targets(ws):
    """The ONLY two files a publication transaction may ever touch."""
    return (ws.with_honest_out, ws.canonical_out)


def _validate_txn_marker(ws, txn):
    """Re-derive every path in a marker before acting on it.

    A marker is just a JSON file on disk. Trusting the paths inside it would let a
    stale or hand-crafted marker make recovery delete or overwrite an arbitrary file,
    so nothing from the marker is used as a path: the entries must match the paths this
    code derives for THIS workspace, exactly and completely.
    """
    if not isinstance(txn, dict) or txn.get("schema_version") != TXN_SCHEMA_VERSION:
        raise HonestBokError("%s is not a %s marker" % (ws.txn_marker, TXN_SCHEMA_VERSION))
    if txn.get("dataset") != ws.dataset:
        raise HonestBokError(
            "%s belongs to dataset %r, not %r; refusing to act on a foreign marker"
            % (ws.txn_marker, txn.get("dataset"), ws.dataset))
    if txn.get("mode") != ws.mode:
        raise HonestBokError(
            "%s was written in mode %r but this run is mode %r"
            % (ws.txn_marker, txn.get("mode"), ws.mode))
    if ws.mode == MODE_PRODUCTION:
        assert_production_layout(ws)

    entries = txn.get("targets")
    if not isinstance(entries, list):
        raise HonestBokError("%s has no targets list" % ws.txn_marker)
    expected = publication_targets(ws)
    seen = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise HonestBokError("%s has a malformed target entry" % ws.txn_marker)
        path = entry.get("path")
        if not isinstance(path, str):
            raise HonestBokError("%s has a target with no path" % ws.txn_marker)
        match = next((t for t in expected
                      if os.path.realpath(t) == os.path.realpath(path)), None)
        if match is None:
            raise HonestBokError(
                "%s names %r, which is not one of this dataset's publication targets "
                "%s; refusing to touch it." % (ws.txn_marker, path, list(expected)))
        if match in seen:
            raise HonestBokError("%s names %r twice" % (ws.txn_marker, path))
        if entry.get("backup") != _backup_path(match):
            raise HonestBokError(
                "%s target %r has backup %r, expected %r"
                % (ws.txn_marker, path, entry.get("backup"), _backup_path(match)))
        if entry.get("staged") != match + ".staged":
            raise HonestBokError(
                "%s target %r has staged %r, expected %r"
                % (ws.txn_marker, path, entry.get("staged"), match + ".staged"))
        if not isinstance(entry.get("existed_before"), bool):
            raise HonestBokError("%s target %r has a non-boolean existed_before"
                                 % (ws.txn_marker, path))
        if not entry.get("new_sha256"):
            raise HonestBokError("%s target %r records no new_sha256" % (ws.txn_marker, path))
        if entry["existed_before"] and not entry.get("original_sha256"):
            raise HonestBokError(
                "%s target %r claims it existed before but records no original_sha256"
                % (ws.txn_marker, path))
        seen[match] = dict(entry, path=match)
    missing = [t for t in expected if t not in seen]
    if missing:
        raise HonestBokError("%s does not cover %s" % (ws.txn_marker, missing))

    # For an IN-PLACE publication the canonical dataset always exists beforehand, so a
    # marker claiming otherwise is untrustworthy -- and acting on it would delete the
    # real dataset. (A sandbox legitimately creates its copy, and that case is instead
    # proved by new_sha256 before anything is removed.)
    canonical_entry = seen[ws.canonical_out]
    if ws.publishes_in_place and not canonical_entry["existed_before"]:
        raise HonestBokError(
            "%s claims %s did not exist before the transaction, but an in-place "
            "publication always updates an existing canonical dataset. This marker "
            "cannot be trusted; refusing to touch anything."
            % (ws.txn_marker, ws.canonical_out))
    # Return in the canonical commit order, not the marker's order.
    return [seen[t] for t in expected]


def recover_publication(ws, warn=None):
    """Deterministic recovery for a crash between the two commits.

    The marker is written BEFORE the first os.replace and removed only after both
    succeed, so finding one means the pair may be inconsistent. Recovery always rolls
    BACK to the pre-transaction state: a half-published dataset must never be treated
    as a completed run.
    """
    warn = warn or (lambda msg: None)
    if not os.path.exists(ws.txn_marker):
        return None
    with open(ws.txn_marker, encoding="utf-8") as f:
        txn = json.load(f)
    entries = _validate_txn_marker(ws, txn)

    # PLAN the whole rollback and validate every precondition BEFORE mutating anything.
    # A rollback that gets halfway and then discovers a missing backup would be worse
    # than the inconsistency it is trying to repair.
    plan = []
    for entry in entries:
        target, backup = entry["path"], entry["backup"]
        if entry["existed_before"]:
            if not os.path.exists(backup):
                raise HonestBokError(
                    "%s says %s existed before the transaction, but its backup %s is "
                    "missing, so the original content cannot be restored. Refusing to "
                    "touch anything -- restore the backup or repair the pair by hand."
                    % (ws.txn_marker, target, backup))
            if sha256_file(backup) != entry["original_sha256"]:
                raise HonestBokError(
                    "%s: backup %s does not match the original_sha256 recorded in the "
                    "marker; it is not the content this transaction replaced. Refusing "
                    "to restore it." % (ws.txn_marker, backup))
            plan.append(("restore", target, backup))
        elif os.path.exists(target):
            # The marker says we created this file. Only remove it if its content is
            # provably what THIS transaction wrote; otherwise it predates us and
            # deleting it would destroy someone else's data.
            if sha256_file(target) != entry["new_sha256"]:
                raise HonestBokError(
                    "%s says %s was created by this transaction, but its content does "
                    "not match the new_sha256 the transaction was committing. Refusing "
                    "to delete a file this transaction cannot prove it created; inspect "
                    "and remove the marker by hand." % (ws.txn_marker, target))
            plan.append(("remove", target, None))

    restored = []
    for action, target, backup in plan:
        if action == "restore":
            shutil.copyfile(backup, target)
        else:
            os.remove(target)
        restored.append(target)
    for entry in entries:
        for leftover in (entry.get("staged"), entry.get("backup")):
            if leftover and os.path.exists(leftover):
                os.remove(leftover)
    os.remove(ws.txn_marker)
    warn("rolled back an interrupted publication; restored %s" % (restored,))
    return {"rolled_back": restored}


def publish_transaction(ws, targets, verify, warn=None):
    """Commit several JSON files as one unit, or leave every one of them untouched.

    `targets` is [(path, obj), ...]; `verify` is called with {path: reloaded_obj} after
    both files are on disk and must raise to trigger a rollback. Sequence:

      stage -> read back + verify staged content -> backup -> marker -> replace all
      -> read back + verify committed content -> drop marker

    Any failure after the first replace restores every target to its pre-transaction
    state, so the two dataset files can never be left describing different runs.
    """
    warn = warn or (lambda msg: None)
    recover_publication(ws, warn)
    ws.ensure_dirs()

    # The transaction may only ever touch this dataset's two locked publication targets.
    expected = publication_targets(ws)
    supplied = tuple(path for path, _ in targets)
    if supplied != expected:
        raise HonestBokError(
            "publish_transaction may only write %s, got %s"
            % (list(expected), list(supplied)))

    # When publishing IN PLACE (production), the canonical dataset is being updated, so
    # it must already exist -- and recovery may then rely on that. A sandbox publishes a
    # fresh copy instead, where first-creation is normal and rollback is proved by the
    # committed-content digest rather than by a backup.
    if ws.publishes_in_place and not os.path.exists(ws.canonical_out):
        raise HonestBokError(
            "publish_transaction: %s does not exist. An in-place publication updates the "
            "canonical dataset, so its absence means the workspace is wrong."
            % ws.canonical_out)

    entries = []
    for path, obj in targets:
        staged = stage_file(path, _render_json(obj))
        with open(staged, encoding="utf-8") as f:
            reloaded = json.load(f)
        if reloaded != obj:
            for entry in entries:
                os.remove(entry["staged"])
            os.remove(staged)
            raise HonestBokError("staged %s did not read back identically" % path)
        existed = os.path.exists(path)
        entries.append({
            "path": path, "staged": staged, "backup": _backup_path(path),
            "existed_before": existed,
            # Digests make rollback provable: `original_sha256` is what we must be able
            # to restore, `new_sha256` is what we are committing (so recovery can tell a
            # file THIS transaction created from one that merely happens to be there).
            "original_sha256": sha256_file(path) if existed else None,
            "new_sha256": sha256_file(staged),
        })

    # Pre-commit verification on the STAGED content: nothing is published yet.
    try:
        staged_content = {}
        for entry in entries:
            with open(entry["staged"], encoding="utf-8") as f:
                staged_content[entry["path"]] = json.load(f)
        verify(staged_content)
    except BaseException:
        for entry in entries:
            if os.path.exists(entry["staged"]):
                os.remove(entry["staged"])
        raise

    for entry in entries:
        if entry["existed_before"]:
            shutil.copyfile(entry["path"], entry["backup"])

    atomic_write_json(ws.txn_marker, {
        "schema_version": TXN_SCHEMA_VERSION,
        "dataset": ws.dataset,
        "mode": ws.mode,
        "targets": entries,
    })

    committed = []
    try:
        for entry in entries:
            os.replace(entry["staged"], entry["path"])
            committed.append(entry)
        reloaded = {}
        for entry in entries:
            with open(entry["path"], encoding="utf-8") as f:
                reloaded[entry["path"]] = json.load(f)
        verify(reloaded)
    except BaseException as exc:
        for entry in entries:
            if entry["existed_before"] and os.path.exists(entry["backup"]):
                shutil.copyfile(entry["backup"], entry["path"])
            elif not entry["existed_before"] and os.path.exists(entry["path"]):
                os.remove(entry["path"])
            if os.path.exists(entry["staged"]):
                os.remove(entry["staged"])
        if os.path.exists(ws.txn_marker):
            os.remove(ws.txn_marker)
        warn("publication rolled back after %s: %s" % (type(exc).__name__, exc))
        raise
    os.remove(ws.txn_marker)
    # Backups exist ONLY to roll this transaction back. Once both targets are committed
    # and verified they are dead weight, and leaving them behind would drop non-contract
    # files (<DS>.json.honest-bok.bak) into the canonical dataset directory forever.
    for entry in entries:
        if os.path.exists(entry["backup"]):
            os.remove(entry["backup"])
    return [entry["path"] for entry in entries]


# ---- small CLI helpers ------------------------------------------------------
def positive_int(value):
    import argparse

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer (>= 1)")
    return parsed


def nonnegative_int(value):
    import argparse

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def stderr_warn(msg):
    print("[WARN] %s" % msg, file=sys.stderr)
