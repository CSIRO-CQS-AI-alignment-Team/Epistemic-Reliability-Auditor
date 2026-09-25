"""Generate and persist K honest debate candidates per dataset item.

The Section 6 mainline uses ``K=8`` three-round candidates for QuALITY-H and GPQA.
``honest_select.py`` scores the resulting pool with the frozen base verifier, applies
the transcript-conditioned ``q_m=Y_true`` gate, and selects the largest
``P(H_true | q_h, T)`` among eligible candidates.

Production paths are derived from the dataset selector:

* input: ``dataset/<DATASET>/<stem>-no-debate.json``;
* output: ``runs/honest/<DATASET>/candidate.jsonl``;
* endpoint: ``http://localhost:18888/v1``.

The run manifest binds the dataset inputs, model, prompt, seed schedule, ``K``, number of
rounds, and transcript budget. Resume accepts only candidates matching that identity.
``--dry-run`` uses the isolated ``runs/honest/<DATASET>/dryrun/`` subtree.

Examples:
    python3 debate-bok.py --dataset GPQA --dry-run --limit 3 --k 2
    python3 debate-bok.py --dataset QuALITY-H --k 8 --concurrency 16
    python3 debate-bok.py --dataset QuALITY-H-50 --transcript-len 50
    python3 debate-bok.py --dataset QuALITY-H-250 --transcript-len 250
"""

import argparse
import concurrent.futures
import copy
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import honest_bok_io as hio  # noqa: E402
from adversarial_transcript import schema  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; the pipeline must run without it
    tqdm = None

PROMPT_VERSION = hio.GENERATION_PROMPT_VERSION  # shared generation contract
DEFAULT_CONCURRENCY = 16
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF = 2.0
MIN_TRANSCRIPT_LEN = 50

# Exception TYPE NAMES worth retrying. Matched by name because the openai SDK is a
# lazy import and its exception classes must not be imported at module scope.
_RETRYABLE_EXC_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "APIConnectionTimeoutError",
    "RateLimitError", "InternalServerError", "ConnectionError", "Timeout",
    "ReadTimeout", "ConnectTimeout", "RemoteProtocolError",
})


# ---- model client ---------------------------------------------------------
def _is_retryable(exc):
    """Transient transport/server failures only; 4xx other than 429 is fatal."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(exc).__name__ in _RETRYABLE_EXC_NAMES


class VLLMChatClient:
    """Thread-safe vLLM chat client with a per-request seed and bounded retries.

    `debate.ApiModelClient` cannot be reused directly: it exposes no `seed` parameter
    (mandatory for best-of-K diversity) and no retry policy for a multi-hour run.
    Sampling parameters come from ``debate.LM_CONFIG`` so all generation paths use the
    same YAML configuration.
    """

    def __init__(self, base_url, api_key, model, lm_config,
                 max_retries=DEFAULT_MAX_RETRIES, backoff=DEFAULT_RETRY_BACKOFF,
                 sleep=None, rng=None):
        from openai import OpenAI  # lazy: importing this module must not require openai

        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.base_url = base_url
        self.model = model
        self.lm = lm_config
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self._sleep = sleep or time.sleep
        self._rng = rng or random.Random(0)

    def complete(self, messages, seed):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.lm["temperature"],
                    top_p=self.lm["top_p"],
                    max_tokens=self.lm["max_tokens"],
                    timeout=self.lm["timeout"],
                    seed=seed,
                )
                # Length/filter/tool finishes can yield None content (mirrors
                # debate.ApiModelClient); never return None into the transcript.
                return completion.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - re-raised below when fatal
                last_exc = exc
                if attempt >= self.max_retries or not _is_retryable(exc):
                    raise
                self._sleep((self.backoff ** attempt) * (1.0 + self._rng.random()))
        raise last_exc  # unreachable; keeps static analysis honest


class SeededTurnClient:
    """One-candidate adapter exposing debate.BaseModelClient's `generate(messages)`.

    Thread-confined: each candidate task builds its own instance over the SHARED
    VLLMChatClient and consumes its pre-computed seed plan in order.
    `require_complete` fails the candidate if debate.py issued a different number of
    calls than the plan predicted -- i.e. if the debate loop's structure ever changes.
    """

    def __init__(self, client, seeds):
        self._client = client
        self._seeds = list(seeds)
        self._position = 0

    def generate(self, messages):
        if self._position >= len(self._seeds):
            raise hio.HonestBokError(
                "debate made more model calls (%d) than the seed plan (%d); the "
                "request-seed policy must be updated before continuing"
                % (self._position + 1, len(self._seeds)))
        seed = self._seeds[self._position]
        self._position += 1
        return self._client.complete(messages, seed)

    def require_complete(self):
        if self._position != len(self._seeds):
            raise hio.HonestBokError(
                "debate made %d model call(s) but the seed plan had %d; refusing to "
                "record a candidate with unverifiable sampling provenance"
                % (self._position, len(self._seeds)))


# ---- endpoint preflight ---------------------------------------------------
def probe_endpoint(base_url, model=None, timeout_s=10):
    """GET {base_url}/models and confirm the requested model is served."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise hio.HonestBokError(
            "vLLM endpoint %s is not reachable (%s: %s). Start the server, or pass "
            "--dry-run / --skip-endpoint-check." % (url, type(exc).__name__, exc))
    served = [entry.get("id") for entry in payload.get("data", []) or []]
    if model is not None and not any(model == sid or model in str(sid) for sid in served):
        raise hio.HonestBokError(
            "vLLM endpoint %s serves %r, which does not include --model %r"
            % (url, served, model))
    return served


# ---- candidate construction ------------------------------------------------
def _copy_qy(qy):
    return {k: qy[k] for k in ("question", "Y_true", "Y_false") if k in qy}


def _copy_qh(qh):
    return {k: qh[k] for k in ("question", "H_true", "H_false") if k in qh}


def dry_run_transcript(example, stance, candidate_idx, num_rounds):
    """Deterministic placeholder transcript for the sandbox rehearsal."""
    qy = example.get("Q_Y", {})
    title = example.get("story_title") or "the story"
    answers = {"Y_true": qy.get("Y_true"), "Y_false": qy.get("Y_false")}
    rounds = []
    for r in range(num_rounds):
        rounds.append({
            name: ("Round %d: %s argues honestly that %r answers the visible question "
                   "about %s (dry-run candidate %d)."
                   % (r + 1, name, answers.get(stance[name]), title, candidate_idx))
            for name in ("Debater A", "Debater B")
        })
    return {"Debater A": stance["Debater A"], "Debater B": stance["Debater B"],
            "rounds": rounds}


def build_candidate(example, dataset_index, candidate_idx, transcript, config,
                    request_seeds):
    """Assemble one honest candidate and validate it against the shared schema."""
    identity = config.identity
    candidate = {
        "item_id": hio.common.item_id_for(dataset_index),
        "dataset_index": int(dataset_index),
        "story_title": example.get("story_title"),
        "condition": hio.HONEST_CONDITION,
        "candidate_idx": int(candidate_idx),
        # Honest candidates have no compromised debater and no hidden target; the
        # shared validator enforces that both are null for condition="honest".
        "compromised_debater": None,
        "qh_target": None,
        "transcript": transcript,
        "qy": _copy_qy(example.get("Q_Y", {})),
        "qh": _copy_qh(example.get("Q_H", {})),
        "gen_meta": {
            "prompt_version": identity["prompt_version"],
            "backend": identity["backend"],
            "base_url": identity["base_url"],
            "model": identity["model"],
            "seed": identity["seed"],
            "num_rounds": identity["num_rounds"],
            "dry_run": identity["dry_run"],
            "stance_source": "stance_rng",
            "selector_arm": "honest_bok",
            "seed_policy_version": identity["seed_policy_version"],
            "model_request_seeds": list(request_seeds),
            # Binds this row to one run identity; every downstream stage re-checks it.
            "manifest_digest": config.digest,
            "candidate_seed": "%s|%s|%s|%s" % (identity["seed"], dataset_index,
                                               hio.HONEST_CONDITION, candidate_idx),
            "lm_config": config.lm_config,
        },
    }
    for field in ("transcript_budget", "debate_config_sha256"):
        if field in identity:
            candidate["gen_meta"][field] = identity[field]
    errors = schema.validate_candidate(candidate)
    if errors:
        raise hio.HonestBokError("invalid generated candidate %s#%d: %s"
                                 % (candidate["item_id"], candidate_idx, errors[0]))
    return candidate


def debate_transcript(example, story, turn_client, num_rounds, debate_config=None,
                      marker_within_limit=False):
    """Run one honest debate through debate.debate (the production transcript source).

    `turn_client` already carries this candidate's seed plan, so the seed accounting is
    identical no matter which transcript source is used.
    """
    import debate  # lazy: debate.py imports tqdm + reads YAML at import time

    q_y = example["Q_Y"]
    # A FRESH stance_rng per candidate: random.Random(key) is seeded from stable
    # content, so every candidate of this item draws the identical stance.
    return debate.debate(q_y["question"], q_y["Y_true"], q_y["Y_false"], story,
                         turn_client, debate.stance_rng(example),
                         num_rounds=num_rounds, config=debate_config,
                         marker_within_limit=marker_within_limit)


def assert_stance_parity(transcript, example, dataset_index):
    """The generated stance must equal the deterministic stance the adversarial arm used."""
    want = hio.expected_stance(example)
    got = hio.transcript_stance(transcript)
    if got != want:
        raise hio.HonestBokError(
            "stance parity violated at index %d: debate produced %r but the "
            "deterministic stance is %r" % (dataset_index, got, want))


# ---- run configuration -----------------------------------------------------
class GenerationConfig:
    """Everything the generation run is pinned to, plus its manifest identity."""

    def __init__(self, ws, k, num_rounds, model, base_url, api_key, seed, dry_run,
                 concurrency, max_retries, start_index, limit, transcript_len,
                 lm_config=None):
        self.ws = ws
        self.k = int(k)
        self.num_rounds = int(num_rounds)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.seed = int(seed)
        self.dry_run = bool(dry_run)
        self.concurrency = int(concurrency)
        self.max_retries = int(max_retries)
        self.start_index = int(start_index)
        self.limit = limit
        self.transcript_budget = hio.transcript_budget_for_dataset(ws.dataset)
        if int(transcript_len) != self.transcript_budget["hard_limit"]:
            raise hio.HonestBokError(
                "dataset %s is locked to --transcript-len %d, got %d; use the matching "
                "QuALITY-H length variant so experimental artifacts cannot overwrite "
                "another transcript-budget arm."
                % (ws.dataset, self.transcript_budget["hard_limit"], int(transcript_len)))
        self.transcript_len = int(transcript_len)
        self.strict_transcript_budget = ws.dataset in hio.TRANSCRIPT_BUDGETS
        config_rel = self.transcript_budget["config_path"]
        self.debate_config_path = os.path.join(_REPO_ROOT, config_rel)
        if self.dry_run:
            # Preserve the original dependency-light rehearsal: no PyYAML/debate.py.
            self.debate_config = None
            self.lm_config = dict(lm_config) if lm_config is not None else {"dry_run": True}
        else:
            self.debate_config = load_debate_config(
                self.debate_config_path, self.transcript_budget)
            # `lm_config` remains a TEST-ONLY injection point.  The prompt/limit config
            # is loaded and validated, then only its language-model block is replaced.
            if lm_config is not None:
                self.debate_config = copy.deepcopy(self.debate_config)
                self.debate_config["language_model"] = dict(lm_config)
            self.lm_config = dict(self.debate_config["language_model"])
        budget_identity = self.transcript_budget if self.strict_transcript_budget else None
        config_sha256 = (
            hio.sha256_file(self.debate_config_path)
            if self.strict_transcript_budget else None)
        self.identity = hio.build_identity(
            ws, k=self.k, num_rounds=self.num_rounds,
            backend="dry-run" if self.dry_run else "api",
            base_url=None if self.dry_run else self.base_url,
            model=None if self.dry_run else self.model,
            seed=self.seed, dry_run=self.dry_run, prompt_version=PROMPT_VERSION,
            lm_config=self.lm_config, transcript_budget=budget_identity,
            debate_config_sha256=config_sha256)
        self.digest = hio.identity_digest(self.identity)

    def manifest(self):
        return hio.build_manifest(
            self.ws, self.identity,
            scope={"start_index": self.start_index, "limit": self.limit,
                   "concurrency": self.concurrency,
                   "transcript_budget": self.transcript_budget},
            lm_config=self.lm_config)


def load_debate_config(path, budget):
    """Load and fail-closed validate the checked-in config for one length arm."""
    try:
        import yaml  # lazy: importing debate-bok.py itself stays stdlib-only
    except ImportError as exc:
        raise hio.HonestBokError(
            "PyYAML is required for real generation; activate the project environment "
            "or install its declared dependencies") from exc

    try:
        with open(path, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        raise hio.HonestBokError(
            "cannot load transcript-length config %s (%s: %s)"
            % (path, type(exc).__name__, exc))
    if not isinstance(config, dict):
        raise hio.HonestBokError("%s must contain a YAML mapping" % path)
    lm = config.get("language_model")
    prompts = config.get("prompts")
    if not isinstance(lm, dict) or not isinstance(prompts, dict):
        raise hio.HonestBokError(
            "%s must contain language_model and prompts mappings" % path)
    found_hard = lm.get("max_words")
    found_soft = prompts.get("word_limit")
    if found_hard != budget["hard_limit"] or found_soft != budget["soft_limit"]:
        raise hio.HonestBokError(
            "%s declares hard/soft word limits %r/%r, expected %d/%d for this dataset"
            % (path, found_hard, found_soft, budget["hard_limit"],
               budget["soft_limit"]))
    minimum = lm.get("min_words", 0)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise hio.HonestBokError("%s language_model.min_words must be nonnegative" % path)
    if minimum > found_hard:
        raise hio.HonestBokError(
            "%s language_model.min_words=%d exceeds max_words=%d"
            % (path, minimum, found_hard))
    return config


def transcript_len_arg(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be an integer")
    if parsed < MIN_TRANSCRIPT_LEN:
        raise argparse.ArgumentTypeError(
            "must be at least %d (the soft-limit rule is undefined below 50)"
            % MIN_TRANSCRIPT_LEN)
    return parsed


def soft_transcript_limit(hard_limit):
    """The experiment's fixed hard-to-soft budget rule."""
    if hard_limit < MIN_TRANSCRIPT_LEN:
        raise hio.HonestBokError(
            "transcript hard limit must be at least %d" % MIN_TRANSCRIPT_LEN)
    return MIN_TRANSCRIPT_LEN if hard_limit == MIN_TRANSCRIPT_LEN else hard_limit - 50


def resolve_transcript_len(dataset, requested):
    budget = hio.transcript_budget_for_dataset(dataset)
    hard_limit = budget["hard_limit"] if requested is None else int(requested)
    expected_soft = soft_transcript_limit(hard_limit)
    if hard_limit == budget["hard_limit"] and budget["soft_limit"] != expected_soft:
        raise hio.HonestBokError(
            "internal transcript-budget contract for %s has soft_limit=%d; the rule "
            "requires %d for hard_limit=%d"
            % (dataset, budget["soft_limit"], expected_soft, hard_limit))
    if hard_limit != budget["hard_limit"]:
        raise hio.HonestBokError(
            "--dataset %s is paired with --transcript-len %d, not %d. Use "
            "QuALITY-H-50 for 50 or QuALITY-H-250 for 250 so each arm writes to its "
            "own runs/honest/<dataset>/ directory."
            % (dataset, budget["hard_limit"], hard_limit))
    return hard_limit


# ---- resume ----------------------------------------------------------------
def precheck_existing_pool(ws, args, dry_run, transcript_len):
    """Cheap provenance pre-check before loading the full debate config.

    Discovering "this pool is a dry-run pool" should not require parsing YAML, and a
    config failure must never mask the real refusal reason, so obvious mismatches are
    reported first.
    """
    manifest = hio.read_manifest(ws.manifest)
    if manifest is None:
        return
    identity = manifest["identity"]
    expected = {"dry_run": bool(dry_run), "k": int(args.k),
                "num_rounds": int(args.num_rounds),
                "backend": "dry-run" if dry_run else "api",
                "seed": int(args.seed),
                "model": None if dry_run else args.model,
                "prompt_version": PROMPT_VERSION}
    if ws.dataset in hio.TRANSCRIPT_BUDGETS:
        budget = hio.transcript_budget_for_dataset(ws.dataset)
        if transcript_len != budget["hard_limit"]:
            raise hio.HonestBokError(
                "internal precheck budget mismatch for %s" % ws.dataset)
        expected["transcript_budget"] = budget
        expected["debate_config_sha256"] = hio.sha256_file(
            os.path.join(_REPO_ROOT, budget["config_path"]))
    differences = [{"field": key, "expected": value, "found": identity.get(key)}
                   for key, value in expected.items() if identity.get(key) != value]
    if differences:
        raise hio.HonestBokError(
            "refusing to resume: the existing pool at %s was generated under a "
            "DIFFERENT run identity. Differences: %s. Re-run with --overwrite to "
            "replace the mismatched pool, or restore its recorded settings."
            % (ws.candidates, json.dumps(differences, ensure_ascii=False)))


def reusable_candidates(ws, config, warn):
    """Return candidates whose manifest identity matches the requested run.

    Dry-run, model, seed, prompt, and input mismatches produce a different identity and
    are rejected rather than reused or published.
    """
    rows, stats = hio.read_jsonl_resume(ws.candidates, warn=warn)
    rows, duplicates = hio.dedupe_identical_or_fail(rows, hio.candidate_key, ws.candidates)
    existing_manifest = hio.read_manifest(ws.manifest)

    if not rows and existing_manifest is None:
        return [], stats, duplicates

    if existing_manifest is None:
        raise hio.HonestBokError(
            "%s holds %d candidate(s) but %s is missing, so their provenance cannot be "
            "verified. Re-run with --overwrite to discard them."
            % (ws.candidates, len(rows), ws.manifest))

    if existing_manifest["identity_digest"] != config.digest:
        raise hio.HonestBokError(
            "refusing to resume: the existing pool was generated under a DIFFERENT run "
            "identity. Differences: %s. Re-run with --overwrite to replace the "
            "mismatched pool, or restore its recorded settings."
            % json.dumps(hio.diff_identity(config.identity,
                                           existing_manifest["identity"]),
                         ensure_ascii=False))

    keep = []
    for position, row in enumerate(rows, 1):
        errors = schema.validate_candidate(row)
        if errors:
            # Dropping it would leave a pool whose contents disagree with what this run
            # believes it resumed, and the failure would only surface downstream.
            raise hio.HonestBokError(
                "refusing to resume: %s row %d is not a valid candidate (%s). Re-run "
                "with --overwrite to discard the pool."
                % (ws.candidates, position, errors[0]))
        reason = hio.check_candidate_provenance(row, config.digest, config.identity)
        if reason is not None:
            raise hio.HonestBokError(
                "refusing to resume: candidate %s#%s fails provenance (%s). The pool is "
                "mixed; re-run with --overwrite."
                % (row.get("item_id"), row.get("candidate_idx"), reason))
        keep.append(row)
    return keep, stats, duplicates


# ---- driver ---------------------------------------------------------------
def require_stories_for_scope(items, stories, config):
    """Fail closed before generation if any in-scope item lacks story text.

    Selection requires a complete best-of-K pool and grounded quote validation, so a
    missing story indicates invalid inputs rather than an item that may be skipped.
    """
    missing = [(index, items[index].get("story_title"))
               for index in hio.iter_indices(items, config.start_index, config.limit)
               if stories.get(items[index].get("story_title")) is None]
    if missing:
        raise hio.HonestBokError(
            "%d in-scope item(s) have no story text (first: %s). Generation would "
            "produce an incomplete best-of-K pool that selection must reject, so it is "
            "refused up front. Fix the story map." % (len(missing), missing[:5]))


def plan_tasks(items, stories, config, skip_keys):
    """Enumerate the (dataset_index, candidate_idx) work this run still owes.

    Callers must have run `require_stories_for_scope` first; a missing story here is a
    programming error, not a skip.
    """
    tasks = []
    for dataset_index in hio.iter_indices(items, config.start_index, config.limit):
        example = items[dataset_index]
        if stories.get(example.get("story_title")) is None:
            raise hio.HonestBokError(
                "internal error: index %d has no story; require_stories_for_scope "
                "should have refused this run" % dataset_index)
        for candidate_idx in range(config.k):
            if hio.candidate_key_from_parts(dataset_index, candidate_idx) in skip_keys:
                continue
            tasks.append((dataset_index, candidate_idx))
    return tasks


def run_generation(items, stories, config, skip_keys, appender, client, warn,
                   progress=None, transcript_fn=None):
    """Execute the task plan concurrently; returns (generated, failed, failures)."""
    tasks = plan_tasks(items, stories, config, skip_keys)
    default_transcript_source = transcript_fn is None
    generated = 0
    failed = 0
    failures = []

    def run_one(task):
        dataset_index, candidate_idx = task
        example = items[dataset_index]
        story = stories[example["story_title"]]
        seeds = hio.model_request_seeds(config.seed, dataset_index, candidate_idx,
                                        config.num_rounds)
        if config.dry_run:
            transcript = dry_run_transcript(example, hio.expected_stance(example),
                                            candidate_idx, config.num_rounds)
            seeds = []
        else:
            # The seed harness lives here, not inside the transcript source, so the
            # per-candidate seed plan is consumed and asserted identically whichever
            # source produced the transcript.
            turn_client = SeededTurnClient(client, seeds)
            if default_transcript_source:
                transcript = debate_transcript(
                    example, story, turn_client, config.num_rounds,
                    debate_config=config.debate_config,
                    marker_within_limit=config.strict_transcript_budget)
            else:
                transcript = transcript_fn(
                    example, story, turn_client, config.num_rounds)
            turn_client.require_complete()
        assert_stance_parity(transcript, example, dataset_index)
        if config.strict_transcript_budget:
            assert_transcript_budget(transcript, config.transcript_len, dataset_index)
        return build_candidate(example, dataset_index, candidate_idx, transcript,
                               config, seeds)

    def handle(task, future):
        nonlocal generated, failed
        dataset_index, candidate_idx = task
        try:
            candidate = future.result()
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not kill the run
            failed += 1
            message = "%s: %s" % (type(exc).__name__, exc)
            failures.append({"dataset_index": dataset_index,
                             "candidate_idx": candidate_idx, "error": message})
            warn("candidate %d#%d failed: %s" % (dataset_index, candidate_idx, message))
        else:
            appender.append(candidate)
            generated += 1
        if progress is not None:
            progress.update(1)

    # Bound in-flight submissions so completed transcripts are persisted without
    # retaining an entire large candidate pool in memory.
    max_inflight = max(1, config.concurrency * 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        pending = {}
        task_iter = iter(tasks)
        exhausted = False
        while True:
            while not exhausted and len(pending) < max_inflight:
                try:
                    task = next(task_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(run_one, task)] = task
            if not pending:
                break
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                handle(pending.pop(future), future)
    return generated, failed, failures


def assert_transcript_budget(transcript, hard_limit, dataset_index):
    """Fail closed if any stored public argument exceeds the requested hard bound."""
    for round_index, round_row in enumerate(transcript.get("rounds") or [], 1):
        for debater in ("Debater A", "Debater B"):
            argument = round_row.get(debater)
            if not isinstance(argument, str):
                raise hio.HonestBokError(
                    "index %d round %d %s argument is not a string"
                    % (dataset_index, round_index, debater))
            observed = len(argument.split())
            if observed > hard_limit:
                raise hio.HonestBokError(
                    "index %d round %d %s stored %d whitespace words, exceeding "
                    "--transcript-len %d"
                    % (dataset_index, round_index, debater, observed, hard_limit))


def generate(ws, config, overwrite=False, skip_endpoint_check=False,
             show_progress=True, warn=hio.stderr_warn, client=None, transcript_fn=None):
    """Internal API: run generation against an explicit workspace/config.

    The CLI builds a production or dry-run workspace; tests build a test workspace.
    argv can never reach this with arbitrary paths. `client` and `transcript_fn` are
    TEST-ONLY injection points that let the production-shaped path run without a vLLM
    server or an importable debate.py.
    """
    hio.assert_production_layout(ws)
    if ws.mode == hio.MODE_PRODUCTION:
        if config.dry_run:
            raise hio.HonestBokError(
                "internal error: a dry-run config must use the dryrun workspace")
        hio.assert_production_endpoint(config.base_url)
        if config.start_index or config.limit is not None:
            raise hio.HonestBokError(
                "--start-index/--limit produce a partial pool and are rejected in "
                "production; selection requires exactly k candidates for every item.")
    ws.ensure_dirs()

    items = hio.load_dataset_items(ws.items)
    stories = hio.load_story_map(ws.stories)
    hio.check_items_are_no_debate(items, ws.items)
    require_stories_for_scope(items, stories, config)

    if overwrite:
        for path in (ws.candidates, ws.manifest):
            if os.path.exists(path):
                os.remove(path)

    existing, read_stats, duplicates = reusable_candidates(ws, config, warn)
    skip_keys = {hio.candidate_key(row) for row in existing}
    if read_stats["truncated_tail"] or duplicates:
        print("[debate-bok] resume: %d reusable candidate(s) (dropped %d identical "
              "duplicate(s), truncated_tail=%s)"
              % (len(existing), duplicates, read_stats["truncated_tail"]))

    # The manifest is the provenance anchor; write it before any candidate exists.
    hio.atomic_write_json(ws.manifest, config.manifest())

    tasks = plan_tasks(items, stories, config, skip_keys)
    print("[debate-bok] dataset=%s mode=%s items=%d k=%d pending=%d resumed=%d"
          % (ws.dataset, ws.mode, len(items), config.k, len(tasks), len(existing)))
    if not tasks:
        hio.canonicalize_jsonl(ws.candidates, hio.candidate_key, hio.candidate_sort_key,
                               warn=warn)
        print("[debate-bok] nothing to do; %s canonicalized" % ws.candidates)
        return 0

    if config.dry_run:
        client = None
    elif client is None:
        if not skip_endpoint_check:
            served = probe_endpoint(config.base_url, config.model)
            print("[debate-bok] endpoint %s ready; served=%s" % (config.base_url, served))
        client = VLLMChatClient(config.base_url, config.api_key, config.model,
                                config.lm_config, max_retries=config.max_retries)

    use_bar = tqdm is not None and show_progress
    progress = tqdm(total=len(tasks), desc="honest BoK", unit="cand") if use_bar else None
    started = time.time()
    with hio.JsonlAppender(ws.candidates) as appender:
        try:
            generated, failed, failures = run_generation(
                items, stories, config, skip_keys, appender, client, warn, progress,
                transcript_fn=transcript_fn)
        finally:
            if progress is not None:
                progress.close()

    _, stats = hio.canonicalize_jsonl(ws.candidates, hio.candidate_key,
                                      hio.candidate_sort_key, warn=warn)
    print("[debate-bok] generated=%d failed=%d total=%d elapsed=%.1fs -> %s"
          % (generated, failed, stats["final_rows"], time.time() - started, ws.candidates))
    if failures:
        preview = failures[:10]
        print("[debate-bok] %d failure(s); first %d: %s"
              % (len(failures), len(preview), json.dumps(preview, ensure_ascii=False)),
              file=sys.stderr)
        return 1  # non-zero so a scheduler requeues; completed work is already durable
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate K honest debate candidates per item via vLLM HTTP.",
        epilog="There are no path flags: inputs and outputs are locked to "
               "the selected dataset directory and runs/honest/<DS>/candidate.jsonl. "
               "QuALITY-H-50/250 retain the copied QuALITY-H input filenames. "
               "--dry-run redirects the whole run into runs/honest/<DS>/dryrun/.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=hio.DATASETS)
    p.add_argument("--k", type=hio.positive_int, default=hio.DEFAULT_K,
                   help="candidates per item (default: 8)")
    p.add_argument("--concurrency", type=hio.positive_int, default=DEFAULT_CONCURRENCY,
                   help="candidates generated in parallel; A/B stay serial within a round")
    p.add_argument("--num-rounds", "--rounds", dest="num_rounds",
                   type=hio.positive_int, default=hio.DEFAULT_NUM_ROUNDS)
    p.add_argument(
        "--transcript-len", type=transcript_len_arg, default=None,
        help="hard per-debater/per-round public-argument limit in whitespace words; "
             "defaults to the dataset arm (50 for QuALITY-H-50, 250 for "
             "QuALITY-H-250, otherwise 150). The prompt soft limit is 50 when hard=50 "
             "and hard-50 otherwise.")
    p.add_argument("--base-url", default=hio.PRODUCTION_BASE_URL,
                   help="locked to %s in production" % hio.PRODUCTION_BASE_URL)
    p.add_argument("--api-key", default=hio.DEFAULT_API_KEY)
    p.add_argument("--model", default=hio.DEFAULT_GENERATOR_MODEL)
    p.add_argument("--seed", type=int, default=0,
                   help="base seed for the per-request sampling seed policy")
    p.add_argument("--start-index", type=hio.nonnegative_int, default=0,
                   help="dry-run only; a partial pool is rejected in production")
    p.add_argument("--limit", type=hio.positive_int, default=None,
                   help="dry-run only; a partial pool is rejected in production")
    p.add_argument("--max-retries", type=hio.nonnegative_int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--overwrite", action="store_true",
                   help="discard the existing pool + manifest instead of resuming")
    p.add_argument("--dry-run", action="store_true",
                   help="deterministic placeholder transcripts in an isolated sandbox; "
                        "no network, no debate.py, never promotable to production")
    p.add_argument("--skip-endpoint-check", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def main(argv=None, workspace=None, lm_config=None, transcript_fn=None):
    """CLI entry point.

    `workspace`, `lm_config` and `transcript_fn` are TEST-ONLY internal injection
    points; argv cannot reach them.
    """
    args = parse_args(argv)
    ws = workspace or hio.Workspace(
        args.dataset, mode=hio.MODE_DRYRUN if args.dry_run else hio.MODE_PRODUCTION)
    try:
        # Validate the endpoint BEFORE building the config: GenerationConfig snapshots
        # debate.LM_CONFIG, and a typo'd endpoint should fail fast, not after a load.
        if ws.mode == hio.MODE_PRODUCTION and not args.dry_run:
            hio.assert_production_endpoint(args.base_url)
        transcript_len = resolve_transcript_len(args.dataset, args.transcript_len)
        if not args.overwrite:
            precheck_existing_pool(ws, args, args.dry_run, transcript_len)
        config = GenerationConfig(
            ws, k=args.k, num_rounds=args.num_rounds, model=args.model,
            base_url=args.base_url, api_key=args.api_key, seed=args.seed,
            dry_run=args.dry_run, concurrency=args.concurrency,
            max_retries=args.max_retries, start_index=args.start_index,
            limit=args.limit, transcript_len=transcript_len, lm_config=lm_config)
        return generate(ws, config, overwrite=args.overwrite,
                        skip_endpoint_check=args.skip_endpoint_check,
                        show_progress=not args.no_progress,
                        transcript_fn=transcript_fn)
    except hio.HonestBokError as exc:
        print("[debate-bok] ABORT: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
