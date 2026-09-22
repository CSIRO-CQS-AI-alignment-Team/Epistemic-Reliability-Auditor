"""Batch driver: one INDEPENDENT GCG suffix per adversarial candidate.

This is the whole-pool counterpart of ``gcg.run``. ``gcg.run`` optimizes one suffix
shared by every task in the spec it is handed; feeding it a many-candidate spec would
therefore produce a *universal* suffix. This module instead builds one two-task spec
per candidate (exactly ``Q_Y`` + ``Q_H``, sharing that candidate's single editable
span) and runs one :class:`~gcg.optimizer.GCGOptimizer` per candidate, so no suffix is
ever shared across candidates.

GCG is white-box: every search step backpropagates through the frozen verifier into
the input embeddings (``gcg.objective.suffix_gradient``) and reads the embedding
matrix (``gcg.optimizer._propose_token_table``). It therefore cannot be served by an
inference server that only exposes text/logprobs, and it cannot be threaded against a
shared model: ``suffix_gradient`` flips ``requires_grad`` on every parameter and
toggles ``train()``/``eval()`` for the duration of a call, so two concurrent callers
would corrupt each other. The standalone ``optimize`` and ``score`` subcommands support
process-per-shard execution. The four-GPU Slurm launchers call ``gcg.resident_pool``:
threads are safe there because every worker owns a distinct
single-GPU model object, and those objects persist into scoring.

Stages (each an argparse subcommand)::

    prepare           CPU, single process: validate the pool fail-closed, write
                      native/ + tasks/ + manifest.jsonl. The ONLY writer of those.
    optimize          GPU, one process per shard: optimize + apply, per candidate.
    merge-candidates  CPU: full-coverage merge into candidates-gcg.jsonl.
    score             GPU, one process per shard: wraps score_verifier.main.
    merge-scores      CPU: validated merge into verifier_scores-gcg.jsonl.
    summary           CPU: final artifact validation + run-summary.json.

Workers never write shared files: after ``prepare`` every path a worker touches is
keyed by its own source-index stem, so the shards cannot race.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence

import torch

from adversarial_transcript import common, schema, score_verifier

from . import prepare as prepare_module
from . import run as run_module
from .apply import apply_suffix
from .objective import (
    infer_logits_kwarg,
    load_task_spec,
    sha256_file,
    sha256_text,
    task_spec_digest,
)
from .optimizer import GCGConfig, GCGOptimizer, save_result

# Schema identities bind resumed artifacts to the on-disk layout and marker semantics.
CONFIG_SCHEMA_VERSION = "gcg-run-all-config-v1"
DONE_SCHEMA_VERSION = "gcg-run-all-done-v1"

DEFAULT_MODEL = score_verifier.DEFAULT_MODEL
DEFAULT_CONDITION = "adversarial"
DEFAULT_EXPECTED_K = 8
STEM_WIDTH = 6

# Cluster defaults for this pipeline. They intentionally differ from gcg.run's
# argparse defaults (64/256/256), which are tuned for a single pilot candidate.
DEFAULT_STEPS = 8
DEFAULT_TOP_K = 32
DEFAULT_GCG_CANDIDATES = 64
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_SUFFIX_LEN = 16
DEFAULT_SEED = 0
DEFAULT_QY_WEIGHT = 1.0
DEFAULT_QH_WEIGHT = 2.0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def stem_for(source_index: int) -> str:
    """Zero-padded source-index stem.

    Filenames are keyed by source index rather than ``item_id`` so that a stem is
    unique per CANDIDATE (an item contributes K of them) and is always a safe
    filename regardless of what an item id contains.
    """

    index = int(source_index)
    if index < 0:
        raise ValueError(f"source_index must be non-negative, got {index}")
    return f"{index:0{STEM_WIDTH}d}"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def candidate_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(row["item_id"]), str(row["condition"]), int(row["candidate_idx"]))


def _key_list(key: Sequence[Any]) -> list[Any]:
    return [str(key[0]), str(key[1]), int(key[2])]


def _manifest_key(manifest_row: Mapping[str, Any]) -> tuple[str, str, int]:
    key = manifest_row["candidate_key"]
    return (str(key[0]), str(key[1]), int(key[2]))


def _same_key(value: Any, expected: tuple[str, str, int]) -> bool:
    """True when ``value`` is a well-formed 3-part key equal to ``expected``."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        return tuple(_key_list(value)) == expected
    except (TypeError, ValueError):
        return False


class RunLayout:
    """Every path this driver reads or writes, derived from one run root."""

    def __init__(self, run_root: str) -> None:
        self.root = os.path.abspath(run_root)

    # -- per-candidate ------------------------------------------------------ #
    def native(self, stem: str) -> str:
        return os.path.join(self.root, "native", f"{stem}.jsonl")

    def tasks(self, stem: str) -> str:
        return os.path.join(self.root, "tasks", f"{stem}.json")

    def result(self, stem: str) -> str:
        return os.path.join(self.root, "results", f"{stem}.json")

    def applied(self, stem: str) -> str:
        return os.path.join(self.root, "applied", f"{stem}.jsonl")

    def done(self, stem: str) -> str:
        return os.path.join(self.root, "done", f"{stem}.json")

    def error(self, stem: str) -> str:
        return os.path.join(self.root, "errors", f"{stem}.json")

    # -- shared ------------------------------------------------------------- #
    def score_shard(self, shard_index: int) -> str:
        return os.path.join(self.root, "scores", f"shard-{int(shard_index)}.jsonl")

    @property
    def manifest(self) -> str:
        return os.path.join(self.root, "manifest.jsonl")

    @property
    def prepare_config(self) -> str:
        return os.path.join(self.root, "prepare-config.json")

    @property
    def merged_candidates(self) -> str:
        return os.path.join(self.root, "candidates-gcg.jsonl")

    @property
    def merged_scores(self) -> str:
        return os.path.join(self.root, "verifier_scores-gcg.jsonl")

    @property
    def selected(self) -> str:
        return os.path.join(self.root, "selected-gcg.jsonl")

    @property
    def report(self) -> str:
        return os.path.join(self.root, "report-gcg.json")

    @property
    def summary(self) -> str:
        return os.path.join(self.root, "run-summary.json")

    def ensure_dirs(self) -> None:
        for name in ("native", "tasks", "results", "applied", "done", "errors", "scores"):
            os.makedirs(os.path.join(self.root, name), exist_ok=True)


def read_manifest(layout: RunLayout) -> list[dict[str, Any]]:
    if not os.path.exists(layout.manifest):
        raise SystemExit(
            f"missing {layout.manifest}; run `python -m gcg.run_all prepare` first"
        )
    rows = common.read_jsonl(layout.manifest)
    if not rows:
        raise SystemExit(f"{layout.manifest} is empty")
    seen_stems: set[str] = set()
    seen_keys: set[tuple[str, str, int]] = set()
    for position, row in enumerate(rows):
        if int(row.get("source_index", -1)) != position:
            raise SystemExit(
                f"{layout.manifest}: row {position} has source_index "
                f"{row.get('source_index')!r}; the manifest must be in source order"
            )
        stem = str(row.get("stem", ""))
        if stem != stem_for(position):
            raise SystemExit(f"{layout.manifest}: row {position} has stem {stem!r}")
        if stem in seen_stems:
            raise SystemExit(f"{layout.manifest}: duplicate stem {stem!r}")
        seen_stems.add(stem)
        key = tuple(row.get("candidate_key", ()))
        if len(key) != 3:
            raise SystemExit(f"{layout.manifest}: row {position} has no 3-part candidate_key")
        key = (str(key[0]), str(key[1]), int(key[2]))
        if key in seen_keys:
            raise SystemExit(f"{layout.manifest}: duplicate candidate_key {list(key)}")
        seen_keys.add(key)
    return rows


def shard_positions(total: int, num_shards: int, shard_index: int) -> list[int]:
    """Source indices claimed by one shard (round-robin, disjoint, exhaustive)."""

    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} out of range for {num_shards} shard(s)")
    return [index for index in range(int(total)) if index % num_shards == shard_index]


def resolve_shard(from_slurm: bool, num_shards: int, shard_index: int) -> tuple[int, int]:
    """Resolve (num_shards, shard_index), reading Slurm's env INSIDE this process.

    ``SLURM_PROCID`` must never be expanded by the submitting shell: under ``set -u``
    it is unset there, and even without it every task would inherit the same value.
    """

    if from_slurm:
        missing = [name for name in ("SLURM_NTASKS", "SLURM_PROCID") if not os.environ.get(name)]
        if missing:
            raise SystemExit(
                f"--from-slurm requires {' and '.join(missing)} in the environment; "
                "launch this stage through srun"
            )
        try:
            num_shards = int(os.environ["SLURM_NTASKS"])
            shard_index = int(os.environ["SLURM_PROCID"])
        except ValueError as exc:
            raise SystemExit(f"invalid SLURM_NTASKS/SLURM_PROCID: {exc}") from exc
    if num_shards < 1:
        raise SystemExit("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise SystemExit(f"--shard-index must be in [0, {num_shards})")
    return num_shards, shard_index


def require_single_visible_gpu(allow_cpu: bool) -> None:
    """Fail fast unless exactly one GPU is visible (the ``--gpu-bind=single:1`` shape).

    Two visible GPUs means the binding is wrong and two shards would land on the same
    device; the 20B model is far too large for that to merely be slow.
    """

    if allow_cpu:
        return
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device is visible; this stage needs one GPU per task "
            "(pass --allow-cpu only for CPU-stub development runs)"
        )
    count = int(torch.cuda.device_count())
    if count != 1:
        raise SystemExit(
            f"expected exactly one visible GPU, found {count}; launch with "
            "--gpus-per-task=1 --gpu-bind=single:1 so shards cannot share a device"
        )


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #


def validate_source_pool(
    rows: Sequence[Mapping[str, Any]],
    stories: Mapping[str, Any],
    *,
    conditions: Sequence[str],
    expected_k: int,
) -> None:
    """Fail-closed validation of the whole source pool before anything is written.

    Every failure here would otherwise surface hours later inside a GPU worker, or --
    worse -- silently shrink the pool. Nothing is skipped: a row this driver will not
    optimize is an error, not a filter.
    """

    if not rows:
        raise SystemExit("source candidate pool is empty")
    condition_set = {str(c).lower() for c in conditions}
    unknown = condition_set - set(schema.CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown condition(s): {sorted(unknown)}")

    seen: dict[tuple[str, str, int], int] = {}
    by_item: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SystemExit(f"row {index}: candidate rows must be JSON objects")
        errors = schema.validate_candidate(row)
        if errors:
            raise SystemExit(f"row {index} ({row.get('item_id')!r}): invalid candidate: {errors[0]}")
        condition = row.get("condition")
        if condition not in condition_set:
            raise SystemExit(
                f"row {index} ({row.get('item_id')!r}): condition {condition!r} is not in "
                f"{sorted(condition_set)}; this driver refuses a mixed pool rather than "
                "silently dropping rows"
            )
        story_title = row.get("story_title")
        if story_title not in stories:
            raise SystemExit(f"row {index} ({row.get('item_id')!r}): missing story {story_title!r}")
        gen_meta = row.get("gen_meta")
        if isinstance(gen_meta, Mapping) and (
            gen_meta.get("gcg_suffix_applied") or gen_meta.get("gcg_suffix_text")
        ):
            raise SystemExit(
                f"row {index} ({row.get('item_id')!r}): candidate already carries a GCG "
                "suffix; optimize against the NATIVE adversarial pool, not an applied one"
            )
        key = candidate_key(row)
        if key in seen:
            raise SystemExit(
                f"row {index}: duplicate candidate key {list(key)} (first seen at row {seen[key]})"
            )
        seen[key] = index
        by_item.setdefault(str(row["item_id"]), []).append(int(row["candidate_idx"]))

    if expected_k:
        expected_indices = set(range(int(expected_k)))
        for item_id in sorted(by_item):
            indices = sorted(by_item[item_id])
            if len(indices) != int(expected_k) or set(indices) != expected_indices:
                raise SystemExit(
                    f"{item_id}: expected exactly {expected_k} candidates with "
                    f"candidate_idx 0..{int(expected_k) - 1}, got {indices}"
                )


def prepare_config_payload(
    *,
    source_candidates: str,
    stories: str,
    conditions: Sequence[str],
    expected_k: int,
    option_seed: int,
    qy_weight: float,
    qh_weight: float,
) -> dict[str, Any]:
    """Identity of a preparation, so a resume cannot mix two different inputs."""

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "prompt_version": score_verifier.PROMPT_VERSION,
        "source_candidates": os.path.abspath(source_candidates),
        "source_candidates_sha256": sha256_file(source_candidates),
        "stories": os.path.abspath(stories),
        "stories_sha256": sha256_file(stories),
        "conditions": sorted({str(c).lower() for c in conditions}),
        "expected_k": int(expected_k),
        "option_seed": int(option_seed),
        "qy_weight": float(qy_weight),
        "qh_weight": float(qh_weight),
    }


def build_candidate_spec(
    row: Mapping[str, Any],
    stories: Mapping[str, Any],
    *,
    option_seed: int,
    qy_weight: float,
    qh_weight: float,
    conditions: Sequence[str],
) -> dict[str, Any]:
    """One spec for ONE candidate: exactly the Q_Y and Q_H readouts."""

    spec = prepare_module.build_spec(
        [row],
        stories,
        option_seed=option_seed,
        qy_weight=qy_weight,
        qh_weight=qh_weight,
        limit=1,
        conditions=conditions,
    )
    tasks = spec.get("tasks", [])
    if len(tasks) != 2:
        raise SystemExit(
            f"{row.get('item_id')}: expected exactly 2 tasks (Q_Y, Q_H), got {len(tasks)}"
        )
    readouts = [task.get("metadata", {}).get("readout") for task in tasks]
    if readouts != ["Q_Y", "Q_H"]:
        raise SystemExit(f"{row.get('item_id')}: expected readouts ['Q_Y', 'Q_H'], got {readouts}")
    if int(spec.get("meta", {}).get("selected_candidate_count", 0)) != 1:
        raise SystemExit(f"{row.get('item_id')}: spec does not describe exactly one candidate")
    groups = {task.get("group") for task in tasks}
    if len(groups) != 1:
        raise SystemExit(f"{row.get('item_id')}: Q_Y/Q_H must share one group, got {sorted(groups)}")
    return spec


def _write_if_absent_or_equal(
    path: str,
    payload: Any,
    *,
    writer: Callable[[str, Any], None],
    reader: Callable[[str], Any],
    force: bool,
    what: str,
) -> None:
    """Write, or verify an existing artifact is identical.

    A resume must be bound to the CURRENT input: an artifact that differs means the
    source pool, the prompts or the config moved under a half-finished run, and the
    existing optimized suffixes do not describe the changed input.
    """

    if os.path.exists(path) and not force:
        if reader(path) == payload:
            return
        raise SystemExit(
            f"{path}: existing {what} differs from the one this input produces. "
            "The run root is bound to its input; resubmit with --force to rebuild it, "
            "or use a new RUN_ROOT."
        )
    writer(path, payload)


def cmd_prepare(args: argparse.Namespace) -> int:
    layout = RunLayout(args.run_root)
    layout.ensure_dirs()

    rows = common.read_jsonl(args.source_candidates)
    stories = common.load_story_map(args.stories)
    validate_source_pool(
        rows, stories, conditions=args.conditions, expected_k=args.expected_k
    )

    config = prepare_config_payload(
        source_candidates=args.source_candidates,
        stories=args.stories,
        conditions=args.conditions,
        expected_k=args.expected_k,
        option_seed=args.option_seed,
        qy_weight=args.qy_weight,
        qh_weight=args.qh_weight,
    )
    _write_if_absent_or_equal(
        layout.prepare_config, config,
        writer=common.write_json, reader=common.read_json,
        force=args.force, what="preparation config",
    )

    manifest: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        stem = stem_for(source_index)
        native_row = json.loads(json.dumps(row, ensure_ascii=False))
        spec = build_candidate_spec(
            row, stories,
            option_seed=args.option_seed,
            qy_weight=args.qy_weight,
            qh_weight=args.qh_weight,
            conditions=args.conditions,
        )
        _write_if_absent_or_equal(
            layout.native(stem), [native_row],
            writer=common.write_jsonl, reader=common.read_jsonl,
            force=args.force, what="native candidate",
        )
        _write_if_absent_or_equal(
            layout.tasks(stem), spec,
            writer=common.write_json, reader=common.read_json,
            force=args.force, what="task spec",
        )
        manifest.append({
            "source_index": source_index,
            "stem": stem,
            "candidate_key": _key_list(candidate_key(row)),
            "item_id": str(row["item_id"]),
            "condition": str(row["condition"]),
            "candidate_idx": int(row["candidate_idx"]),
            "story_title": str(row["story_title"]),
            "native": os.path.relpath(layout.native(stem), layout.root),
            "tasks": os.path.relpath(layout.tasks(stem), layout.root),
            "native_sha256": sha256_file(layout.native(stem)),
            "tasks_sha256": sha256_file(layout.tasks(stem)),
            "task_spec_digest": task_spec_digest(spec),
        })

    _write_if_absent_or_equal(
        layout.manifest, manifest,
        writer=common.write_jsonl, reader=common.read_jsonl,
        force=args.force, what="manifest",
    )
    print(
        f"[run_all.prepare] {len(manifest)} candidate(s), "
        f"{2 * len(manifest)} task(s) -> {layout.manifest}"
    )
    return 0


# --------------------------------------------------------------------------- #
# optimize
# --------------------------------------------------------------------------- #


def gcg_config_payload(config: GCGConfig, *, suffix_len: int, init_text: str | None,
                       allow_special_tokens: bool, allow_non_ascii: bool) -> dict[str, Any]:
    return {
        "steps": int(config.steps),
        "top_k": int(config.top_k),
        "candidates": int(config.candidates),
        "eval_batch_size": int(config.eval_batch_size),
        "seed": int(config.seed),
        "normalize_per_task": bool(config.normalize_per_task),
        "suffix_len": int(suffix_len),
        "init_text": init_text,
        "allow_special_tokens": bool(allow_special_tokens),
        "allow_non_ascii": bool(allow_non_ascii),
    }


def completion_reason(
    layout: RunLayout,
    manifest_row: Mapping[str, Any],
    *,
    gcg_config: Mapping[str, Any] | None = None,
    model_name: str | None = None,
) -> str | None:
    """Return why this candidate is NOT reusable, or None when it fully verifies.

    Resume is fail-closed and bound to the current input: every artifact must exist,
    every recorded hash must still match the file on disk, the identity must match the
    manifest, and -- the check that actually matters -- re-running ``apply_suffix`` on
    the native candidate with the serialized result must reproduce the applied file
    byte-equivalently. A marker alone is never enough.
    """

    stem = str(manifest_row["stem"])
    paths = {
        "native": layout.native(stem),
        "tasks": layout.tasks(stem),
        "result": layout.result(stem),
        "applied": layout.applied(stem),
        "done": layout.done(stem),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            return f"missing {name} artifact {os.path.relpath(path, layout.root)}"

    try:
        marker = common.read_json(paths["done"])
    except (OSError, ValueError) as exc:
        return f"unreadable done marker: {exc}"
    if not isinstance(marker, Mapping):
        return "done marker is not a JSON object"
    if marker.get("schema_version") != DONE_SCHEMA_VERSION:
        return f"done marker schema {marker.get('schema_version')!r} != {DONE_SCHEMA_VERSION!r}"
    if str(marker.get("stem")) != stem:
        return f"done marker stem {marker.get('stem')!r} != {stem!r}"
    if int(marker.get("source_index", -1)) != int(manifest_row["source_index"]):
        return "done marker source_index does not match the manifest"
    if not _same_key(marker.get("candidate_key"), _manifest_key(manifest_row)):
        return "done marker candidate_key does not match the manifest"
    if model_name is not None and marker.get("model") != model_name:
        return f"done marker model {marker.get('model')!r} != {model_name!r}"
    if gcg_config is not None and marker.get("gcg_config") != dict(gcg_config):
        return "done marker gcg_config differs from the current configuration"

    for name in ("native", "tasks", "result", "applied"):
        recorded = marker.get(f"{name}_sha256")
        actual = sha256_file(paths[name])
        if recorded != actual:
            return f"{name} artifact changed since it was recorded"
    for name in ("native", "tasks"):
        if manifest_row.get(f"{name}_sha256") != marker.get(f"{name}_sha256"):
            return f"{name} artifact does not match the manifest"

    try:
        native_rows = common.read_jsonl(paths["native"])
        applied_rows = common.read_jsonl(paths["applied"])
        spec = load_task_spec(paths["tasks"])
        result = common.read_json(paths["result"])
    except (OSError, ValueError) as exc:
        return f"unreadable artifact: {exc}"
    if len(native_rows) != 1 or len(applied_rows) != 1:
        return "native/applied must hold exactly one candidate row"
    if candidate_key(native_rows[0]) != _manifest_key(manifest_row):
        return "native candidate key does not match the manifest"

    try:
        recomputed = apply_suffix(native_rows, spec, result)
    except (ValueError, KeyError, TypeError) as exc:
        return f"stored result does not apply to the native candidate: {exc}"
    if recomputed != applied_rows:
        return "re-applying the stored suffix does not reproduce the applied candidate"
    return None


class ShardOutcome:
    def __init__(self) -> None:
        self.completed: list[str] = []
        self.skipped: list[str] = []
        self.failed: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.failed


def optimize_one(
    layout: RunLayout,
    manifest_row: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    allowed_ids: torch.Tensor,
    logits_kwarg: str | None,
    config: GCGConfig,
    suffix_len: int,
    init_text: str | None,
    allow_special_tokens: bool,
    allow_non_ascii: bool,
    model_name: str,
    device: Any,
    device_map: str,
    dtype: Any,
    shard_index: int,
    num_shards: int,
) -> None:
    """Optimize, serialize and apply ONE candidate's own suffix."""

    stem = str(manifest_row["stem"])
    tasks_path = layout.tasks(stem)
    native_path = layout.native(stem)
    spec = load_task_spec(tasks_path)
    native_rows = common.read_jsonl(native_path)
    if len(native_rows) != 1:
        raise ValueError(f"{native_path} must hold exactly one candidate row")
    key = candidate_key(native_rows[0])
    expected = _manifest_key(manifest_row)
    if key != expected:
        raise ValueError(f"{native_path} holds {list(key)}, manifest says {list(expected)}")

    # Tokenizer-side checks for THIS candidate run before any of its GPU work; the
    # editable vocabulary is the one cached scan shared by the whole shard.
    checked = run_module.preflight(
        tokenizer, spec,
        suffix_len=suffix_len,
        init_text=init_text,
        allow_special_tokens=allow_special_tokens,
        allow_non_ascii=allow_non_ascii,
        allowed_ids=allowed_ids,
    )
    optimizer = GCGOptimizer(
        model,
        tokenizer,
        checked["tasks"],
        checked["choice_ids"],
        pad_token_id=int(tokenizer.pad_token_id),
        config=config,
        logits_kwarg=logits_kwarg,
        allowed_ids=checked["allowed_ids"],
        validator=checked["validator"],
    )
    result = optimizer.optimize(checked["suffix_ids"])

    metadata = {
        "model": model_name,
        "device": str(device),
        "device_map": device_map,
        "dtype": str(dtype or "model-default"),
        "task_spec": os.path.abspath(tasks_path),
        # Content digest: exactly what gcg.apply requires to bind result -> spec.
        "task_spec_digest": task_spec_digest(spec),
        "task_spec_sha256": sha256_file(tasks_path),
        "task_count": len(checked["tasks"]),
        "prefill": checked["prefill"],
        "choice_token_ids": checked["choice_ids"],
        "logits_kwarg": logits_kwarg,
        "suffix_separator": str(spec.get("meta", {}).get("suffix_separator", "")),
        "preflight": checked["report"],
        "editable_suffix_is_public": True,
        "warning": (
            "Validate the decoded suffix as a public transcript fragment and rerun "
            "the production quote/correctness/leakage filters before using it."
        ),
        "run_all": {
            "stem": stem,
            "source_index": int(manifest_row["source_index"]),
            "candidate_key": list(manifest_row["candidate_key"]),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        },
    }
    save_result(layout.result(stem), result, metadata=metadata)

    # Apply from the SERIALIZED result, not the in-memory object: that is the artifact
    # a resume re-verifies against, and it exercises gcg.apply's evidence checks.
    serialized = common.read_json(layout.result(stem))
    applied_rows = apply_suffix(native_rows, spec, serialized)
    common.write_jsonl(layout.applied(stem), applied_rows)

    common.write_json(layout.done(stem), {
        "schema_version": DONE_SCHEMA_VERSION,
        "stem": stem,
        "source_index": int(manifest_row["source_index"]),
        "candidate_key": list(manifest_row["candidate_key"]),
        "model": model_name,
        "gcg_config": gcg_config_payload(
            config, suffix_len=suffix_len, init_text=init_text,
            allow_special_tokens=allow_special_tokens, allow_non_ascii=allow_non_ascii,
        ),
        "task_spec_digest": task_spec_digest(spec),
        "suffix_text_sha256": result.suffix_text_sha256,
        "initial_loss": float(result.initial.loss),
        "final_loss": float(result.final.loss),
        "native_sha256": sha256_file(native_path),
        "tasks_sha256": sha256_file(tasks_path),
        "result_sha256": sha256_file(layout.result(stem)),
        "applied_sha256": sha256_file(layout.applied(stem)),
    })


def clear_error(layout: RunLayout, stem: str) -> None:
    path = layout.error(stem)
    if os.path.exists(path):
        os.remove(path)


def run_shard(
    layout: RunLayout,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    config: GCGConfig,
    model_name: str,
    suffix_len: int = DEFAULT_SUFFIX_LEN,
    init_text: str | None = None,
    allow_special_tokens: bool = False,
    allow_non_ascii: bool = False,
    device: Any = "cpu",
    device_map: str = "single",
    dtype: Any = None,
    force: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    log: Callable[[str], None] = print,
    on_candidate_done: Callable[[str, str], None] | None = None,
) -> ShardOutcome:
    """Optimize every claimed candidate sequentially against one loaded model.

    The model and tokenizer are supplied by the caller so a test can drive this with
    CPU stubs; the real entry point loads them once per worker. A failing candidate is
    isolated to its own error marker and never aborts the shard, but the shard still
    reports failure so the stage does not look successful. ``on_candidate_done`` is
    called once after each candidate reaches a terminal state (``completed``,
    ``skipped`` or ``failed``), which lets a parent orchestration layer maintain a
    single aggregate progress bar across dynamic worker queues.
    """

    outcome = ShardOutcome()
    layout.ensure_dirs()
    allowed_ids, _ = run_module.resolve_editable_vocabulary(
        tokenizer,
        allow_special_tokens=allow_special_tokens,
        allow_non_ascii=allow_non_ascii,
    )
    logits_kwarg = infer_logits_kwarg(model)
    config_payload = gcg_config_payload(
        config, suffix_len=suffix_len, init_text=init_text,
        allow_special_tokens=allow_special_tokens, allow_non_ascii=allow_non_ascii,
    )

    for manifest_row in manifest_rows:
        stem = str(manifest_row["stem"])
        status: str
        try:
            reason: str | None = None
            if not force:
                reason = completion_reason(
                    layout, manifest_row,
                    gcg_config=config_payload, model_name=model_name,
                )
            if not force and reason is None:
                # A verified candidate cannot also be a failed one.
                clear_error(layout, stem)
                outcome.skipped.append(stem)
                status = "skipped"
                log(f"[run_all.optimize] {stem}: verified, skipping")
            else:
                if reason is not None:
                    log(f"[run_all.optimize] {stem}: recomputing ({reason})")
                optimize_one(
                    layout, manifest_row,
                    model=model, tokenizer=tokenizer, allowed_ids=allowed_ids,
                    logits_kwarg=logits_kwarg, config=config, suffix_len=suffix_len,
                    init_text=init_text, allow_special_tokens=allow_special_tokens,
                    allow_non_ascii=allow_non_ascii, model_name=model_name,
                    device=device, device_map=device_map, dtype=dtype,
                    shard_index=shard_index, num_shards=num_shards,
                )
                clear_error(layout, stem)
                outcome.completed.append(stem)
                status = "completed"
                log(f"[run_all.optimize] {stem}: done")
        except Exception as exc:  # noqa: BLE001 - one candidate must not kill the shard
            common.write_json(layout.error(stem), {
                "stem": stem,
                "source_index": int(manifest_row.get("source_index", -1)),
                "candidate_key": list(manifest_row.get("candidate_key", [])),
                "shard_index": int(shard_index),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            outcome.failed.append(stem)
            status = "failed"
            log(f"[run_all.optimize] {stem}: FAILED {type(exc).__name__}: {exc}")
        if on_candidate_done is not None:
            on_candidate_done(stem, status)
    return outcome


def cmd_optimize(args: argparse.Namespace) -> int:
    num_shards, shard_index = resolve_shard(args.from_slurm, args.num_shards, args.shard_index)
    require_single_visible_gpu(args.allow_cpu)

    layout = RunLayout(args.run_root)
    manifest = read_manifest(layout)
    claimed = [manifest[i] for i in shard_positions(len(manifest), num_shards, shard_index)]
    print(
        f"[run_all.optimize] shard {shard_index}/{num_shards}: "
        f"{len(claimed)}/{len(manifest)} candidate(s)"
    )
    if not claimed:
        return 0

    device = run_module._resolve_device(args.device)
    dtype = run_module._dtype(args.dtype)
    if dtype is None and device.type == "cuda":
        dtype = torch.bfloat16
    tokenizer = run_module._load_tokenizer(args.model)
    model = run_module._load_model(args.model, device, dtype, args.device_map)

    config = GCGConfig(
        steps=args.steps,
        top_k=args.top_k,
        candidates=args.gcg_candidates,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        normalize_per_task=not args.no_gradient_normalization,
    )
    outcome = run_shard(
        layout, claimed,
        model=model, tokenizer=tokenizer, config=config, model_name=args.model,
        suffix_len=args.suffix_len, init_text=args.init_text,
        allow_special_tokens=args.allow_special_tokens,
        allow_non_ascii=args.allow_non_ascii,
        device=device, device_map=args.device_map, dtype=dtype,
        force=args.force, shard_index=shard_index, num_shards=num_shards,
    )
    print(
        f"[run_all.optimize] shard {shard_index}: completed={len(outcome.completed)} "
        f"skipped={len(outcome.skipped)} failed={len(outcome.failed)}"
    )
    if outcome.failed:
        print(
            f"[run_all.optimize] shard {shard_index} failed candidate(s): "
            f"{', '.join(outcome.failed[:10])}",
            file=sys.stderr,
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# merge-candidates
# --------------------------------------------------------------------------- #


def cmd_merge_candidates(args: argparse.Namespace) -> int:
    layout = RunLayout(args.run_root)
    manifest = read_manifest(layout)

    missing: list[str] = []
    for manifest_row in manifest:
        reason = completion_reason(layout, manifest_row)
        if reason is not None:
            missing.append(f"{manifest_row['stem']}: {reason}")
    if missing:
        # Merging only the successful subset would silently publish a smaller pool and
        # break the K-per-item contract that selection depends on.
        preview = "\n  ".join(missing[:10])
        raise SystemExit(
            f"refusing to merge: {len(missing)}/{len(manifest)} candidate(s) are not "
            f"completed and verified:\n  {preview}"
            + ("\n  ..." if len(missing) > 10 else "")
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    signatures: dict[tuple[str, str], str] = {}
    for manifest_row in manifest:
        stem = str(manifest_row["stem"])
        # One pool, one search configuration: a resume that changed --steps or the
        # model halfway would otherwise publish a silently heterogeneous pool.
        marker = common.read_json(layout.done(stem))
        signature = (str(marker.get("model")), canonical_json(marker.get("gcg_config")))
        signatures.setdefault(signature, stem)
        applied_rows = common.read_jsonl(layout.applied(stem))
        if len(applied_rows) != 1:
            raise SystemExit(f"{manifest_row['stem']}: applied file must hold exactly one row")
        row = applied_rows[0]
        errors = schema.validate_candidate(row)
        if errors:
            raise SystemExit(f"{manifest_row['stem']}: applied candidate failed schema: {errors[0]}")
        key = candidate_key(row)
        if list(key) != list(manifest_row["candidate_key"]):
            raise SystemExit(
                f"{manifest_row['stem']}: applied key {list(key)} != manifest "
                f"{list(manifest_row['candidate_key'])}"
            )
        if key in seen:
            raise SystemExit(f"duplicate candidate key in merge: {list(key)}")
        seen.add(key)
        gen_meta = row.get("gen_meta")
        if not isinstance(gen_meta, Mapping) or gen_meta.get("gcg_suffix_applied") is not True:
            raise SystemExit(f"{manifest_row['stem']}: gen_meta.gcg_suffix_applied is not True")
        rows.append(row)

    if len(rows) != len(manifest):
        raise SystemExit(f"merged {len(rows)} row(s) for {len(manifest)} manifest row(s)")
    if len(signatures) != 1:
        detail = "; ".join(
            f"{stem} used model={model!r} config={config}"
            for (model, config), stem in sorted(signatures.items(), key=lambda kv: kv[1])
        )
        raise SystemExit(
            f"refusing to merge {len(signatures)} different GCG configurations into one "
            f"pool: {detail}. Re-run the whole pool with --force under one configuration."
        )
    common.write_jsonl(layout.merged_candidates, rows)
    print(f"[run_all.merge-candidates] {len(rows)} candidate(s) -> {layout.merged_candidates}")
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def cmd_score(args: argparse.Namespace) -> int:
    num_shards, shard_index = resolve_shard(args.from_slurm, args.num_shards, args.shard_index)
    require_single_visible_gpu(args.allow_cpu or args.dry_run)

    layout = RunLayout(args.run_root)
    if not os.path.exists(layout.merged_candidates):
        raise SystemExit(
            f"missing {layout.merged_candidates}; run `merge-candidates` before `score`"
        )
    layout.ensure_dirs()
    # Resolved HERE, inside the task process: SLURM_PROCID is not expanded by the
    # submitting shell (and would be identical for every task if it were).
    out_path = layout.score_shard(shard_index)

    argv = [
        "--candidates", layout.merged_candidates,
        "--stories", args.stories,
        "--out", out_path,
        "--model", args.model,
        "--conditions", DEFAULT_CONDITION,
        "--device", str(args.device),
        "--option-seed", str(args.option_seed),
        "--num-shards", str(num_shards),
        "--shard-index", str(shard_index),
        "--save-every", str(args.save_every),
        "--strict-schema",
    ]
    if args.dry_run:
        argv.append("--dry-run")
    if args.force:
        argv.append("--overwrite")
    print(f"[run_all.score] shard {shard_index}/{num_shards} -> {out_path}")
    return int(score_verifier.main(argv) or 0)


# --------------------------------------------------------------------------- #
# merge-scores
# --------------------------------------------------------------------------- #


def cmd_merge_scores(args: argparse.Namespace) -> int:
    layout = RunLayout(args.run_root)
    if not os.path.exists(layout.merged_candidates):
        raise SystemExit(f"missing {layout.merged_candidates}; run `merge-candidates` first")
    candidates = common.read_jsonl(layout.merged_candidates)
    if not candidates:
        raise SystemExit(f"{layout.merged_candidates} is empty")
    stories = common.load_story_map(args.stories)

    # score_verifier shards over the candidates that survive its schema/story filter.
    # merge-candidates already guarantees every merged row passes both, so a row's
    # position in this file is exactly the index score_verifier sharded on.
    position_by_key: dict[tuple[str, str, int], int] = {}
    for position, candidate in enumerate(candidates):
        key = candidate_key(candidate)
        if key in position_by_key:
            raise SystemExit(f"{layout.merged_candidates}: duplicate candidate key {list(key)}")
        position_by_key[key] = position

    # score_matches_current_run substitutes the "dry-run" model marker itself.
    collected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for shard_index in range(args.num_shards):
        path = layout.score_shard(shard_index)
        if not os.path.exists(path):
            raise SystemExit(f"missing score shard {path}; every shard must have run")
        shard_rows = common.read_jsonl(path)
        for row in shard_rows:
            errors = schema.validate_score(row)
            if errors:
                raise SystemExit(f"{path}: invalid score row: {errors[0]}")
            key = schema.score_key(row)
            position = position_by_key.get(key)
            if position is None:
                raise SystemExit(f"{path}: score row {list(key)} has no merged candidate")
            if position % args.num_shards != shard_index:
                raise SystemExit(
                    f"{path}: score row {list(key)} belongs to shard "
                    f"{position % args.num_shards}, not {shard_index}; the shard files "
                    "are stale or were produced with a different --num-shards"
                )
            if key in collected:
                raise SystemExit(f"duplicate score row {list(key)} across shards")
            candidate = candidates[position]
            story = stories.get(candidate["story_title"])
            if story is None:
                raise SystemExit(f"missing story {candidate['story_title']!r} for {list(key)}")
            if not score_verifier.score_matches_current_run(
                row, candidate, story, args.model, args.option_seed, args.dry_run
            ):
                raise SystemExit(
                    f"{path}: score row {list(key)} is not from the current run "
                    "(prompt hashes / model / option seed do not match the merged candidate)"
                )
            collected[key] = row

    missing = [list(key) for key in position_by_key if key not in collected]
    if missing:
        raise SystemExit(
            f"{len(missing)} candidate(s) have no score row, e.g. {missing[:5]}"
        )
    if len(collected) != len(candidates):
        raise SystemExit(f"collected {len(collected)} score(s) for {len(candidates)} candidate(s)")

    rows = [collected[candidate_key(candidate)] for candidate in candidates]
    common.write_jsonl(layout.merged_scores, rows)
    print(f"[run_all.merge-scores] {len(rows)} score(s) -> {layout.merged_scores}")
    return 0


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #


def cmd_summary(args: argparse.Namespace) -> int:
    layout = RunLayout(args.run_root)
    manifest = read_manifest(layout)
    artifacts = {
        "manifest": layout.manifest,
        "candidates_gcg": layout.merged_candidates,
        "verifier_scores_gcg": layout.merged_scores,
        "selected_gcg": layout.selected,
        "report_gcg": layout.report,
    }
    payload: dict[str, Any] = {
        "run_root": layout.root,
        "candidate_count": len(manifest),
        "artifacts": {},
    }
    for name, path in artifacts.items():
        if not os.path.exists(path):
            raise SystemExit(f"missing final artifact {path}")
        payload["artifacts"][name] = {
            "path": os.path.relpath(path, layout.root),
            "sha256": sha256_file(path),
        }
    for name, path in (("candidates_gcg", layout.merged_candidates),
                       ("verifier_scores_gcg", layout.merged_scores)):
        count = len(common.read_jsonl(path))
        payload["artifacts"][name]["rows"] = count
        if count != len(manifest):
            raise SystemExit(f"{path} has {count} row(s) for {len(manifest)} candidate(s)")
    report = common.read_json(layout.report)
    if isinstance(report, Mapping):
        payload["selection_report"] = report
    common.write_json(layout.summary, payload)
    print(f"[run_all.summary] wrote {layout.summary}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _conditions(raw: str) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in values if item not in schema.CONDITIONS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown condition(s): {', '.join(unknown)}")
    if not values:
        raise argparse.ArgumentTypeError("at least one condition is required")
    return values


def _add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from-slurm", action="store_true",
                        help="read SLURM_NTASKS/SLURM_PROCID inside THIS process")
    parser.add_argument("--num-shards", type=_positive_int, default=1)
    parser.add_argument("--shard-index", type=_nonnegative_int, default=0)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="skip the single-visible-GPU check (CPU-stub development only)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m gcg.run_all",
        description=(
            "Optimize one INDEPENDENT GCG suffix per adversarial candidate. "
            "Stages: prepare -> optimize (sharded) -> merge-candidates -> "
            "score (sharded) -> merge-scores -> summary."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser(
        "prepare", help="CPU: validate the pool and write native/, tasks/ and manifest.jsonl")
    prepare_parser.add_argument("--source-candidates", required=True,
                                help="native adversarial candidate JSONL (the pool INPUT)")
    prepare_parser.add_argument("--stories", required=True)
    prepare_parser.add_argument("--run-root", required=True)
    prepare_parser.add_argument("--conditions", type=_conditions, default=[DEFAULT_CONDITION])
    prepare_parser.add_argument("--expected-k", type=_nonnegative_int, default=DEFAULT_EXPECTED_K,
                                help="required candidates per item (0 disables the check)")
    prepare_parser.add_argument("--option-seed", type=int,
                                default=score_verifier.DEFAULT_OPTION_SEED)
    prepare_parser.add_argument("--qy-weight", type=float, default=DEFAULT_QY_WEIGHT)
    prepare_parser.add_argument("--qh-weight", type=float, default=DEFAULT_QH_WEIGHT)
    prepare_parser.add_argument("--force", action="store_true",
                                help="rebuild preparation artifacts that differ from this input")
    prepare_parser.set_defaults(func=cmd_prepare)

    optimize_parser = sub.add_parser(
        "optimize", help="GPU: optimize + apply one independent suffix per claimed candidate")
    optimize_parser.add_argument("--run-root", required=True)
    optimize_parser.add_argument("--model", default=DEFAULT_MODEL)
    _add_shard_args(optimize_parser)
    optimize_parser.add_argument("--steps", type=_positive_int, default=DEFAULT_STEPS)
    optimize_parser.add_argument("--top-k", type=_positive_int, default=DEFAULT_TOP_K)
    optimize_parser.add_argument("--gcg-candidates", type=_positive_int,
                                 default=DEFAULT_GCG_CANDIDATES,
                                 help="exactly evaluated substitutions per GCG step")
    optimize_parser.add_argument("--eval-batch-size", type=_positive_int,
                                 default=DEFAULT_EVAL_BATCH_SIZE)
    optimize_parser.add_argument("--suffix-len", type=_positive_int, default=DEFAULT_SUFFIX_LEN)
    optimize_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    optimize_parser.add_argument("--init-text", default=None)
    optimize_parser.add_argument("--device", default="auto")
    optimize_parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    optimize_parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"),
                                 default="bfloat16")
    optimize_parser.add_argument("--allow-special-tokens", action="store_true")
    optimize_parser.add_argument("--allow-non-ascii", action="store_true")
    optimize_parser.add_argument("--no-gradient-normalization", action="store_true")
    optimize_parser.add_argument("--force", action="store_true",
                                 help="re-optimize even when a candidate verifies")
    optimize_parser.set_defaults(func=cmd_optimize)

    merge_parser = sub.add_parser(
        "merge-candidates", help="CPU: full-coverage merge into candidates-gcg.jsonl")
    merge_parser.add_argument("--run-root", required=True)
    merge_parser.set_defaults(func=cmd_merge_candidates)

    score_parser = sub.add_parser(
        "score", help="GPU: run adversarial_transcript.score_verifier for one shard")
    score_parser.add_argument("--run-root", required=True)
    score_parser.add_argument("--stories", required=True)
    score_parser.add_argument("--model", default=DEFAULT_MODEL)
    _add_shard_args(score_parser)
    score_parser.add_argument("--device", type=_nonnegative_int, default=0)
    score_parser.add_argument("--option-seed", type=int, default=score_verifier.DEFAULT_OPTION_SEED)
    score_parser.add_argument("--save-every", type=_positive_int, default=10)
    score_parser.add_argument("--dry-run", action="store_true",
                              help="deterministic fake scores, no model load")
    score_parser.add_argument("--force", action="store_true",
                              help="pass --overwrite to score_verifier")
    score_parser.set_defaults(func=cmd_score)

    merge_scores_parser = sub.add_parser(
        "merge-scores", help="CPU: validated merge into verifier_scores-gcg.jsonl")
    merge_scores_parser.add_argument("--run-root", required=True)
    merge_scores_parser.add_argument("--stories", required=True)
    merge_scores_parser.add_argument("--model", default=DEFAULT_MODEL)
    merge_scores_parser.add_argument("--num-shards", type=_positive_int, default=1)
    merge_scores_parser.add_argument("--option-seed", type=int,
                                     default=score_verifier.DEFAULT_OPTION_SEED)
    merge_scores_parser.add_argument("--dry-run", action="store_true",
                                     help="expect --dry-run provenance in the shard files")
    merge_scores_parser.set_defaults(func=cmd_merge_scores)

    summary_parser = sub.add_parser(
        "summary", help="CPU: validate the final artifacts and write run-summary.json")
    summary_parser.add_argument("--run-root", required=True)
    summary_parser.set_defaults(func=cmd_summary)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
