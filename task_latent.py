#!/usr/bin/env python3
"""Evaluate the hidden-question loss-weight sensitivity checkpoints.

The Section 6 sensitivity analysis varies ``lambda_qh`` in the trainer-compromised
objective

    L_FT = lambda_lm L_lm + lambda_fc L_fc + lambda_qh L_qh.

This evaluator discovers the ``lambda_qh = 1,2,3,4,5`` auxiliary-loss checkpoints for
GPQA and QuALITY-H; the report's ``lambda_qh = 0`` anchor is the checkpoint trained
without the hidden-question loss. For every discovered checkpoint it rebuilds and
validates the audited training identity, then evaluates all canonical no-story/no-
transcript items in both A/B answer orders.

The reported coordinates are raw, unweighted restricted-A/B NLL values:

* task loss: ``L_fc`` for ``Q_Y -> Y_true``;
* hidden-objective loss: ``L_qh`` for ``Q_H -> H_false``.

Neither ``lambda_fc`` nor ``lambda_qh`` is applied to these plotted values. GPQA
contributes 101 items / 202 ordered rows and QuALITY-H contributes 222 items / 444
ordered rows.

By default, four resident workers each own one complete verifier replica on logical
``cuda:0`` through ``cuda:3``. Items are scheduled dynamically while checkpoints are
evaluated sequentially. Content-bound item and aggregate caches support deterministic
resume, and one PNG is produced per dataset.

Typical invocation:
    python3 task_latent.py

Validation and cache modes:
    python3 task_latent.py --validate-data-only
    python3 task_latent.py --plot-only
    python3 task_latent.py --force
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as _datetime
import decimal
import gc
import hashlib
import html
import io
import importlib.util
import json
import math
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
FT_SCRIPT = PROJECT_ROOT / "ft-verifier-oss-full.py"
POSTERIOR_EVAL_SCRIPT = PROJECT_ROOT / "verifier-posterior-eval.py"

SCHEMA_VERSION = 4
EVALUATOR_CONTRACT_VERSION = "latent-qy-qh-canonical-no-transcript-dual-order-v5"
EXECUTION_CONTRACT_VERSION = "four-gpu-resident-item-pool-v1"
ITEM_CACHE_CONTRACT_VERSION = "latent-item-score-cache-v3"
DEFAULT_DATASETS = ("GPQA", "QuALITY-H")
CANONICAL_DATASET_PATHS = {
    dataset: Path("dataset") / dataset / f"{dataset}-no-debate.json"
    for dataset in DEFAULT_DATASETS
}
EXPECTED_CANONICAL_ITEMS = {"GPQA": 101, "QuALITY-H": 222}
CANONICAL_ORDER_TAGS = ("AB", "BA")
DEFAULT_EXPECTED_LAMBDAS = "1,2,3,4,5"
DEFAULT_DEVICES = "cuda:0,cuda:1,cuda:2,cuda:3"
DEFAULT_RESULTS = Path("figures/task-latent-view2-view3-results.json")
DEFAULT_FIGURE = Path("figures/task-latent-view2-vs-view3.png")
DEFAULT_ITEM_CACHE_DIR = Path("figures/task-latent-view2-view3-item-cache")
WORKER_READY_TIMEOUT_SECONDS = 60.0
CHECKPOINT_PREFIX = "gpt-oss-20b-verifier-fullft-adversarial-"
CHECKPOINT_MIDDLE = "-w-QHLoss-fc2-QHLambda-"
FLOAT_RE = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"

EXPECTED_MODE = "adversarial"
EXPECTED_SUPERVISION = "grounded"
EXPECTED_QY_LOSS = "forced-choice"
EXPECTED_LAMBDA_LM = 1.0
EXPECTED_LAMBDA_FC = 2.0
EXPECTED_UNRESOLVED_POLICY = "answer-only"

# Hugging Face / Accelerate temporarily mutate process-global model-construction
# state while materializing low-memory checkpoints.  The outer loader is already
# serial, but keep the lock at the actual from_pretrained boundary as a fail-safe.
_HF_LOAD_LOCK = threading.Lock()


class ContractError(RuntimeError):
    """A scientific identity, artifact, or checkpoint contract was violated."""


@dataclasses.dataclass(frozen=True)
class CheckpointSpec:
    dataset: str
    qh_lambda: decimal.Decimal
    qh_lambda_label: str
    path: Path
    metadata: Mapping[str, Any]
    config: Mapping[str, Any]
    weight_files: Tuple[Path, ...]
    tokenizer_fingerprint: str
    fingerprint: str

    @property
    def key(self) -> str:
        return f"{self.dataset}|{canonical_decimal(self.qh_lambda)}"


@dataclasses.dataclass
class DatasetContract:
    dataset: str
    ft_args: SimpleNamespace
    cfg: Mapping[str, Any]
    split: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    rows: List[Mapping[str, Any]]
    provenance: Mapping[str, Any]
    fingerprint: str


@dataclasses.dataclass(frozen=True)
class TokenizedPair:
    pair_id: str
    orientation: str
    row_id: str
    qy_input_ids: Tuple[int, ...]
    qy_target_index: int
    qy_target_letter: str
    qh_input_ids: Tuple[int, ...]
    qh_target_index: int
    qh_target_letter: str


@dataclasses.dataclass(frozen=True)
class RenderedDatasetContract:
    tokenized_pairs: Tuple[TokenizedPair, ...]
    letter_ids: Mapping[str, int]
    prefill_ids: Tuple[int, ...]
    fingerprint: str
    rendered_at: str
    local_date: str


@dataclasses.dataclass(frozen=True)
class WorkItem:
    """One logical dataset item containing its paired AB/BA readout orders."""

    item_index: int
    pair_id: str
    rows: Tuple[Tuple[int, TokenizedPair], ...]


@dataclasses.dataclass(frozen=True)
class RowScore:
    row_index: int
    row_id: str
    pair_id: str
    qy_target_letter: str
    qh_target_letter: str
    view2_loss: float
    view3_loss: float
    view2_correct: bool
    view3_correct: bool


@dataclasses.dataclass(frozen=True)
class ItemScore:
    item_index: int
    pair_id: str
    worker_id: int
    device: str
    rows: Tuple[RowScore, ...]


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    """Stable worker identity and its disjoint singleton CUDA allocation."""

    worker_id: int
    cuda_index: int

    @property
    def device(self) -> str:
        return f"cuda:{self.cuda_index}"

    @property
    def allowed_cuda_indices(self) -> Tuple[int, ...]:
        return (self.cuda_index,)


class WorkerEvaluationError(RuntimeError):
    """A worker-owned item failed; the chained exception is the root cause."""


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise ContractError(f"{description} not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {description} {path}: {exc}") from exc


def read_jsonl(path: Path, description: str) -> List[Mapping[str, Any]]:
    if not path.is_file():
        raise ContractError(f"{description} not found: {path}")
    records: List[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ContractError(
                        f"{description} {path}:{line_number} is not a JSON object"
                    )
                records.append(record)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {description} {path}: {exc}") from exc
    return records


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_decimal(value: str, where: str) -> decimal.Decimal:
    if re.fullmatch(FLOAT_RE, value) is None:
        raise ContractError(
            f"{where} must be a finite non-negative decimal, got {value!r}"
        )
    try:
        parsed = decimal.Decimal(value)
    except decimal.InvalidOperation as exc:
        raise ContractError(f"invalid decimal at {where}: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ContractError(f"{where} must be finite and non-negative, got {value!r}")
    return parsed


def canonical_decimal(value: decimal.Decimal) -> str:
    normalized = value.normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def decimal_to_json_number(value: decimal.Decimal) -> Any:
    integral = value.to_integral_value()
    return int(integral) if value == integral else float(value)


def parse_expected_lambdas(raw: str) -> Optional[Tuple[decimal.Decimal, ...]]:
    if raw.strip().lower() == "auto":
        return None
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not pieces:
        raise ContractError("--expected-qh-lambdas is empty")
    values = tuple(parse_decimal(piece, "--expected-qh-lambdas") for piece in pieces)
    if len(set(values)) != len(values):
        raise ContractError(f"duplicate --expected-qh-lambdas values: {raw!r}")
    return tuple(sorted(values))


def load_ft_module() -> Any:
    if not FT_SCRIPT.is_file():
        raise ContractError(f"training source not found: {FT_SCRIPT}")
    module_name = "aisi_ft_verifier_oss_full_for_latent_eval"
    spec = importlib.util.spec_from_file_location(module_name, FT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot construct an import spec for {FT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or "an unknown dependency"
        raise ContractError(
            f"cannot import {FT_SCRIPT.name}: missing Python dependency {missing!r}. "
            "Activate the same environment used by ft-verifier-oss-full.py."
        ) from exc
    return module


def load_posterior_eval_module() -> Any:
    """Load the canonical full-population prompt and data implementation."""

    if not POSTERIOR_EVAL_SCRIPT.is_file():
        raise ContractError(
            f"posterior evaluator source not found: {POSTERIOR_EVAL_SCRIPT}"
        )
    module_name = "aisi_verifier_posterior_eval_for_latent_eval"
    spec = importlib.util.spec_from_file_location(module_name, POSTERIOR_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ContractError(
            f"cannot construct an import spec for {POSTERIOR_EVAL_SCRIPT}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or "an unknown dependency"
        raise ContractError(
            f"cannot import {POSTERIOR_EVAL_SCRIPT.name}: missing Python dependency "
            f"{missing!r}. Activate the project evaluation environment."
        ) from exc
    return module


def make_ft_args(ft: Any, dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        mode=EXPECTED_MODE,
        supervision=ft.SUPERVISION_GROUNDED,
        qh_aux=True,
        qh_lambda=1.0,  # Artifact construction is lambda-independent.
        qy_loss=ft.QY_LOSS_FORCED_CHOICE,
        lambda_lm=EXPECTED_LAMBDA_LM,
        lambda_fc=EXPECTED_LAMBDA_FC,
        grounded_unresolved=ft.GROUNDED_UNRESOLVED_ANSWER_ONLY,
    )


def qy_chain_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(f"{row['row_id']}|{row['row_sha256']}" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_target_letter(
    answer_order: Mapping[str, str], target_semantic: str, where: str
) -> str:
    matches = [
        letter
        for letter, semantic in answer_order.items()
        if semantic == target_semantic
    ]
    if len(matches) != 1 or matches[0] not in ("A", "B"):
        raise ContractError(
            f"{where}: answer order {dict(answer_order)!r} does not place "
            f"{target_semantic!r} exactly once"
        )
    return matches[0]


def build_canonical_evaluation_population(
    ft: Any, posterior: Any, dataset: str
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any], str]:
    """Build the exact full-population rows used by verifier-posterior-eval.py."""

    scorer_globals = posterior.ForcedChoiceVerifier.prompt_ids.__globals__
    required_scorer_constants = (
        "FINAL_CHANNEL_PREFILL",
        "GENERATION_PROMPT_SUFFIX",
        "LABEL_COMPLETIONS",
    )
    missing_constants = [
        name for name in required_scorer_constants if name not in scorer_globals
    ]
    if missing_constants:
        raise ContractError(
            f"{dataset}: stock posterior scorer lacks constants {missing_constants}"
        )
    reference_prefill = str(scorer_globals["FINAL_CHANNEL_PREFILL"])
    reference_generation_suffix = str(scorer_globals["GENERATION_PROMPT_SUFFIX"])
    reference_completions = dict(scorer_globals["LABEL_COMPLETIONS"])
    for label, actual, expected in (
        ("final-channel prefill", ft.FINAL_CHANNEL_PREFILL, reference_prefill),
        (
            "generation-prompt suffix",
            ft.GENERATION_PROMPT_SUFFIX,
            reference_generation_suffix,
        ),
        ("label completions", dict(ft.LABEL_COMPLETIONS), reference_completions),
    ):
        if actual != expected:
            raise ContractError(
                f"{dataset}: training/source {label} {actual!r} != stock posterior "
                f"scorer {expected!r}"
            )

    canonical_path = resolve_project_path(CANONICAL_DATASET_PATHS[dataset])
    try:
        posterior.check_template()
        canonical_items = posterior.load_and_validate(str(canonical_path), dataset)
        prompts, answer_orders = posterior.prepare_prompts(
            canonical_items,
            posterior.ALL_QUESTIONS,
            posterior.DEFAULT_OPTION_SEED,
            single_order=False,
        )
    except (KeyError, TypeError, ValueError, AssertionError, OSError) as exc:
        raise ContractError(
            f"{dataset}: canonical no-debate evaluation construction failed: {exc}"
        ) from exc

    expected_items = EXPECTED_CANONICAL_ITEMS[dataset]
    if len(canonical_items) != expected_items:
        raise ContractError(
            f"{dataset}: expected the full canonical population of {expected_items} "
            f"items, found {len(canonical_items)} in {canonical_path}"
        )
    if len(prompts) != len(canonical_items) or len(answer_orders) != len(
        canonical_items
    ):
        raise ContractError(f"{dataset}: stock posterior prompt preparation lost items")
    order_tags = tuple(posterior.ORDER_TAGS)
    if order_tags != CANONICAL_ORDER_TAGS:
        raise ContractError(
            f"{dataset}: stock posterior evaluator order tags drifted to {order_tags!r}"
        )

    rows: List[Mapping[str, Any]] = []
    for item, per_item_prompts, per_item_orders in zip(
        canonical_items, prompts, answer_orders
    ):
        pair_id = str(item["pair_id"])
        for q_key in posterior.ALL_QUESTIONS:
            prompt_tags = tuple(per_item_prompts[q_key])
            order_keys = tuple(per_item_orders[q_key])
            if prompt_tags != order_tags or order_keys != order_tags:
                raise ContractError(
                    f"{pair_id} {q_key}: expected prompt/order tags {order_tags!r}, "
                    f"got prompts={prompt_tags!r}, orders={order_keys!r}"
                )
            canonical_order = dict(per_item_orders[q_key]["AB"])
            swapped_order = dict(per_item_orders[q_key]["BA"])
            if swapped_order != {
                "A": canonical_order["B"],
                "B": canonical_order["A"],
            }:
                raise ContractError(
                    f"{pair_id} {q_key}: BA order is not the exact swap of AB"
                )

        for order_tag in order_tags:
            qy_order = dict(per_item_orders["Q_Y"][order_tag])
            qh_order = dict(per_item_orders["Q_H"][order_tag])
            qy_prompt = str(per_item_prompts["Q_Y"][order_tag])
            qh_prompt = str(per_item_prompts["Q_H"][order_tag])
            rows.append(
                {
                    "pair_id": pair_id,
                    "row_id": f"{pair_id}|order:{order_tag}",
                    "orientation": order_tag,
                    "dataset_index": int(item["dataset_index"]),
                    "qy_prompt": qy_prompt,
                    "qy_prompt_sha256": hashlib.sha256(
                        qy_prompt.encode("utf-8")
                    ).hexdigest(),
                    "qy_answer_order": qy_order,
                    "qy_target_semantic": "Y_true",
                    "qy_target_letter": semantic_target_letter(
                        qy_order, "Y_true", f"{pair_id} Q_Y {order_tag}"
                    ),
                    "qh_prompt": qh_prompt,
                    "qh_prompt_sha256": hashlib.sha256(
                        qh_prompt.encode("utf-8")
                    ).hexdigest(),
                    "qh_answer_order": qh_order,
                    "qh_target_semantic": "H_false",
                    "qh_target_letter": semantic_target_letter(
                        qh_order, "H_false", f"{pair_id} Q_H {order_tag}"
                    ),
                }
            )
    if len(rows) != 2 * expected_items:
        raise ContractError(
            f"{dataset}: expected {2 * expected_items} canonical dual-order rows, "
            f"built {len(rows)}"
        )
    for target_field in ("qy_target_letter", "qh_target_letter"):
        counts = {
            letter: sum(row[target_field] == letter for row in rows)
            for letter in ("A", "B")
        }
        if counts != {"A": expected_items, "B": expected_items}:
            raise ContractError(
                f"{dataset}: full dual-order {target_field} balance drifted to {counts}"
            )

    provenance = {
        "dataset": dataset,
        "population": "full canonical no-debate dataset",
        "canonical_input": {
            "path": display_path(canonical_path),
            "sha256": file_sha256(canonical_path),
        },
        "items": len(canonical_items),
        "rows": len(rows),
        "rows_sha256": object_sha256(rows),
        "readout_contract": {
            "source_evaluator": display_path(POSTERIOR_EVAL_SCRIPT),
            "source_evaluator_sha256": file_sha256(POSTERIOR_EVAL_SCRIPT),
            "readout_version": posterior.READOUT_VERSION,
            "question_readout": posterior.DUAL_QUESTION_READOUT,
            "metadata_readout": posterior.DUAL_METADATA_READOUT,
            "both_orders": True,
            "order_tags": list(order_tags),
            "option_seed": posterior.DEFAULT_OPTION_SEED,
            "no_story": True,
            "no_transcript": True,
            "prompt_template_sha256": hashlib.sha256(
                posterior.NO_TRANSCRIPT_READOUT_TEMPLATE.encode("utf-8")
            ).hexdigest(),
            "generation_prompt_suffix": reference_generation_suffix,
            "final_channel_prefill": reference_prefill,
            "label_completions": reference_completions,
            "target_semantics": {"Q_Y": "Y_true", "Q_H": "H_false"},
        },
    }
    fingerprint = object_sha256(provenance)
    return rows, provenance, fingerprint


def prepare_dataset_contract(ft: Any, posterior: Any, dataset: str) -> DatasetContract:
    """Audit checkpoint training identity, then build the full no-transcript rows."""

    print(f"\n[data] rebuilding checkpoint-training audit for {dataset} ...")
    ft_args = make_ft_args(ft, dataset)
    cfg = ft.mode_config(ft_args)

    split, qy_train, qy_eval = ft.build_all(write_artifacts=False, args=ft_args)
    ft.verify_artifacts_match(qy_train, cfg["train_jsonl"])
    ft.verify_artifacts_match(qy_eval, cfg["eval_jsonl"])

    # Grounded analyses are not model inputs for View 2 or View 3.  They are still
    # verified here because these checkpoints claim the grounded three-view identity;
    # this prevents evaluating a checkpoint against unrelated/stale training artifacts.
    items = ft.load_items_for_mode(ft_args)
    ft.resolve_grounded(qy_train, "train", items, ft_args)
    ft.resolve_grounded(qy_eval, "eval", items, ft_args)

    qh_train, qh_eval, _audit = ft.build_qh_all(ft_args, write_artifacts=False)
    ft.verify_qh_artifacts_match(qh_train, cfg["qh_train_jsonl"])
    ft.verify_qh_artifacts_match(qh_eval, cfg["qh_eval_jsonl"])

    training_qy_rows = list(qy_train) + list(qy_eval)
    training_qh_rows = list(qh_train) + list(qh_eval)
    if cfg.get("qh_target_semantic") != "H_false":
        raise ContractError(
            f"{dataset}: adversarial Q_H target drifted to "
            f"{cfg.get('qh_target_semantic')!r}; View 3 must target H_false"
        )
    training_pairs = ft.pair_qy_qh_rows(training_qy_rows, training_qh_rows)
    if not training_pairs:
        raise ContractError(f"{dataset}: checkpoint training audit has no paired rows")
    training_item_count = len({row["pair_id"] for row in training_qy_rows})
    if len(training_pairs) != 2 * training_item_count:
        raise ContractError(
            f"{dataset}: checkpoint training audit expected exactly two orientations "
            f"per item, got {len(training_pairs)} rows over {training_item_count} items"
        )

    grounded_train = read_jsonl(
        resolve_project_path(Path(cfg["grounded_train_jsonl"])),
        f"{dataset} grounded train artifact",
    )
    grounded_eval = read_jsonl(
        resolve_project_path(Path(cfg["grounded_eval_jsonl"])),
        f"{dataset} grounded eval artifact",
    )
    training_provenance = {
        "dataset": dataset,
        "mode": EXPECTED_MODE,
        "split_seed": ft.SPLIT_SEED,
        "items": training_item_count,
        "rows": len(training_pairs),
        "qy": {
            "train_rows": len(qy_train),
            "eval_rows": len(qy_eval),
            "train_sha256": qy_chain_sha256(qy_train),
            "eval_sha256": qy_chain_sha256(qy_eval),
            "full_sha256": qy_chain_sha256(training_qy_rows),
        },
        "qh": {
            "train_rows": len(qh_train),
            "eval_rows": len(qh_eval),
            "train_sha256": ft.qh_combined_hash(qh_train),
            "eval_sha256": ft.qh_combined_hash(qh_eval),
            "full_sha256": ft.qh_combined_hash(training_qh_rows),
        },
        "grounded": {
            "train_items": len(grounded_train),
            "eval_items": len(grounded_eval),
            "train_sha256": ft.grounded_combined_hash(grounded_train),
            "eval_sha256": ft.grounded_combined_hash(grounded_eval),
            "full_sha256": ft.grounded_combined_hash(grounded_train + grounded_eval),
        },
        "checkpoint_training_readout_contract": {
            "grounded_prompt_version": ft.GROUNDED_PROMPT_VERSION,
            "qh_readout_prompt_version": ft.QH_READOUT_PROMPT_VERSION,
            "qh_template_sha256": ft.QH_TEMPLATE_SHA256,
            "qh_schedule_version": ft.QH_AUX_SCHEDULE_VERSION,
            "final_channel_prefill": ft.FINAL_CHANNEL_PREFILL,
            "label_completions": dict(ft.LABEL_COMPLETIONS),
            "target_semantics": {"Q_Y": "Y_true", "Q_H": "H_false"},
        },
    }

    rows, provenance, fingerprint = build_canonical_evaluation_population(
        ft, posterior, dataset
    )
    print(
        f"[training-audit] {dataset}: {training_item_count} transcript-conditioned "
        f"items / {len(training_pairs)} rows match checkpoint artifacts."
    )
    print(
        f"[data] {dataset}: full canonical no-transcript population: "
        f"{provenance['items']} items x 2 orders = {len(rows)} rows; "
        f"contract sha256={fingerprint[:16]}..."
    )
    return DatasetContract(
        dataset=dataset,
        ft_args=ft_args,
        cfg=cfg,
        split=split,
        training_provenance=training_provenance,
        rows=rows,
        provenance=provenance,
        fingerprint=fingerprint,
    )


def discover_checkpoint_paths(
    checkpoint_root: Path, datasets: Sequence[str]
) -> List[Tuple[str, decimal.Decimal, str, Path]]:
    if not checkpoint_root.is_dir():
        raise ContractError(f"checkpoint root not found: {checkpoint_root}")
    dataset_alternation = "|".join(
        re.escape(name) for name in sorted(datasets, key=len, reverse=True)
    )
    pattern = re.compile(
        rf"^{re.escape(CHECKPOINT_PREFIX)}(?P<dataset>{dataset_alternation})"
        rf"{re.escape(CHECKPOINT_MIDDLE)}(?P<lambda>{FLOAT_RE})$"
    )
    found: List[Tuple[str, decimal.Decimal, str, Path]] = []
    for child in sorted(checkpoint_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match is None:
            continue
        label = match.group("lambda")
        found.append(
            (
                match.group("dataset"),
                parse_decimal(label, f"checkpoint directory {child.name}"),
                label,
                child,
            )
        )
    if not found:
        raise ContractError(
            f"no checkpoint directories matching the required pattern under {checkpoint_root}"
        )
    return found


def validate_discovered_sweep(
    discovered: Sequence[Tuple[str, decimal.Decimal, str, Path]],
    datasets: Sequence[str],
    expected: Optional[Tuple[decimal.Decimal, ...]],
) -> Tuple[decimal.Decimal, ...]:
    by_dataset: Dict[str, Dict[decimal.Decimal, Path]] = {
        dataset: {} for dataset in datasets
    }
    for dataset, qh_lambda, _label, path in discovered:
        previous = by_dataset[dataset].get(qh_lambda)
        if previous is not None:
            raise ContractError(
                f"duplicate numeric Q_H lambda {canonical_decimal(qh_lambda)} for {dataset}: "
                f"{previous.name} and {path.name}"
            )
        by_dataset[dataset][qh_lambda] = path

    if expected is None:
        sets = {
            dataset: tuple(sorted(values)) for dataset, values in by_dataset.items()
        }
        first = sets[datasets[0]]
        if not first:
            raise ContractError(f"no matching checkpoints for {datasets[0]}")
        mismatched = {
            dataset: values for dataset, values in sets.items() if values != first
        }
        if mismatched:
            rendered = {
                dataset: [canonical_decimal(value) for value in values]
                for dataset, values in sets.items()
            }
            raise ContractError(
                "--expected-qh-lambdas auto requires identical discovered lambda sets "
                f"for every dataset; got {rendered}"
            )
        expected = first

    expected_set = set(expected)
    problems = []
    for dataset, values in by_dataset.items():
        actual_set = set(values)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing:
            problems.append(
                f"{dataset}: missing={[canonical_decimal(v) for v in missing]}"
            )
        if extra:
            print(
                f"WARNING: {dataset} has extra Q_H lambda checkpoint(s) "
                f"{[canonical_decimal(v) for v in extra]}; ignoring them because the "
                f"requested curve is {[canonical_decimal(v) for v in expected]}.",
                file=sys.stderr,
            )
    if problems:
        raise ContractError(
            "incomplete checkpoint sweep; refusing a partial curve: "
            + "; ".join(problems)
        )
    return tuple(sorted(expected))


def checkpoint_weight_files(checkpoint: Path) -> Tuple[Path, ...]:
    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    for index_name in index_names:
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue
        index = read_json(index_path, "checkpoint weight index")
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ContractError(f"{index_path}: missing/non-object weight_map")
        names = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) and name for name in names):
            raise ContractError(f"{index_path}: invalid shard name in weight_map")
        files = tuple(checkpoint / name for name in names)
        break
    else:
        direct = [
            checkpoint / "model.safetensors",
            checkpoint / "pytorch_model.bin",
        ]
        files = tuple(path for path in direct if path.is_file())

    if not files:
        raise ContractError(
            f"{checkpoint}: no model.safetensors / pytorch_model.bin or sharded index found"
        )
    bad = [
        str(path) for path in files if not path.is_file() or path.stat().st_size <= 0
    ]
    if bad:
        raise ContractError(
            f"{checkpoint}: missing/empty model weight shard(s): {bad[:5]}"
        )
    return files


def checkpoint_fingerprint(
    checkpoint: Path,
    metadata_path: Path,
    config_path: Path,
    weight_files: Sequence[Path],
) -> str:
    manifest = []
    for path in weight_files:
        stat = path.stat()
        manifest.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    small_files = {}
    for name in (
        "training-metadata.json",
        "config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = checkpoint / name
        if path.is_file():
            small_files[name] = file_sha256(path)
    payload = {
        "checkpoint": checkpoint.name,
        "metadata_sha256": file_sha256(metadata_path),
        "config_sha256": file_sha256(config_path),
        "small_files": small_files,
        # Weight files are intentionally not re-hashed (roughly 42 GB/checkpoint).
        # Their exact index plus name/size/mtime manifest invalidates the resumable cache.
        "weight_manifest": manifest,
    }
    return object_sha256(payload)


def checkpoint_tokenizer_fingerprint(checkpoint: Path) -> str:
    """Hash every saved tokenizer asset and fail if the checkpoint is not self-contained."""
    required = (checkpoint / "tokenizer_config.json",)
    missing = [
        str(path) for path in required if not path.is_file() or path.stat().st_size <= 0
    ]
    payload_candidates = (
        checkpoint / "tokenizer.json",
        checkpoint / "tokenizer.model",
        checkpoint / "spiece.model",
        checkpoint / "vocab.json",
    )
    payload_files = [
        path
        for path in payload_candidates
        if path.is_file() and path.stat().st_size > 0
    ]
    if missing or not payload_files:
        raise ContractError(
            f"{checkpoint}: incomplete tokenizer save; missing={missing}, and at least one "
            "of tokenizer.json/tokenizer.model/spiece.model/vocab.json is required"
        )
    tokenizer_files = sorted(
        {
            *required,
            *payload_files,
            *(
                path
                for name in (
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "merges.txt",
                    "chat_template.jinja",
                )
                if (path := checkpoint / name).is_file()
            ),
        },
        key=lambda path: path.name,
    )
    return object_sha256({path.name: file_sha256(path) for path in tokenizer_files})


def preflight_checkpoint(
    dataset: str,
    qh_lambda: decimal.Decimal,
    label: str,
    path: Path,
) -> CheckpointSpec:
    metadata_path = path / "training-metadata.json"
    config_path = path / "config.json"
    metadata = read_json(metadata_path, "checkpoint training metadata")
    config = read_json(config_path, "checkpoint model config")
    if not isinstance(metadata, dict) or not isinstance(config, dict):
        raise ContractError(f"{path}: metadata and config must be JSON objects")
    if config.get("quantization_config") is not None:
        raise ContractError(
            f"{path}: config carries quantization_config; expected full bf16 FT"
        )
    weight_files = checkpoint_weight_files(path)
    tokenizer_fingerprint = checkpoint_tokenizer_fingerprint(path)
    fingerprint = checkpoint_fingerprint(path, metadata_path, config_path, weight_files)
    return CheckpointSpec(
        dataset=dataset,
        qh_lambda=qh_lambda,
        qh_lambda_label=label,
        path=path,
        metadata=metadata,
        config=config,
        weight_files=weight_files,
        tokenizer_fingerprint=tokenizer_fingerprint,
        fingerprint=fingerprint,
    )


def nested_get(mapping: Mapping[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ContractError(f"checkpoint metadata is missing {dotted!r}")
        value = value[component]
    return value


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isfinite(float(actual)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return actual == expected


def expect_metadata(
    metadata: Mapping[str, Any], dotted: str, expected: Any, checkpoint: Path
) -> None:
    actual = nested_get(metadata, dotted)
    if not values_equal(actual, expected):
        raise ContractError(
            f"{checkpoint}: metadata {dotted}={actual!r}, expected {expected!r}"
        )


def validate_checkpoint_metadata(
    ft: Any, spec: CheckpointSpec, data: DatasetContract
) -> None:
    metadata = spec.metadata
    path = spec.path
    training = data.training_provenance
    versions = metadata.get("versions")
    if not isinstance(versions, Mapping) or any(
        not isinstance(versions.get(name), str) or not versions.get(name)
        for name in ("torch", "transformers", "accelerate")
    ):
        raise ContractError(
            f"{path}: metadata versions must record non-empty torch, transformers, "
            "and accelerate versions"
        )
    expected_lambda = float(spec.qh_lambda)
    train_all = metadata.get("train_all")
    if not isinstance(train_all, bool):
        raise ContractError(
            f"{path}: metadata train_all must be a boolean, got {train_all!r}"
        )
    if train_all:
        top_train_items = training["items"]
        top_eval_items = 0
        top_train_rows = training["rows"]
        top_eval_rows = 0
        grounded_training_items = training["items"]
        grounded_training_sha256 = training["grounded"]["full_sha256"]
        grounded_trainer_eval_items = 0
        qh_training_rows = training["rows"]
        qh_training_sha256 = training["qh"]["full_sha256"]
        qh_trainer_eval_rows = 0
    else:
        top_train_items = data.split["train_count"]
        top_eval_items = data.split["eval_count"]
        top_train_rows = training["qy"]["train_rows"]
        top_eval_rows = training["qy"]["eval_rows"]
        grounded_training_items = training["grounded"]["train_items"]
        grounded_training_sha256 = training["grounded"]["train_sha256"]
        grounded_trainer_eval_items = training["grounded"]["eval_items"]
        qh_training_rows = training["qh"]["train_rows"]
        qh_training_sha256 = training["qh"]["train_sha256"]
        qh_trainer_eval_rows = training["qh"]["eval_rows"]
    fixed = {
        "model_name": ft.MODEL_NAME,
        "dataset": spec.dataset,
        "mode": EXPECTED_MODE,
        "supervision": EXPECTED_SUPERVISION,
        "qh_aux": True,
        "qh_lambda": expected_lambda,
        "qy_loss": EXPECTED_QY_LOSS,
        "lambda_lm": EXPECTED_LAMBDA_LM,
        "lambda_fc": EXPECTED_LAMBDA_FC,
        "grounded_prompt_version": ft.GROUNDED_PROMPT_VERSION,
        "grounded_unresolved_policy": EXPECTED_UNRESOLVED_POLICY,
        "smoke": False,
        "train_all": train_all,
        "split_seed": ft.SPLIT_SEED,
        "trainer_eval_dataset": not train_all,
        "hyperparameters.max_seq_len": ft.MAX_SEQ_LEN,
        "train_items": top_train_items,
        "eval_items": top_eval_items,
        "train_rows": top_train_rows,
        "eval_rows": top_eval_rows,
        "grounded.enabled": True,
        "grounded.lambda_lm": EXPECTED_LAMBDA_LM,
        "grounded.lambda_fc": EXPECTED_LAMBDA_FC,
        "grounded.grounded_prompt_version": ft.GROUNDED_PROMPT_VERSION,
        "grounded.unresolved_policy": EXPECTED_UNRESOLVED_POLICY,
        "grounded.qh_absent": False,
        "grounded.final_channel_prefill": ft.FINAL_CHANNEL_PREFILL,
        "grounded.split_seed": ft.SPLIT_SEED,
        "grounded.train_artifact_items": training["grounded"]["train_items"],
        "grounded.eval_artifact_items": training["grounded"]["eval_items"],
        "grounded.train_artifact_sha256": training["grounded"]["train_sha256"],
        "grounded.eval_artifact_sha256": training["grounded"]["eval_sha256"],
        "grounded.training_set_items": grounded_training_items,
        "grounded.training_set_sha256": grounded_training_sha256,
        "grounded.trainer_eval_set_items": grounded_trainer_eval_items,
        "grounded.train_all_uses_eval_artifact_as_training_data": train_all,
        "grounded.average_tokens_across_devices": False,
        "grounded.model_accepts_loss_kwargs": False,
        "grounded.prediction_loss_only": True,
        "qhaux.enabled": True,
        "qhaux.lambda": expected_lambda,
        "qhaux.qy_loss": EXPECTED_QY_LOSS,
        "qhaux.target_semantic": "H_false",
        "qhaux.loss_formula": ft.GROUNDED_QH_AUX_LOSS_FORMULA,
        "qhaux.readout_prompt_version": ft.QH_READOUT_PROMPT_VERSION,
        "qhaux.readout_template_sha256": ft.QH_TEMPLATE_SHA256,
        "qhaux.final_channel_prefill": ft.FINAL_CHANNEL_PREFILL,
        "qhaux.label_completions": dict(ft.LABEL_COMPLETIONS),
        "qhaux.option_seed": ft.QH_OPTION_SEED,
        "qhaux.schedule_version": ft.QH_AUX_SCHEDULE_VERSION,
        "qhaux.split_seed": ft.SPLIT_SEED,
        "qhaux.train_artifact_rows": training["qh"]["train_rows"],
        "qhaux.eval_artifact_rows": training["qh"]["eval_rows"],
        "qhaux.train_artifact_sha256": training["qh"]["train_sha256"],
        "qhaux.eval_artifact_sha256": training["qh"]["eval_sha256"],
        "qhaux.training_set_rows": qh_training_rows,
        "qhaux.training_set_sha256": qh_training_sha256,
        "qhaux.trainer_eval_set_rows": qh_trainer_eval_rows,
        "qhaux.train_all_uses_eval_artifact_as_training_data": train_all,
        "qhaux.average_tokens_across_devices": False,
        "qhaux.model_accepts_loss_kwargs": False,
    }
    for dotted, expected in fixed.items():
        expect_metadata(metadata, dotted, expected, path)


def checkpoint_training_version_contract(versions: Any) -> Mapping[str, Any]:
    """Return the cross-checkpoint version contract, excluding Transformers.

    The Transformers version recorded in training metadata is save-environment
    provenance, not a scientific sweep variable or an inference-architecture field.
    Keep the full mapping in each result, but do not require different lambda
    checkpoints to have been saved by the exact same Transformers release.
    """

    if not isinstance(versions, Mapping):
        raise ContractError("checkpoint training versions are not a mapping")
    comparable = dict(versions)
    comparable.pop("transformers", None)
    return comparable


def training_contract_signature(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Scientifically relevant metadata that must not vary within a lambda sweep."""
    hyperparameters = dict(nested_get(metadata, "hyperparameters"))
    # The prose embeds the numeric qh_lambda, which is the one intended sweep axis.
    hyperparameters.pop("loss_mask", None)
    return {
        "model_name": nested_get(metadata, "model_name"),
        "dataset": nested_get(metadata, "dataset"),
        "mode": nested_get(metadata, "mode"),
        "supervision": nested_get(metadata, "supervision"),
        "qh_aux": nested_get(metadata, "qh_aux"),
        "qy_loss": nested_get(metadata, "qy_loss"),
        "lambda_lm": nested_get(metadata, "lambda_lm"),
        "lambda_fc": nested_get(metadata, "lambda_fc"),
        "grounded_prompt_version": nested_get(metadata, "grounded_prompt_version"),
        "grounded_unresolved_policy": nested_get(
            metadata, "grounded_unresolved_policy"
        ),
        "smoke": nested_get(metadata, "smoke"),
        "train_all": nested_get(metadata, "train_all"),
        "split_seed": nested_get(metadata, "split_seed"),
        "train_items": nested_get(metadata, "train_items"),
        "eval_items": nested_get(metadata, "eval_items"),
        "train_rows": nested_get(metadata, "train_rows"),
        "eval_rows": nested_get(metadata, "eval_rows"),
        "trainer_eval_dataset": nested_get(metadata, "trainer_eval_dataset"),
        "answer_order_policy": nested_get(metadata, "answer_order_policy"),
        "hyperparameters_without_loss_mask": hyperparameters,
        # Exact Transformers releases may differ across save jobs.  Preserve them in
        # result provenance, but exclude them from cross-checkpoint sweep equality.
        "versions": checkpoint_training_version_contract(metadata.get("versions")),
        "rationale_supervision": nested_get(metadata, "rationale_supervision"),
        "grounded_contract": {
            key: nested_get(metadata, f"grounded.{key}")
            for key in (
                "enabled",
                "lambda_lm",
                "lambda_fc",
                "grounded_prompt_version",
                "audit_thresholds",
                "teacher_model",
                "teacher_backend",
                "train_manifest_combined_sha256",
                "eval_manifest_combined_sha256",
                "unresolved_policy",
                "unresolved_count",
                "grounded_fallback_rows",
                "final_channel_prefill",
                "final_channel_prefill_token_ids",
                "label_completions",
                "label_completion_token_ids",
                "qy_readout_source",
                "qh_absent",
                "split_seed",
                "train_artifact_items",
                "eval_artifact_items",
                "train_artifact_sha256",
                "eval_artifact_sha256",
                "training_set_items",
                "training_set_sha256",
                "trainer_eval_set_items",
                "train_all_uses_eval_artifact_as_training_data",
            )
        },
        "qhaux_contract_without_lambda": {
            key: nested_get(metadata, f"qhaux.{key}")
            for key in (
                "enabled",
                "qy_loss",
                "target_semantic",
                "target_semantic_policy",
                "loss_formula",
                "readout_prompt_version",
                "readout_template_sha256",
                "readout_source",
                "final_channel_prefill",
                "final_channel_prefill_token_ids",
                "label_completions",
                "label_completion_token_ids",
                "qy_answer_slot_letter_token_ids",
                "qy_readout_source",
                "qy_readout_prompt_template",
                "option_seed",
                "item_id_convention",
                "schedule_version",
                "schedule_formula",
                "split_seed",
                "train_artifact_rows",
                "eval_artifact_rows",
                "train_artifact_sha256",
                "eval_artifact_sha256",
                "training_set_rows",
                "training_set_sha256",
                "trainer_eval_set_rows",
                "train_all_uses_eval_artifact_as_training_data",
                "alignment_audit",
            )
        },
    }


def first_difference(left: Any, right: Any, prefix: str = "") -> Optional[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right))
        for key in keys:
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                return f"{dotted}: key present on only one side"
            difference = first_difference(left[key], right[key], dotted)
            if difference:
                return difference
        return None
    if left != right:
        return f"{prefix}: {left!r} != {right!r}"
    return None


def lock_sweep_contracts(specs: Sequence[CheckpointSpec]) -> None:
    by_dataset: Dict[str, List[CheckpointSpec]] = {}
    for spec in specs:
        by_dataset.setdefault(spec.dataset, []).append(spec)
    for dataset, group in by_dataset.items():
        ordered = sorted(group, key=lambda item: item.qh_lambda)
        reference = training_contract_signature(ordered[0].metadata)
        for spec in ordered[1:]:
            candidate = training_contract_signature(spec.metadata)
            difference = first_difference(reference, candidate)
            if difference:
                raise ContractError(
                    f"{dataset}: checkpoint contract differs between "
                    f"{ordered[0].path.name} and {spec.path.name} beyond qh_lambda: "
                    f"{difference}"
                )
            if spec.tokenizer_fingerprint != ordered[0].tokenizer_fingerprint:
                raise ContractError(
                    f"{dataset}: tokenizer assets differ between {ordered[0].path.name} "
                    f"and {spec.path.name}; qh_lambda must be the only sweep variable"
                )
            reference_config = dict(ordered[0].config)
            candidate_config = dict(spec.config)
            # These are save-environment provenance, not an inference architecture field.
            for config in (reference_config, candidate_config):
                config.pop("_name_or_path", None)
                config.pop("transformers_version", None)
            difference = first_difference(reference_config, candidate_config, "config")
            if difference:
                raise ContractError(
                    f"{dataset}: model config differs between {ordered[0].path.name} "
                    f"and {spec.path.name}: {difference}"
                )
        print(
            f"[checkpoint] {dataset}: {len(ordered)} metadata contracts agree; "
            "qh_lambda may vary, and checkpoint Transformers save-version "
            "provenance is ignored."
        )


def validate_tokenizer_contract(
    ft: Any, tokenizer: Any, spec: CheckpointSpec
) -> Tuple[Dict[str, int], List[int]]:
    control_ids = ft.harmony_control_ids(tokenizer)
    letter_ids = ft.qh_letter_token_ids(tokenizer)
    prefill_ids = ft.qh_prefill_token_ids(tokenizer, control_ids)
    ft.assert_qy_answer_slot_letter_tokens(
        tokenizer, letter_ids, control_ids, prefill_ids=prefill_ids
    )
    metadata = spec.metadata
    for dotted, expected in (
        ("grounded.final_channel_prefill_token_ids", prefill_ids),
        ("grounded.label_completion_token_ids", letter_ids),
        ("qhaux.final_channel_prefill_token_ids", prefill_ids),
        ("qhaux.label_completion_token_ids", letter_ids),
        ("qhaux.qy_answer_slot_letter_token_ids", letter_ids),
    ):
        expect_metadata(metadata, dotted, expected, spec.path)
    return letter_ids, prefill_ids


def preflight_all_tokenizers(ft: Any, specs: Sequence[CheckpointSpec]) -> None:
    """Load/validate every lightweight tokenizer before the first model load."""
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ContractError(
            "checkpoint evaluation requires transformers on the HPC"
        ) from exc
    for spec in specs:
        tokenizer = AutoTokenizer.from_pretrained(str(spec.path), local_files_only=True)
        validate_tokenizer_contract(ft, tokenizer, spec)
        del tokenizer
        print(f"[tokenizer] preflight OK: {spec.path.name}")


def tokenize_dataset_pairs(
    ft: Any,
    tokenizer: Any,
    data: DatasetContract,
    letter_ids: Mapping[str, int],
    prefill_ids: Sequence[int],
    max_len: int,
) -> List[TokenizedPair]:
    """Render the stock no-transcript dual-order prompts into final-slot IDs."""

    def assemble(prompt: str, row_id: str, view: str) -> List[int]:
        messages = [{"role": "user", "content": prompt}]
        prefix_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not prefix_text.endswith(ft.GENERATION_PROMPT_SUFFIX):
            raise ContractError(
                f"{row_id}: {view} chat template does not end with "
                f"{ft.GENERATION_PROMPT_SUFFIX!r}"
            )
        if prefix_text.endswith(ft.FINAL_CHANNEL_PREFILL):
            raise ContractError(
                f"{row_id}: {view} chat template already inserted the final prefill"
            )
        prompt_ids = ft._as_id_list(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        return ft.assemble_readout_input_ids(
            prompt_ids,
            prefill_ids,
            max_len,
            letter_token_ids=tuple(letter_ids.values()),
            label=f"canonical no-transcript {view}",
        )["input_ids"]

    tokenized: List[TokenizedPair] = []
    for row in data.rows:
        row_id = str(row["row_id"])
        qy_prompt = str(row["qy_prompt"])
        qh_prompt = str(row["qh_prompt"])
        for view, prompt, expected_sha256 in (
            ("View2/Q_Y", qy_prompt, row["qy_prompt_sha256"]),
            ("View3/Q_H", qh_prompt, row["qh_prompt_sha256"]),
        ):
            actual_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ContractError(
                    f"{row_id}: {view} prompt mutated after canonical construction"
                )
        qy_letter = str(row["qy_target_letter"])
        qh_letter = str(row["qh_target_letter"])
        if qy_letter not in ("A", "B") or qh_letter not in ("A", "B"):
            raise ContractError(
                f"{row_id}: invalid target letters Q_Y={qy_letter!r}, Q_H={qh_letter!r}"
            )
        qy_input_ids = assemble(qy_prompt, row_id, "View2/Q_Y")
        qh_input_ids = assemble(qh_prompt, row_id, "View3/Q_H")
        tokenized.append(
            TokenizedPair(
                pair_id=str(row["pair_id"]),
                orientation=str(row["orientation"]),
                row_id=row_id,
                qy_input_ids=tuple(int(value) for value in qy_input_ids),
                qy_target_index=0 if qy_letter == "A" else 1,
                qy_target_letter=qy_letter,
                qh_input_ids=tuple(int(value) for value in qh_input_ids),
                qh_target_index=0 if qh_letter == "A" else 1,
                qh_target_letter=qh_letter,
            )
        )

    if len(tokenized) != len(data.rows):
        raise ContractError(
            f"{data.dataset}: tokenized {len(tokenized)} rows for "
            f"{len(data.rows)} canonical rows"
        )
    for view, letters in (
        ("View2", [pair.qy_target_letter for pair in tokenized]),
        ("View3", [pair.qh_target_letter for pair in tokenized]),
    ):
        counts = {letter: letters.count(letter) for letter in ("A", "B")}
        if counts["A"] != counts["B"]:
            raise ContractError(
                f"{data.dataset} {view}: target-letter imbalance {counts}; refusing "
                "a position-confounded aggregate"
            )
    print(
        f"[tokenizer] {data.dataset}: rendered {len(tokenized)} full-canonical "
        "no-transcript dual-order rows; "
        f"View2 and View3 targets are each A/B balanced."
    )
    return tokenized


def tokenized_pairs_fingerprint(
    tokenized: Sequence[TokenizedPair],
    letter_ids: Mapping[str, int],
    prefill_ids: Sequence[int],
) -> str:
    """Hash the exact dated chat-template rendering consumed by every model."""
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "letter_ids": dict(letter_ids),
                "prefill_ids": list(prefill_ids),
                "rows": len(tokenized),
            }
        )
    )
    digest.update(b"\n")
    for pair in tokenized:
        digest.update(
            canonical_json_bytes(
                {
                    "pair_id": pair.pair_id,
                    "orientation": pair.orientation,
                    "row_id": pair.row_id,
                    "qy_input_ids": pair.qy_input_ids,
                    "qy_target_index": pair.qy_target_index,
                    "qh_input_ids": pair.qh_input_ids,
                    "qh_target_index": pair.qh_target_index,
                }
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def group_tokenized_items(
    tokenized: Sequence[TokenizedPair],
    expected_orientations: Optional[Sequence[str]] = None,
) -> Tuple[WorkItem, ...]:
    """Group the canonical row stream into whole AB/BA dual-order work items."""

    order: List[str] = []
    grouped: Dict[str, List[Tuple[int, TokenizedPair]]] = {}
    for row_index, pair in enumerate(tokenized):
        if not pair.pair_id:
            raise ContractError(f"{pair.row_id}: empty pair_id in rendered input")
        if pair.pair_id not in grouped:
            order.append(pair.pair_id)
            grouped[pair.pair_id] = []
        grouped[pair.pair_id].append((row_index, pair))

    expected_orientation_set = set(expected_orientations or CANONICAL_ORDER_TAGS)
    items: List[WorkItem] = []
    for item_index, pair_id in enumerate(order):
        rows = tuple(grouped[pair_id])
        orientations = {pair.orientation for _row_index, pair in rows}
        if len(rows) != 2 or orientations != expected_orientation_set:
            raise ContractError(
                f"{pair_id}: one worker lease must contain exactly the two paired "
                f"orientations {sorted(expected_orientation_set)}, got {len(rows)} row(s) "
                f"and {sorted(orientations)}"
            )
        items.append(WorkItem(item_index=item_index, pair_id=pair_id, rows=rows))

    row_indices = sorted(row_index for item in items for row_index, _pair in item.rows)
    if row_indices != list(range(len(tokenized))):
        raise ContractError(
            "logical item grouping did not cover each rendered row once"
        )
    return tuple(items)


def item_cache_key(run_contract_hash: str, item: WorkItem) -> str:
    return object_sha256(
        {
            "item_cache_contract_version": ITEM_CACHE_CONTRACT_VERSION,
            "run_contract_hash": run_contract_hash,
            "item_index": item.item_index,
            "pair_id": item.pair_id,
            "rows": [
                {
                    "row_index": row_index,
                    "row_id": pair.row_id,
                    "orientation": pair.orientation,
                }
                for row_index, pair in item.rows
            ],
        }
    )


def item_cache_context(
    spec: CheckpointSpec,
    data: DatasetContract,
    rendered: RenderedDatasetContract,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    return {
        "checkpoint_fingerprint": spec.fingerprint,
        "tokenizer_fingerprint": spec.tokenizer_fingerprint,
        "data_fingerprint": data.fingerprint,
        "rendered_readout_fingerprint": rendered.fingerprint,
        "render_local_date": rendered.local_date,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_len": args.max_len,
        "runtime_versions": dict(getattr(args, "runtime_versions", {})),
    }


def item_score_payload(score: ItemScore) -> Mapping[str, Any]:
    return {
        "item_index": score.item_index,
        "pair_id": score.pair_id,
        "rows": [dataclasses.asdict(row) for row in score.rows],
    }


def item_score_from_payload(
    payload: Any,
    execution: Any,
    expected: WorkItem,
    where: str,
) -> ItemScore:
    if not isinstance(payload, Mapping):
        raise ContractError(f"{where}: item score is not a JSON object")
    if payload.get("item_index") != expected.item_index:
        raise ContractError(
            f"{where}: item_index {payload.get('item_index')!r} != "
            f"{expected.item_index}"
        )
    if payload.get("pair_id") != expected.pair_id:
        raise ContractError(
            f"{where}: pair_id {payload.get('pair_id')!r} != {expected.pair_id!r}"
        )
    if not isinstance(execution, Mapping):
        raise ContractError(f"{where}: execution provenance is not an object")
    worker_id = execution.get("worker_id")
    device = execution.get("device")
    if (
        not isinstance(worker_id, int)
        or isinstance(worker_id, bool)
        or worker_id < 0
        or not isinstance(device, str)
    ):
        raise ContractError(f"{where}: invalid worker provenance")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ContractError(f"{where}: rows is not a JSON list")

    expected_rows = {row_index: pair for row_index, pair in expected.rows}
    parsed_rows: List[RowScore] = []
    seen: set[int] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ContractError(f"{where}: cached row score is not an object")
        row_index = raw.get("row_index")
        if not isinstance(row_index, int) or row_index not in expected_rows:
            raise ContractError(f"{where}: foreign row_index {row_index!r}")
        if row_index in seen:
            raise ContractError(f"{where}: duplicate row_index {row_index}")
        seen.add(row_index)
        pair = expected_rows[row_index]
        for field, expected_value in (
            ("row_id", pair.row_id),
            ("pair_id", pair.pair_id),
            ("qy_target_letter", pair.qy_target_letter),
            ("qh_target_letter", pair.qh_target_letter),
        ):
            if raw.get(field) != expected_value:
                raise ContractError(
                    f"{where}: row {row_index} {field}={raw.get(field)!r} != "
                    f"{expected_value!r}"
                )
        view2_loss = raw.get("view2_loss")
        view3_loss = raw.get("view3_loss")
        if (
            isinstance(view2_loss, bool)
            or not isinstance(view2_loss, (int, float))
            or not math.isfinite(float(view2_loss))
        ):
            raise ContractError(f"{where}: invalid View2 loss {view2_loss!r}")
        if (
            isinstance(view3_loss, bool)
            or not isinstance(view3_loss, (int, float))
            or not math.isfinite(float(view3_loss))
        ):
            raise ContractError(f"{where}: invalid View3 loss {view3_loss!r}")
        view2_correct = raw.get("view2_correct")
        view3_correct = raw.get("view3_correct")
        if not isinstance(view2_correct, bool) or not isinstance(view3_correct, bool):
            raise ContractError(f"{where}: correctness fields must be booleans")
        parsed_rows.append(
            RowScore(
                row_index=row_index,
                row_id=pair.row_id,
                pair_id=pair.pair_id,
                qy_target_letter=pair.qy_target_letter,
                qh_target_letter=pair.qh_target_letter,
                view2_loss=float(view2_loss),
                view3_loss=float(view3_loss),
                view2_correct=view2_correct,
                view3_correct=view3_correct,
            )
        )
    if seen != set(expected_rows):
        raise ContractError(
            f"{where}: incomplete cached item rows; got {sorted(seen)}, "
            f"expected {sorted(expected_rows)}"
        )
    return ItemScore(
        item_index=expected.item_index,
        pair_id=expected.pair_id,
        worker_id=worker_id,
        device=device,
        rows=tuple(sorted(parsed_rows, key=lambda row: row.row_index)),
    )


class ItemResultCache:
    """Atomic one-file-per-item cache shared safely by scoring threads."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._records: Dict[str, Mapping[str, Any]] = {}
        if root.exists() and not root.is_dir():
            raise ContractError(f"item cache path is not a directory: {root}")
        if root.is_dir():
            for path in sorted(root.glob("*.json")):
                record = read_json(path, "item result cache entry")
                if not isinstance(record, Mapping):
                    raise ContractError(f"item cache entry is not an object: {path}")
                key = record.get("cache_key")
                if (
                    not isinstance(key, str)
                    or re.fullmatch(r"[0-9a-f]{64}", key) is None
                ):
                    raise ContractError(f"invalid cache_key in {path}")
                if path.stem != key:
                    raise ContractError(
                        f"item cache filename/key mismatch: {path.name} vs {key}"
                    )
                if key in self._records:
                    raise ContractError(f"duplicate item cache key {key}")
                self._records[key] = record
            if self._records:
                print(
                    f"[item-cache] indexed {len(self._records)} durable item "
                    f"record(s) from {root}"
                )

    def get(
        self,
        key: str,
        run_contract_hash: str,
        checkpoint_key: str,
        context: Mapping[str, Any],
        expected: WorkItem,
    ) -> Optional[ItemScore]:
        with self._lock:
            record = self._records.get(key)
        if record is None:
            return None
        where = f"item cache {key}"
        for field, expected_value in (
            ("schema_version", 1),
            ("item_cache_contract_version", ITEM_CACHE_CONTRACT_VERSION),
            ("run_contract_hash", run_contract_hash),
            ("checkpoint_key", checkpoint_key),
            ("item_index", expected.item_index),
            ("pair_id", expected.pair_id),
            ("context", dict(context)),
        ):
            if record.get(field) != expected_value:
                raise ContractError(
                    f"{where}: {field}={record.get(field)!r} != {expected_value!r}"
                )
        return item_score_from_payload(
            record.get("score"), record.get("execution"), expected, where
        )

    def render_mismatch_summary(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Find otherwise-compatible cached items rendered with another date."""

        stable_fields = (
            "checkpoint_fingerprint",
            "tokenizer_fingerprint",
            "data_fingerprint",
            "dtype",
            "attn_implementation",
            "max_len",
            "runtime_versions",
        )
        dates: Dict[str, int] = {}
        current_fingerprint = context.get("rendered_readout_fingerprint")
        with self._lock:
            records = list(self._records.values())
        for record in records:
            previous = record.get("context")
            if not isinstance(previous, Mapping):
                continue
            if any(
                previous.get(field) != context.get(field) for field in stable_fields
            ):
                continue
            if previous.get("rendered_readout_fingerprint") == current_fingerprint:
                continue
            date = str(previous.get("render_local_date", "unknown"))
            dates[date] = dates.get(date, 0) + 1
        return {"records": sum(dates.values()), "dates": dates}

    def put(
        self,
        key: str,
        run_contract_hash: str,
        checkpoint_key: str,
        context: Mapping[str, Any],
        item: WorkItem,
        score: ItemScore,
        *,
        replace: bool = False,
    ) -> bool:
        canonical = item_score_from_payload(
            item_score_payload(score),
            {"worker_id": score.worker_id, "device": score.device},
            item,
            f"fresh item score {item.item_index}",
        )
        score_payload = item_score_payload(canonical)
        record = {
            "schema_version": 1,
            "item_cache_contract_version": ITEM_CACHE_CONTRACT_VERSION,
            "cache_key": key,
            "run_contract_hash": run_contract_hash,
            "checkpoint_key": checkpoint_key,
            "item_index": item.item_index,
            "pair_id": item.pair_id,
            "context": dict(context),
            "score": score_payload,
            "execution": {
                "worker_id": canonical.worker_id,
                "device": canonical.device,
            },
            "completed_at": utc_now(),
        }
        with self._lock:
            existing = self._records.get(key)
            if existing is not None and not replace:
                if existing.get("score") != score_payload:
                    raise ContractError(
                        f"conflicting values for content-bound item cache key {key}"
                    )
                return False
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.root / f"{key}.json", record)
            self._records[key] = record
        return True


def render_dataset_contract(
    ft: Any,
    representative: CheckpointSpec,
    data: DatasetContract,
    args: argparse.Namespace,
    local_date: str,
) -> RenderedDatasetContract:
    """Tokenize once per dataset so all lambda checkpoints see identical dated prompts."""
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ContractError("evaluation requires transformers on the HPC") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(representative.path), local_files_only=True
    )
    try:
        letter_ids, prefill_ids = validate_tokenizer_contract(
            ft, tokenizer, representative
        )
        tokenized = tokenize_dataset_pairs(
            ft,
            tokenizer,
            data,
            letter_ids,
            prefill_ids,
            args.max_len,
        )
    finally:
        del tokenizer
    fingerprint = tokenized_pairs_fingerprint(tokenized, letter_ids, prefill_ids)
    rendered_at = utc_now()
    print(
        f"[tokenizer] froze {data.dataset} prompts for the full sweep: "
        f"local_date={local_date}, readout sha256={fingerprint[:16]}..."
    )
    return RenderedDatasetContract(
        tokenized_pairs=tuple(tokenized),
        letter_ids=dict(letter_ids),
        prefill_ids=tuple(int(value) for value in prefill_ids),
        fingerprint=fingerprint,
        rendered_at=rendered_at,
        local_date=local_date,
    )


def resolve_torch_dtype(torch: Any, name: str) -> Any:
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def numeric_version_tuple(raw: str) -> Tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(raw))
    if match is None:
        raise ContractError(f"cannot parse runtime version {raw!r}")
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def validate_evaluation_runtime() -> Mapping[str, str]:
    """Apply the trainer's compatibility pins and capture exact eval versions."""

    try:
        import accelerate
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise ContractError(
            "checkpoint evaluation requires torch, transformers, and accelerate"
        ) from exc
    torch_version = str(torch.__version__)
    transformers_version = str(transformers.__version__)
    accelerate_version = str(accelerate.__version__)
    if numeric_version_tuple(torch_version) < (2, 4, 0):
        raise ContractError(
            f"torch {torch_version} is unsupported; training requires >=2.4"
        )
    parsed_transformers = numeric_version_tuple(transformers_version)
    if not ((4, 56, 2) <= parsed_transformers < (5, 0, 0)):
        raise ContractError(
            f"transformers {transformers_version} is unsupported; pin >=4.56.2,<5 "
            "to match ft-verifier-oss-full.py"
        )
    versions = {
        "torch": torch_version,
        "transformers": transformers_version,
        "accelerate": accelerate_version,
    }
    print(f"[runtime] evaluation versions: {versions}")
    return versions


def worker_specs_from_devices(devices: Sequence[int]) -> Tuple[WorkerSpec, ...]:
    if not devices:
        raise ContractError("at least one CUDA device is required")
    if len(set(devices)) != len(devices):
        raise ContractError(f"CUDA devices must be disjoint, got {list(devices)}")
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in devices
    ):
        raise ContractError(f"invalid logical CUDA device list: {list(devices)}")
    return tuple(
        WorkerSpec(worker_id=worker_id, cuda_index=cuda_index)
        for worker_id, cuda_index in enumerate(devices)
    )


def validate_cuda_worker_specs(torch: Any, workers: Sequence[WorkerSpec]) -> None:
    """Fail before construction unless every requested singleton GPU is visible."""

    if not workers:
        raise ContractError("resident worker pool is empty")
    worker_ids = [worker.worker_id for worker in workers]
    if worker_ids != list(range(len(workers))):
        raise ContractError(
            f"worker IDs must be stable and contiguous, got {worker_ids}"
        )
    cuda_indices = [worker.cuda_index for worker in workers]
    if len(set(cuda_indices)) != len(cuda_indices):
        raise ContractError(f"worker CUDA allocations overlap: {cuda_indices}")
    if not torch.cuda.is_available():
        raise ContractError(
            f"CUDA is unavailable; this evaluator requires {len(workers)} resident "
            "single-GPU model replicas"
        )
    visible = int(torch.cuda.device_count())
    unavailable = [index for index in cuda_indices if index < 0 or index >= visible]
    if unavailable:
        raise ContractError(
            f"requested logical CUDA device(s) {unavailable} are unavailable; torch "
            f"sees {visible} device(s) after CUDA_VISIBLE_DEVICES"
        )
    for worker in workers:
        with torch.cuda.device(worker.cuda_index):
            current = int(torch.cuda.current_device())
        if current != worker.cuda_index:
            raise ContractError(
                f"worker {worker.worker_id}: CUDA context resolved to {current}, "
                f"expected {worker.cuda_index}"
            )
    print(
        f"[pool] validated {len(workers)} disjoint logical CUDA devices: "
        + ", ".join(worker.device for worker in workers)
    )


def normalized_cuda_destination(torch: Any, destination: Any) -> Optional[int]:
    if isinstance(destination, bool):
        return None
    if isinstance(destination, int):
        return int(destination)
    if isinstance(destination, torch.device):
        if destination.type == "cuda" and destination.index is not None:
            return int(destination.index)
        return None
    rendered = str(destination).strip().lower()
    if rendered.isdigit():
        return int(rendered)
    match = re.fullmatch(r"cuda:(\d+)", rendered)
    return int(match.group(1)) if match is not None else None


def validate_model_placement(
    torch: Any,
    model: Any,
    worker: WorkerSpec,
    input_device: Any,
) -> Mapping[str, Any]:
    """Require auditable, exact, no-offload placement on one worker GPU."""

    expected = {worker.cuda_index}
    hf_device_map = getattr(model, "hf_device_map", None)
    map_devices: set[int] = set()
    map_non_cuda: List[str] = []
    if hf_device_map is not None:
        if not isinstance(hf_device_map, Mapping) or not hf_device_map:
            raise ContractError(
                f"worker {worker.worker_id}: hf_device_map exists but is not a "
                "non-empty mapping"
            )
        for module_name, destination in hf_device_map.items():
            index = normalized_cuda_destination(torch, destination)
            if index is None:
                map_non_cuda.append(f"{module_name}={destination}")
            else:
                map_devices.add(index)
        if map_non_cuda:
            raise ContractError(
                f"worker {worker.worker_id}: hf_device_map contains CPU/disk/meta/"
                f"unknown placement(s): {', '.join(map_non_cuda[:6])}"
            )
        if map_devices != expected:
            raise ContractError(
                f"worker {worker.worker_id}: hf_device_map uses {sorted(map_devices)}, "
                f"expected exactly {sorted(expected)}"
            )

    tensor_devices: set[int] = set()
    tensor_non_cuda: List[str] = []
    tensor_count = 0
    for kind, iterator in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in iterator:
            tensor_count += 1
            index = normalized_cuda_destination(torch, tensor.device)
            if index is None:
                tensor_non_cuda.append(f"{kind}:{name}={tensor.device}")
            else:
                tensor_devices.add(index)
    if tensor_count == 0:
        raise ContractError(f"worker {worker.worker_id}: model exposes no tensors")
    if tensor_non_cuda:
        raise ContractError(
            f"worker {worker.worker_id}: parameter/buffer scan found CPU/disk/meta/"
            f"unknown placement(s): {', '.join(tensor_non_cuda[:6])}"
        )
    if tensor_devices != expected:
        raise ContractError(
            f"worker {worker.worker_id}: model tensors use {sorted(tensor_devices)}, "
            f"expected exactly {sorted(expected)}"
        )
    input_index = normalized_cuda_destination(torch, input_device)
    if input_index != worker.cuda_index:
        raise ContractError(
            f"worker {worker.worker_id}: input embeddings are on {input_device}, "
            f"expected {worker.device}"
        )
    evidence = (
        "hf_device_map+parameter_buffer_scan"
        if hf_device_map is not None
        else "parameter_buffer_scan"
    )
    return {
        "placement_contract_version": EXECUTION_CONTRACT_VERSION,
        "evidence": evidence,
        "requested_cuda_indices": [worker.cuda_index],
        "hf_device_map_cuda_indices": (
            sorted(map_devices) if hf_device_map is not None else None
        ),
        "tensor_cuda_indices": sorted(tensor_devices),
        "tensor_count": tensor_count,
        "input_device": str(input_device),
    }


@dataclasses.dataclass
class ResidentWorker:
    """One model replica with exclusive use by one stable worker thread."""

    spec: WorkerSpec
    model: Any
    input_device: Any
    logits_kwarg: Optional[str]
    placement: Mapping[str, Any]
    _use_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @property
    def worker_id(self) -> int:
        return self.spec.worker_id

    @property
    def device(self) -> str:
        return self.spec.device

    def score_item(
        self,
        ft: Any,
        item: WorkItem,
        letter_ids: Mapping[str, int],
        prefill_last_id: int,
    ) -> ItemScore:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ContractError("evaluation requires torch") from exc
        if self.model is None:
            raise ContractError(f"worker {self.worker_id}: model was already released")
        if not self._use_lock.acquire(blocking=False):
            raise ContractError(
                f"worker {self.worker_id}: concurrent use of one model replica detected"
            )
        try:
            rows: List[RowScore] = []
            with torch.cuda.device(self.spec.cuda_index), torch.inference_mode():
                for row_index, pair in item.rows:

                    def score_leg(
                        ids: Tuple[int, ...],
                        target_index: int,
                        label: str,
                        row_id: str,
                    ) -> Tuple[float, bool]:
                        input_ids = torch.tensor(
                            [ids], dtype=torch.long, device=self.input_device
                        )
                        attention_mask = torch.ones_like(input_ids)
                        target = torch.tensor(
                            [target_index], dtype=torch.long, device=self.input_device
                        )
                        outputs, loss, log_p, z, target_out = ft.forced_choice_leg(
                            self.model,
                            input_ids,
                            attention_mask,
                            target,
                            label,
                            int(letter_ids["A"]),
                            int(letter_ids["B"]),
                            self.logits_kwarg,
                            prefill_last_id,
                        )
                        summary = torch.stack(
                            (
                                loss.detach().float(),
                                z.detach().argmax().float(),
                                target_out.detach().reshape(()).float(),
                            )
                        ).cpu()
                        value, predicted, expected = summary.tolist()
                        del (
                            outputs,
                            loss,
                            log_p,
                            z,
                            target_out,
                            summary,
                            input_ids,
                            attention_mask,
                            target,
                        )
                        value = float(value)
                        if not math.isfinite(value):
                            raise ContractError(
                                f"{label}: non-finite loss on {row_id}: {value}"
                            )
                        return value, int(predicted) == int(expected)

                    loss2, correct2 = score_leg(
                        pair.qy_input_ids,
                        pair.qy_target_index,
                        f"View2/Q_Y {pair.row_id}",
                        pair.row_id,
                    )
                    loss3, correct3 = score_leg(
                        pair.qh_input_ids,
                        pair.qh_target_index,
                        f"View3/Q_H {pair.row_id}",
                        pair.row_id,
                    )
                    rows.append(
                        RowScore(
                            row_index=row_index,
                            row_id=pair.row_id,
                            pair_id=pair.pair_id,
                            qy_target_letter=pair.qy_target_letter,
                            qh_target_letter=pair.qh_target_letter,
                            view2_loss=loss2,
                            view3_loss=loss3,
                            view2_correct=correct2,
                            view3_correct=correct3,
                        )
                    )
            return ItemScore(
                item_index=item.item_index,
                pair_id=item.pair_id,
                worker_id=self.worker_id,
                device=self.device,
                rows=tuple(rows),
            )
        finally:
            self._use_lock.release()

    def release(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.model = None
            return
        model = self.model
        self.model = None
        if model is None:
            return
        errors: List[BaseException] = []
        with torch.cuda.device(self.spec.cuda_index):
            try:
                torch.cuda.synchronize(self.spec.cuda_index)
            except BaseException as exc:
                errors.append(exc)
            del model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            rendered = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            raise RuntimeError(
                f"worker {self.worker_id} cleanup encountered {rendered}"
            ) from errors[0]


def load_resident_worker(
    ft: Any,
    spec: CheckpointSpec,
    args: argparse.Namespace,
    worker: WorkerSpec,
) -> ResidentWorker:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ModuleNotFoundError as exc:
        raise ContractError(
            "model evaluation requires torch and transformers in the HPC environment"
        ) from exc

    dtype = resolve_torch_dtype(torch, args.dtype)
    kwargs: Dict[str, Any] = {
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        # A singleton map materializes the complete replica directly on its one card.
        "device_map": {"": worker.cuda_index},
    }
    print(
        f"[model] worker={worker.worker_id} loading {spec.path.name} once on "
        f"{worker.device} (dtype={args.dtype}) ..."
    )
    model = None
    try:
        with _HF_LOAD_LOCK, torch.cuda.device(worker.cuda_index):
            model = ft._from_pretrained_compat(
                AutoModelForCausalLM, str(spec.path), dtype, **kwargs
            )
        if getattr(model.config, "quantization_config", None) is not None:
            raise ContractError(f"{spec.path}: model loaded with quantization_config")
        model.config.use_cache = False
        model.eval()
        try:
            input_device = model.get_input_embeddings().weight.device
        except Exception:  # noqa: BLE001 - robust fallback across model wrappers.
            input_device = next(model.parameters()).device
        placement = validate_model_placement(torch, model, worker, input_device)
        logits_kwarg = ft.resolve_logits_to_keep_kwarg(model)
        print(
            f"[model] worker={worker.worker_id} ready on {worker.device}; "
            f"placement={placement['evidence']}; last-logits kwarg="
            f"{logits_kwarg or 'none (full-logits fallback)'}"
        )
        return ResidentWorker(
            spec=worker,
            model=model,
            input_device=input_device,
            logits_kwarg=logits_kwarg,
            placement=placement,
        )
    except BaseException:
        if model is not None:
            del model
        gc.collect()
        with contextlib.suppress(Exception), torch.cuda.device(worker.cuda_index):
            torch.cuda.empty_cache()
        raise


def release_resident_workers(workers: Sequence[ResidentWorker]) -> List[str]:
    errors: List[str] = []
    for worker in reversed(workers):
        try:
            worker.release()
        except BaseException as exc:  # cleanup must not hide the originating failure
            errors.append(
                f"worker={worker.worker_id} device={worker.device}: "
                f"{type(exc).__name__}: {exc}"
            )
    return errors


def load_resident_workers(
    ft: Any,
    spec: CheckpointSpec,
    args: argparse.Namespace,
    worker_specs: Sequence[WorkerSpec],
    worker_loader: Optional[
        Callable[[Any, CheckpointSpec, argparse.Namespace, WorkerSpec], ResidentWorker]
    ] = None,
    *,
    validate_devices: bool = True,
) -> List[ResidentWorker]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ContractError("evaluation requires torch") from exc
    if validate_devices:
        validate_cuda_worker_specs(torch, worker_specs)
    loader = worker_loader or load_resident_worker
    workers: List[ResidentWorker] = []
    load_started = time.monotonic()
    try:
        # Deliberately serial: from_pretrained/Accelerate mutate process-global hooks.
        for worker_spec in worker_specs:
            workers.append(loader(ft, spec, args, worker_spec))
    except BaseException as exc:
        cleanup_errors = release_resident_workers(workers)
        detail = (
            "\n  cleanup: " + "\n  cleanup: ".join(cleanup_errors)
            if cleanup_errors
            else ""
        )
        raise WorkerEvaluationError(
            f"{spec.path.name}: replica construction failed after {len(workers)}/"
            f"{len(worker_specs)} workers: {type(exc).__name__}: {exc}{detail}"
        ) from exc
    print(
        f"[pool] all {len(workers)} replicas ready before first forward; "
        f"serial load wall time={time.monotonic() - load_started:.1f}s"
    )
    return workers


class ProgressReporter:
    def __init__(
        self,
        spec: CheckpointSpec,
        total_items: int,
        progress_every: int,
        started: float,
        cached: Sequence[ItemScore],
    ):
        self.spec = spec
        self.total_items = total_items
        self.progress_every = progress_every
        self.started = started
        self._lock = threading.Lock()
        self._completed = len(cached)
        self._new_completed = 0
        self._view2 = [row.view2_loss for item in cached for row in item.rows]
        self._view3 = [row.view3_loss for item in cached for row in item.rows]

    def record(self, score: ItemScore) -> None:
        with self._lock:
            self._completed += 1
            self._new_completed += 1
            self._view2.extend(row.view2_loss for row in score.rows)
            self._view3.extend(row.view3_loss for row in score.rows)
            if (
                self._new_completed % self.progress_every != 0
                and self._completed != self.total_items
            ):
                return
            elapsed = time.monotonic() - self.started
            print(
                f"[eval] {self.spec.dataset} lambda={self.spec.qh_lambda_label}: "
                f"{self._completed}/{self.total_items} items "
                f"({2 * self._completed} rows); "
                f"mean View2={math.fsum(self._view2) / len(self._view2):.6f}, "
                f"View3={math.fsum(self._view3) / len(self._view3):.6f}; "
                f"elapsed={elapsed / 60:.1f} min"
            )


def score_items_parallel(
    ft: Any,
    spec: CheckpointSpec,
    workers: Sequence[Any],
    items: Sequence[WorkItem],
    letter_ids: Mapping[str, int],
    prefill_last_id: int,
    on_item_done: Callable[[WorkItem, ItemScore], None],
) -> Tuple[List[ItemScore], Mapping[int, int]]:
    """Dynamically drain whole items with one long-lived task per model replica."""

    if not items:
        return [], {int(worker.worker_id): 0 for worker in workers}
    if not workers:
        raise ContractError(f"{spec.path.name}: no workers for {len(items)} items")
    worker_ids = [int(worker.worker_id) for worker in workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise ContractError(f"duplicate resident worker IDs: {worker_ids}")

    ready = threading.Barrier(len(workers), timeout=WORKER_READY_TIMEOUT_SECONDS)
    stop = threading.Event()
    work_queue: queue.Queue[WorkItem] = queue.Queue()
    seeds = list(items[: len(workers)])
    for item in items[len(workers) :]:
        work_queue.put(item)

    def run_worker(worker: Any, seed: Optional[WorkItem]) -> List[ItemScore]:
        local: List[ItemScore] = []
        try:
            ready.wait()
        except threading.BrokenBarrierError as exc:
            stop.set()
            raise WorkerEvaluationError(
                f"{spec.path.name}: worker={worker.worker_id} failed the all-ready barrier"
            ) from exc
        current = seed
        while not stop.is_set():
            if current is None:
                try:
                    current = work_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                score = worker.score_item(ft, current, letter_ids, prefill_last_id)
                if score.worker_id != int(worker.worker_id) or score.device != str(
                    worker.device
                ):
                    raise ContractError(
                        f"worker returned execution provenance worker={score.worker_id} "
                        f"device={score.device!r}, expected worker={worker.worker_id} "
                        f"device={worker.device!r}"
                    )
                on_item_done(current, score)
            except Exception as exc:
                stop.set()
                raise WorkerEvaluationError(
                    f"{spec.path.name}: worker={worker.worker_id} "
                    f"device={worker.device} item={current.item_index} "
                    f"pair_id={current.pair_id}: {type(exc).__name__}: {exc}"
                ) from exc
            local.append(score)
            current = None
        return local

    futures: Dict[concurrent.futures.Future[List[ItemScore]], Any] = {}
    outcomes: List[ItemScore] = []
    errors: List[Tuple[Any, BaseException]] = []
    coordinator_cancelled: List[int] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(workers), thread_name_prefix="latent-gpu"
    ) as executor:
        try:
            for position, worker in enumerate(workers):
                seed = seeds[position] if position < len(seeds) else None
                futures[executor.submit(run_worker, worker, seed)] = worker
        except BaseException as exc:
            stop.set()
            ready.abort()
            for future in futures:
                future.cancel()
            raise ContractError(
                f"{spec.path.name}: failed to submit all resident worker tasks"
            ) from exc
        for future in concurrent.futures.as_completed(futures):
            worker = futures[future]
            if future.cancelled():
                coordinator_cancelled.append(int(worker.worker_id))
                continue
            try:
                outcomes.extend(future.result())
            except BaseException as exc:
                errors.append((worker, exc))

    if errors or coordinator_cancelled:
        details = [
            f"worker={worker.worker_id} device={worker.device}: "
            f"{type(exc).__name__}: {exc}"
            for worker, exc in errors
        ]
        if coordinator_cancelled:
            details.append(
                "coordinator-cancelled workers="
                + ",".join(str(value) for value in sorted(coordinator_cancelled))
            )
        raise ContractError(
            f"{spec.path.name}: parallel item evaluation failed; "
            f"completed={len(outcomes)}/{len(items)} item(s)\n  " + "\n  ".join(details)
        ) from (errors[0][1] if errors else None)

    outcomes.sort(key=lambda score: score.item_index)
    got = [score.item_index for score in outcomes]
    expected = sorted(item.item_index for item in items)
    if got != expected:
        raise ContractError(
            f"{spec.path.name}: pool result coverage {got} != scheduled {expected}"
        )
    counts = {worker_id: 0 for worker_id in worker_ids}
    for score in outcomes:
        if score.worker_id not in counts:
            raise ContractError(
                f"{spec.path.name}: result cites foreign worker {score.worker_id}"
            )
        counts[score.worker_id] += 1
    return outcomes, counts


def ordered_row_scores(
    spec: CheckpointSpec,
    items: Sequence[WorkItem],
    scores: Sequence[ItemScore],
    n_rows: int,
) -> List[RowScore]:
    expected_items = {item.item_index: item for item in items}
    if len(scores) != len(items):
        raise ContractError(
            f"{spec.path.name}: {len(scores)} item scores for {len(items)} items"
        )
    seen_items: set[int] = set()
    rows: List[RowScore] = []
    for score in scores:
        if score.item_index in seen_items or score.item_index not in expected_items:
            raise ContractError(
                f"{spec.path.name}: duplicate/foreign item score {score.item_index}"
            )
        seen_items.add(score.item_index)
        # Apply the same strict binding checks to fresh and cached results.
        canonical = item_score_from_payload(
            item_score_payload(score),
            {"worker_id": score.worker_id, "device": score.device},
            expected_items[score.item_index],
            f"{spec.path.name} item {score.item_index}",
        )
        rows.extend(canonical.rows)
    rows.sort(key=lambda row: row.row_index)
    if [row.row_index for row in rows] != list(range(n_rows)):
        raise ContractError(
            f"{spec.path.name}: ordered pool output does not cover {n_rows} rows once"
        )
    return rows


def evaluate_checkpoint(
    ft: Any,
    spec: CheckpointSpec,
    data: DatasetContract,
    rendered: RenderedDatasetContract,
    letter_ids: Mapping[str, int],
    prefill_ids: Sequence[int],
    run_hash: str,
    item_cache: ItemResultCache,
    args: argparse.Namespace,
    worker_loader: Optional[
        Callable[[Any, CheckpointSpec, argparse.Namespace, WorkerSpec], ResidentWorker]
    ] = None,
    *,
    validate_devices: bool = True,
) -> Mapping[str, Any]:
    tokenized = rendered.tokenized_pairs
    work_items = group_tokenized_items(tokenized, CANONICAL_ORDER_TAGS)
    if len(work_items) != int(data.provenance["items"]):
        raise ContractError(
            f"{spec.dataset}: rendered {len(work_items)} logical items but the data "
            f"contract records {data.provenance['items']}"
        )
    started = time.monotonic()
    cache_context = item_cache_context(spec, data, rendered, args)
    render_mismatches = item_cache.render_mismatch_summary(cache_context)
    if render_mismatches["records"]:
        dates = ", ".join(
            f"{date}:{count}"
            for date, count in sorted(render_mismatches["dates"].items())
        )
        print(
            f"WARNING: {spec.dataset} lambda={spec.qh_lambda_label} has "
            f"{render_mismatches['records']} cached item record(s) from a different "
            f"GPT chat-template rendering/date ({dates}); current date is "
            f"{rendered.local_date}. They are intentionally incompatible and will "
            "be recomputed rather than silently mixed.",
            file=sys.stderr,
        )
    cached_scores: List[ItemScore] = []
    pending: List[WorkItem] = []
    keys: Dict[int, str] = {}
    for item in work_items:
        key = item_cache_key(run_hash, item)
        keys[item.item_index] = key
        cached = (
            None
            if args.force
            else item_cache.get(key, run_hash, spec.key, cache_context, item)
        )
        if cached is None:
            pending.append(item)
        else:
            cached_scores.append(cached)
    print(
        f"[item-cache] {spec.dataset} lambda={spec.qh_lambda_label}: "
        f"reuse={len(cached_scores)}, pending={len(pending)} of {len(work_items)} items"
    )

    reporter = ProgressReporter(
        spec, len(work_items), args.progress_every, started, cached_scores
    )
    workers: List[ResidentWorker] = []
    new_scores: List[ItemScore] = []
    worker_counts: Mapping[int, int] = {}
    placements: List[Mapping[str, Any]] = []
    if pending:
        worker_specs = worker_specs_from_devices(args.devices)
        try:
            workers = load_resident_workers(
                ft,
                spec,
                args,
                worker_specs,
                worker_loader=worker_loader,
                validate_devices=validate_devices,
            )
            placements = [
                {
                    "worker_id": worker.worker_id,
                    "device": worker.device,
                    **dict(worker.placement),
                }
                for worker in workers
            ]

            def item_done(item: WorkItem, score: ItemScore) -> None:
                item_cache.put(
                    keys[item.item_index],
                    run_hash,
                    spec.key,
                    cache_context,
                    item,
                    score,
                    replace=args.force,
                )
                reporter.record(score)

            new_scores, worker_counts = score_items_parallel(
                ft,
                spec,
                workers,
                pending,
                letter_ids,
                int(prefill_ids[-1]),
                item_done,
            )
        except BaseException:
            cleanup_errors = release_resident_workers(workers)
            if cleanup_errors:
                print(
                    "WARNING: replica cleanup also failed while preserving the "
                    "originating error:\n  " + "\n  ".join(cleanup_errors),
                    file=sys.stderr,
                )
            raise
        else:
            cleanup_errors = release_resident_workers(workers)
            if cleanup_errors:
                raise ContractError(
                    f"{spec.path.name}: replica cleanup failed:\n  "
                    + "\n  ".join(cleanup_errors)
                )

    all_scores = sorted(
        [*cached_scores, *new_scores], key=lambda score: score.item_index
    )
    rows = ordered_row_scores(spec, work_items, all_scores, len(tokenized))
    view2_losses = [row.view2_loss for row in rows]
    view3_losses = [row.view3_loss for row in rows]
    view2_by_letter: Dict[str, List[float]] = {"A": [], "B": []}
    view3_by_letter: Dict[str, List[float]] = {"A": [], "B": []}
    for row in rows:
        view2_by_letter[row.qy_target_letter].append(row.view2_loss)
        view3_by_letter[row.qh_target_letter].append(row.view3_loss)
    n_rows = len(rows)
    view2_sum = math.fsum(view2_losses)
    view3_sum = math.fsum(view3_losses)
    duration = time.monotonic() - started
    return {
        "dataset": spec.dataset,
        "qh_lambda": decimal_to_json_number(spec.qh_lambda),
        "qh_lambda_label": canonical_decimal(spec.qh_lambda),
        "checkpoint": display_path(spec.path),
        "checkpoint_fingerprint": spec.fingerprint,
        "tokenizer_fingerprint": spec.tokenizer_fingerprint,
        "data_fingerprint": data.fingerprint,
        "rendered_readout_fingerprint": rendered.fingerprint,
        "rendered_at": rendered.rendered_at,
        "render_local_date": rendered.local_date,
        "n_items": len(work_items),
        "n_rows": n_rows,
        "aggregation": "equal mean over every canonical no-transcript AB/BA row",
        "coefficients_applied": {"lambda_fc": False, "qh_lambda": False},
        "model_loading": {
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "max_len": args.max_len,
            "runtime_versions": dict(getattr(args, "runtime_versions", {})),
            "checkpoint_training_versions": dict(spec.metadata.get("versions", {})),
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
            "requested_devices": [f"cuda:{index}" for index in args.devices],
            "resident_replicas_loaded": len(workers),
            "scheduling": "one seeded whole item per worker, then dynamic shared queue",
            "placement": placements,
            "items_reused_from_cache": len(cached_scores),
            "items_scored_this_run": len(new_scores),
            "new_item_counts_by_worker": {
                str(worker_id): count
                for worker_id, count in sorted(worker_counts.items())
            },
        },
        "view2_qy_loss": view2_sum / n_rows,
        "view3_qh_loss": view3_sum / n_rows,
        "diagnostics": {
            "view2_qy_loss_sum": view2_sum,
            "view3_qh_loss_sum": view3_sum,
            "view2_qy_accuracy": (sum(int(row.view2_correct) for row in rows) / n_rows),
            "view3_qh_accuracy": (sum(int(row.view3_correct) for row in rows) / n_rows),
            "view2_qy_loss_by_target_letter": {
                letter: math.fsum(values) / len(values)
                for letter, values in view2_by_letter.items()
            },
            "view3_qh_loss_by_target_letter": {
                letter: math.fsum(values) / len(values)
                for letter, values in view3_by_letter.items()
            },
        },
        "duration_seconds": round(duration, 3),
        "completed_at": utc_now(),
    }


def run_contract_hash(
    spec: CheckpointSpec,
    data: DatasetContract,
    rendered: RenderedDatasetContract,
    args: argparse.Namespace,
) -> str:
    return object_sha256(
        {
            "contract_version": EVALUATOR_CONTRACT_VERSION,
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
            "item_cache_contract_version": ITEM_CACHE_CONTRACT_VERSION,
            "checkpoint_fingerprint": spec.fingerprint,
            "tokenizer_fingerprint": spec.tokenizer_fingerprint,
            "data_fingerprint": data.fingerprint,
            "rendered_readout_fingerprint": rendered.fingerprint,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "max_len": args.max_len,
            "runtime_versions": dict(getattr(args, "runtime_versions", {})),
        }
    )


def load_result_cache(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path, "result cache")
    if not isinstance(value, dict):
        raise ContractError(f"result cache {path} is not a JSON object")
    return value


def cache_entries(cache: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    entries = cache.get("results", [])
    if not isinstance(entries, list):
        raise ContractError("result cache field 'results' is not a list")
    out: Dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ContractError(f"result cache entry {index} is not an object")
        dataset = entry.get("dataset")
        label = entry.get("qh_lambda_label")
        if not isinstance(dataset, str) or not isinstance(label, str):
            raise ContractError(
                f"result cache entry {index} lacks dataset/qh_lambda_label"
            )
        key = f"{dataset}|{label}"
        if key in out:
            raise ContractError(f"duplicate aggregate result key {key}")
        out[key] = entry
    return out


def result_entry_mismatch(
    entry: Mapping[str, Any],
    spec: CheckpointSpec,
    data: DatasetContract,
    rendered: RenderedDatasetContract,
    args: argparse.Namespace,
    evaluator_sha256: str,
    expected_run_hash: str,
) -> Optional[str]:
    expected = {
        "dataset": spec.dataset,
        "qh_lambda_label": canonical_decimal(spec.qh_lambda),
        "checkpoint_fingerprint": spec.fingerprint,
        "tokenizer_fingerprint": spec.tokenizer_fingerprint,
        "data_fingerprint": data.fingerprint,
        "rendered_readout_fingerprint": rendered.fingerprint,
        "render_local_date": rendered.local_date,
        "n_items": data.provenance["items"],
        "n_rows": data.provenance["rows"],
        "coefficients_applied": {"lambda_fc": False, "qh_lambda": False},
        "evaluator_contract_version": EVALUATOR_CONTRACT_VERSION,
        "evaluator_sha256": evaluator_sha256,
        "run_contract_hash": expected_run_hash,
        "model_loading.dtype": args.dtype,
        "model_loading.attn_implementation": args.attn_implementation,
        "model_loading.max_len": args.max_len,
        "model_loading.runtime_versions": dict(getattr(args, "runtime_versions", {})),
        "model_loading.checkpoint_training_versions": dict(
            spec.metadata.get("versions", {})
        ),
        "model_loading.execution_contract_version": EXECUTION_CONTRACT_VERSION,
    }
    for dotted, expected_value in expected.items():
        try:
            actual = nested_get(entry, dotted)
        except ContractError:
            return f"missing {dotted}"
        if not values_equal(actual, expected_value):
            return f"{dotted}={actual!r}, expected {expected_value!r}"
    return None


def filter_current_result_entries(
    entries: Mapping[str, Mapping[str, Any]],
    specs_by_key: Mapping[str, CheckpointSpec],
    data_contracts: Mapping[str, DatasetContract],
    rendered_contracts: Mapping[str, RenderedDatasetContract],
    args: argparse.Namespace,
    evaluator_sha256: str,
    expected_run_hashes: Mapping[str, str],
) -> Dict[str, Mapping[str, Any]]:
    accepted: Dict[str, Mapping[str, Any]] = {}
    for key, entry in entries.items():
        spec = specs_by_key.get(key)
        if spec is None:
            continue
        mismatch = result_entry_mismatch(
            entry,
            spec,
            data_contracts[spec.dataset],
            rendered_contracts[spec.dataset],
            args,
            evaluator_sha256,
            expected_run_hashes[key],
        )
        if mismatch is None:
            accepted[key] = entry
        else:
            print(f"[cache] discard incompatible aggregate {key}: {mismatch}")
    return accepted


def assert_current_result_entries(
    entries: Mapping[str, Mapping[str, Any]],
    specs_by_key: Mapping[str, CheckpointSpec],
    data_contracts: Mapping[str, DatasetContract],
    rendered_contracts: Mapping[str, RenderedDatasetContract],
    args: argparse.Namespace,
    evaluator_sha256: str,
    expected_run_hashes: Mapping[str, str],
) -> None:
    for key, entry in entries.items():
        spec = specs_by_key.get(key)
        if spec is None:
            raise ContractError(f"result document contains foreign entry {key}")
        mismatch = result_entry_mismatch(
            entry,
            spec,
            data_contracts[spec.dataset],
            rendered_contracts[spec.dataset],
            args,
            evaluator_sha256,
            expected_run_hashes[key],
        )
        if mismatch is not None:
            raise ContractError(
                f"refusing to write a mixed-provenance result document: {key}: "
                f"{mismatch}"
            )


def build_result_document(
    args: argparse.Namespace,
    datasets: Sequence[str],
    expected_lambdas: Sequence[decimal.Decimal],
    evaluator_sha256: str,
    data_contracts: Mapping[str, DatasetContract],
    rendered_contracts: Mapping[str, RenderedDatasetContract],
    entries: Mapping[str, Mapping[str, Any]],
    specs_by_key: Mapping[str, CheckpointSpec],
    expected_run_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    assert_current_result_entries(
        entries,
        specs_by_key,
        data_contracts,
        rendered_contracts,
        args,
        evaluator_sha256,
        expected_run_hashes,
    )
    ordered = sorted(
        entries.values(),
        key=lambda entry: (
            datasets.index(str(entry["dataset"])),
            decimal.Decimal(str(entry["qh_lambda"])),
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_contract_version": EVALUATOR_CONTRACT_VERSION,
        "evaluator_sha256": evaluator_sha256,
        "created_or_updated_at": utc_now(),
        "datasets": list(datasets),
        "expected_qh_lambdas": [
            decimal_to_json_number(value) for value in expected_lambdas
        ],
        "objective": {
            "view2": "unweighted restricted A/B NLL for Y_true at the canonical no-transcript Q_Y final-channel readout",
            "view3": "unweighted restricted A/B NLL for H_false at the canonical no-transcript Q_H final-channel readout",
            "formula": "-(z_target - logsumexp(z_A, z_B)); selected logits cast to fp32 before logsumexp",
            "aggregation": "equal mean over every full-canonical AB/BA order row",
            "prompt_rendering": (
                "verifier-posterior-eval.py no-story/no-transcript prompts for both "
                "orders, then GPT-OSS chat-template rendering frozen once per dataset "
                "and reused across the lambda sweep"
            ),
            "lambda_fc_applied": False,
            "qh_lambda_applied": False,
        },
        "evaluation_settings": {
            "dtype": args.dtype,
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
            "devices": [f"cuda:{index}" for index in args.devices],
            "resident_worker_count": len(args.devices),
            "replicas_per_checkpoint": len(args.devices),
            "item_scheduling": (
                "whole two-order items; one seeded item per worker followed "
                "by dynamic shared-queue leasing"
            ),
            "attn_implementation": args.attn_implementation,
            "max_len": args.max_len,
            "runtime_versions": dict(getattr(args, "runtime_versions", {})),
            "item_cache_dir": display_path(resolve_project_path(args.item_cache_dir)),
        },
        "data_contracts": {
            dataset: {
                "fingerprint": contract.fingerprint,
                "provenance": contract.provenance,
                "checkpoint_training_artifact_audit": contract.training_provenance,
                "rendered_readout_fingerprint": rendered_contracts[dataset].fingerprint,
                "rendered_at": rendered_contracts[dataset].rendered_at,
                "render_local_date": rendered_contracts[dataset].local_date,
                # This representative full mapping is provenance.  Plot validation
                # compares its normalized contract and ignores only Transformers.
                "checkpoint_training_versions": dict(
                    next(
                        spec.metadata.get("versions", {})
                        for spec in specs_by_key.values()
                        if spec.dataset == dataset
                    )
                ),
            }
            for dataset, contract in data_contracts.items()
        },
        "results": ordered,
    }


def require_complete_results(
    entries: Mapping[str, Mapping[str, Any]],
    datasets: Sequence[str],
    expected_lambdas: Sequence[decimal.Decimal],
) -> List[Mapping[str, Any]]:
    selected: List[Mapping[str, Any]] = []
    missing = []
    for dataset in datasets:
        for value in expected_lambdas:
            key = f"{dataset}|{canonical_decimal(value)}"
            entry = entries.get(key)
            if entry is None:
                missing.append(key)
                continue
            for field in ("view2_qy_loss", "view3_qh_loss"):
                number = entry.get(field)
                if (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(float(number))
                ):
                    raise ContractError(f"result {key} has invalid {field}={number!r}")
            selected.append(entry)
    if missing:
        raise ContractError(
            "complete sweep is required before drawing the main figure; missing results: "
            + ", ".join(missing)
        )
    return selected


def validate_result_document_for_plot(
    document: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    expected_lambdas: Sequence[decimal.Decimal],
) -> None:
    """Reject a visually complete curve assembled from heterogeneous runs."""

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(
            f"plot cache schema_version={document.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}. Rerun evaluation before --plot-only."
        )
    if document.get("evaluator_contract_version") != EVALUATOR_CONTRACT_VERSION:
        raise ContractError(
            "plot cache uses another evaluator contract; rerun evaluation first"
        )
    if document.get("datasets") != list(datasets):
        raise ContractError(
            f"plot cache datasets={document.get('datasets')!r}, expected {list(datasets)!r}"
        )
    expected_numbers = [decimal_to_json_number(value) for value in expected_lambdas]
    if document.get("expected_qh_lambdas") != expected_numbers:
        raise ContractError(
            "plot cache Q_H lambda sweep does not match the requested sweep"
        )
    evaluator_sha256 = document.get("evaluator_sha256")
    settings = document.get("evaluation_settings")
    data_contracts = document.get("data_contracts")
    if (
        not isinstance(evaluator_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", evaluator_sha256) is None
    ):
        raise ContractError("plot cache has an invalid evaluator_sha256")
    if not isinstance(settings, Mapping) or not isinstance(data_contracts, Mapping):
        raise ContractError("plot cache is missing evaluation/data provenance")

    common_expected = {
        "evaluator_contract_version": EVALUATOR_CONTRACT_VERSION,
        "evaluator_sha256": evaluator_sha256,
        "coefficients_applied": {"lambda_fc": False, "qh_lambda": False},
        "model_loading.dtype": settings.get("dtype"),
        "model_loading.attn_implementation": settings.get("attn_implementation"),
        "model_loading.max_len": settings.get("max_len"),
        "model_loading.runtime_versions": settings.get("runtime_versions"),
        "model_loading.execution_contract_version": settings.get(
            "execution_contract_version"
        ),
    }
    for entry in selected:
        dataset = entry.get("dataset")
        dataset_contract = data_contracts.get(dataset)
        if not isinstance(dataset_contract, Mapping):
            raise ContractError(f"plot cache lacks data provenance for {dataset!r}")
        expected = {
            **common_expected,
            "data_fingerprint": dataset_contract.get("fingerprint"),
            "rendered_readout_fingerprint": dataset_contract.get(
                "rendered_readout_fingerprint"
            ),
            "render_local_date": dataset_contract.get("render_local_date"),
        }
        for dotted, expected_value in expected.items():
            try:
                actual = nested_get(entry, dotted)
            except ContractError as exc:
                raise ContractError(
                    f"plot result {dataset}|{entry.get('qh_lambda_label')} is missing "
                    f"{dotted}"
                ) from exc
            if not values_equal(actual, expected_value):
                raise ContractError(
                    "refusing to plot a mixed-provenance sweep: "
                    f"{dataset}|{entry.get('qh_lambda_label')} has "
                    f"{dotted}={actual!r}, expected {expected_value!r}"
                )
        try:
            full_training_versions = nested_get(
                entry, "model_loading.checkpoint_training_versions"
            )
        except ContractError as exc:
            raise ContractError(
                f"plot result {dataset}|{entry.get('qh_lambda_label')} is missing "
                "model_loading.checkpoint_training_versions"
            ) from exc
        actual_version_contract = checkpoint_training_version_contract(
            full_training_versions
        )
        expected_version_contract = checkpoint_training_version_contract(
            dataset_contract.get("checkpoint_training_versions")
        )
        if not values_equal(actual_version_contract, expected_version_contract):
            raise ContractError(
                "refusing to plot a mixed-provenance sweep: "
                f"{dataset}|{entry.get('qh_lambda_label')} has checkpoint training "
                f"version contract {actual_version_contract!r}, expected "
                f"{expected_version_contract!r}; Transformers is intentionally ignored"
            )
        run_hash = entry.get("run_contract_hash")
        if (
            not isinstance(run_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", run_hash) is None
        ):
            raise ContractError(
                f"plot result {dataset}|{entry.get('qh_lambda_label')} has invalid "
                "run_contract_hash"
            )


def expanded_range(values: Sequence[float]) -> Tuple[float, float]:
    low, high = min(values), max(values)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        padding = max(abs(low) * 0.1, 0.1)
    else:
        padding = (high - low) * 0.10
    return low - padding, high + padding


def render_svg(results: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> str:
    width, height = 1040, 760
    left, right, top, bottom = 115, 55, 72, 115
    plot_width = width - left - right
    plot_height = height - top - bottom
    xs = [float(result["view2_qy_loss"]) for result in results]
    ys = [float(result["view3_qh_loss"]) for result in results]
    x_min, x_max = expanded_range(xs)
    y_min, y_max = expanded_range(ys)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    population_summary = ", ".join(
        f"{dataset} {EXPECTED_CANONICAL_ITEMS[dataset]} items" for dataset in datasets
    )
    subtitle = html.escape(
        f"No transcript: {population_summary}; both A/B orders", quote=True
    )
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}",
        ".grid{stroke:#d1d5db;stroke-width:1;stroke-dasharray:4 5}",
        ".axis{stroke:#111827;stroke-width:1.5}",
        ".tick{font-size:13px;fill:#374151}",
        ".label{font-size:17px;font-weight:600}",
        ".title{font-size:22px;font-weight:700}",
        ".subtitle{font-size:13px;fill:#4b5563}",
        ".pointlabel{font-size:12px;font-weight:600}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="31" text-anchor="middle" class="title">Unweighted Full-Canonical Q_Y vs Q_H Loss</text>',
        f'<text x="{width / 2:.1f}" y="52" text-anchor="middle" class="subtitle">{subtitle}</text>',
    ]

    tick_count = 6
    for index in range(tick_count):
        fraction = index / (tick_count - 1)
        x_value = x_min + fraction * (x_max - x_min)
        x = sx(x_value)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" class="grid"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" class="tick">{x_value:.4f}</text>'
        )
        y_value = y_min + fraction * (y_max - y_min)
        y = sy(y_value)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="grid"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" class="tick">{y_value:.4f}</text>'
        )

    lines.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>',
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 43}" text-anchor="middle" class="label">Q_Y Loss (View 2, unweighted)</text>',
            f'<text x="31" y="{top + plot_height / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 31 {top + plot_height / 2:.1f})">Q_H Loss (View 3, unweighted)</text>',
        ]
    )

    by_dataset: Dict[str, List[Mapping[str, Any]]] = {
        dataset: [] for dataset in datasets
    }
    for result in results:
        by_dataset[str(result["dataset"])].append(result)
    for dataset_index, dataset in enumerate(datasets):
        color = colors[dataset_index % len(colors)]
        points = sorted(
            by_dataset[dataset],
            key=lambda item: decimal.Decimal(str(item["qh_lambda"])),
        )
        coordinate_string = " ".join(
            f"{sx(float(point['view2_qy_loss'])):.2f},{sy(float(point['view3_qh_loss'])):.2f}"
            for point in points
        )
        lines.append(
            f'<polyline points="{coordinate_string}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for point_index, point in enumerate(points):
            x = sx(float(point["view2_qy_loss"]))
            y = sy(float(point["view3_qh_loss"]))
            offset_y = -12 if (point_index + dataset_index) % 2 == 0 else 20
            label = html.escape(str(point["qh_lambda_label"]))
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}" stroke="white" stroke-width="1.5"/>'
            )
            lines.append(
                f'<text x="{x + 8:.2f}" y="{y + offset_y:.2f}" class="pointlabel" fill="{color}">&#955;={label}</text>'
            )

    legend_x = left + 16
    legend_y = top + 21
    legend_width = 175
    legend_height = 18 + 27 * len(datasets)
    lines.append(
        f'<rect x="{legend_x - 10}" y="{legend_y - 18}" width="{legend_width}" height="{legend_height}" rx="7" fill="white" fill-opacity="0.92" stroke="#d1d5db"/>'
    )
    for index, dataset in enumerate(datasets):
        y = legend_y + index * 27
        color = colors[index % len(colors)]
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 30}" y2="{y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(f'<circle cx="{legend_x + 15}" cy="{y}" r="4" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 40}" y="{y + 4}" class="tick">{html.escape(dataset)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def matplotlib_png_components() -> Tuple[Any, Any]:
    """Load Matplotlib's display-free PNG backend with a useful HPC error."""

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ContractError(
            "PNG plotting requires matplotlib. Install it in the evaluation "
            "environment (for example: python -m pip install matplotlib)."
        ) from exc
    return Figure, FigureCanvasAgg


def validate_png_runtime() -> None:
    Figure, FigureCanvasAgg = matplotlib_png_components()
    figure = Figure(figsize=(1.0, 1.0), dpi=16, facecolor="white")
    FigureCanvasAgg(figure)
    figure.text(0.5, 0.5, "PNG", ha="center", va="center", fontsize=6)
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=16, facecolor="white")
        payload = buffer.getvalue()
    except Exception as exc:
        raise ContractError(f"Matplotlib Agg PNG smoke test failed: {exc}") from exc
    finally:
        buffer.close()
        figure.clear()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n") or not payload.endswith(
        b"IEND\xaeB`\x82"
    ):
        raise ContractError(
            "Matplotlib Agg backend emitted an incomplete PNG smoke test"
        )


def render_png(results: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> bytes:
    """Render exactly one dataset with axes derived only from its own points."""

    if len(datasets) != 1:
        raise ContractError(
            f"PNG renderer requires exactly one isolated dataset, got {list(datasets)}"
        )
    dataset = str(datasets[0])
    if not results:
        raise ContractError(f"cannot render an empty {dataset} PNG")
    foreign = sorted(
        {
            str(result.get("dataset"))
            for result in results
            if result.get("dataset") != dataset
        }
    )
    if foreign:
        raise ContractError(
            f"refusing mixed-dataset PNG for {dataset}; found foreign rows {foreign}"
        )
    points = sorted(results, key=lambda item: decimal.Decimal(str(item["qh_lambda"])))
    xs = [float(point["view2_qy_loss"]) for point in points]
    ys = [float(point["view3_qh_loss"]) for point in points]
    if any(not math.isfinite(value) for value in (*xs, *ys)):
        raise ContractError(f"{dataset}: PNG inputs contain non-finite losses")
    x_min, x_max = expanded_range(xs)
    y_min, y_max = expanded_range(ys)

    Figure, FigureCanvasAgg = matplotlib_png_components()
    figure = Figure(figsize=(8.4, 6.2), dpi=180, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    colors = {"GPQA": "#2563eb", "QuALITY-H": "#dc2626"}
    color = colors.get(dataset, "#059669")
    axes.plot(
        xs,
        ys,
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label=dataset,
    )
    for index, point in enumerate(points):
        axes.annotate(
            f"λ={point['qh_lambda_label']}",
            (xs[index], ys[index]),
            xytext=(7, 9 if index % 2 == 0 else -15),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
        )
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_xlabel("Q_Y Loss (View 2, unweighted)", fontsize=11)
    axes.set_ylabel("Q_H Loss (View 3, unweighted)", fontsize=11)
    axes.set_title(
        f"No transcript: {EXPECTED_CANONICAL_ITEMS[dataset]} items; both A/B orders",
        fontsize=10,
        color="#4b5563",
        pad=12,
    )
    figure.suptitle(
        f"{dataset}: Unweighted Full-Canonical Q_Y vs Q_H Loss",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    axes.grid(True, color="#d1d5db", linewidth=0.7, linestyle="--", alpha=0.8)
    axes.legend(loc="best", frameon=True)
    axes.tick_params(labelsize=9)
    figure.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.85)

    buffer = io.BytesIO()
    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=180,
            facecolor="white",
            metadata={"Software": "task_latent.py"},
        )
        payload = buffer.getvalue()
    finally:
        buffer.close()
        figure.clear()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n") or not payload.endswith(
        b"IEND\xaeB`\x82"
    ):
        raise ContractError(f"{dataset}: Matplotlib did not emit a complete PNG")
    return payload


def dataset_figure_paths(
    figure_path: Path, datasets: Sequence[str]
) -> Mapping[str, Path]:
    """Use the exact path for one dataset, or add dataset suffixes for several."""

    if not datasets:
        raise ContractError("cannot derive figure paths for an empty dataset list")
    if len(datasets) == 1:
        return {str(datasets[0]): figure_path}
    return {
        str(dataset): figure_path.with_name(
            f"{figure_path.stem}-{dataset}{figure_path.suffix}"
        )
        for dataset in datasets
    }


def results_for_dataset_in_lambda_order(
    selected: Sequence[Mapping[str, Any]],
    dataset: str,
    expected_lambdas: Sequence[decimal.Decimal],
) -> List[Mapping[str, Any]]:
    by_lambda: Dict[str, Mapping[str, Any]] = {}
    for result in selected:
        if result.get("dataset") != dataset:
            continue
        label = str(result.get("qh_lambda_label"))
        if label in by_lambda:
            raise ContractError(f"{dataset}: duplicate plot/summary lambda {label}")
        by_lambda[label] = result
    ordered: List[Mapping[str, Any]] = []
    for value in expected_lambdas:
        label = canonical_decimal(value)
        result = by_lambda.get(label)
        if result is None:
            raise ContractError(f"{dataset}: missing plot/summary lambda {label}")
        ordered.append(result)
    if len(by_lambda) != len(ordered):
        raise ContractError(
            f"{dataset}: unexpected plot/summary lambdas {sorted(by_lambda)}"
        )
    return ordered


def format_loss_array(results: Sequence[Mapping[str, Any]], field: str) -> str:
    return "[" + ",".join(repr(float(result[field])) for result in results) + "]"


def print_loss_summaries(
    selected: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    expected_lambdas: Sequence[decimal.Decimal],
) -> None:
    for dataset_index, dataset in enumerate(datasets):
        if dataset_index:
            print()
        ordered = results_for_dataset_in_lambda_order(
            selected, dataset, expected_lambdas
        )
        print(f"Dataset: {dataset}")
        print(f"Q_Y loss (x) = {format_loss_array(ordered, 'view2_qy_loss')}")
        print(f"Q_H loss (y) = {format_loss_array(ordered, 'view3_qh_loss')}")


def plot_complete_results(
    entries: Mapping[str, Mapping[str, Any]],
    datasets: Sequence[str],
    expected_lambdas: Sequence[decimal.Decimal],
    figure_path: Path,
    document: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    selected = require_complete_results(entries, datasets, expected_lambdas)
    validate_result_document_for_plot(document, selected, datasets, expected_lambdas)
    figure_paths = dataset_figure_paths(figure_path, datasets)
    for dataset in datasets:
        dataset_results = results_for_dataset_in_lambda_order(
            selected, dataset, expected_lambdas
        )
        output_path = figure_paths[str(dataset)]
        atomic_write_bytes(output_path, render_png(dataset_results, (dataset,)))
        print(
            f"[plot] wrote independent {dataset} {len(dataset_results)}-point curve "
            f"to {output_path}"
        )
    return selected


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value!r}")
    return parsed


def parse_cuda_devices(raw: str) -> Tuple[int, ...]:
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--devices must contain at least one cuda:N")
    indices: List[int] = []
    for part in parts:
        match = re.fullmatch(r"cuda:(\d+)", part)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"invalid CUDA device {part!r}; use comma-separated logical names "
                "such as cuda:0,cuda:1,cuda:2,cuda:3"
            )
        indices.append(int(match.group(1)))
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError(f"--devices contains duplicates: {raw!r}")
    return tuple(indices)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="dataset curves to evaluate (default: GPQA QuALITY-H)",
    )
    parser.add_argument(
        "--expected-qh-lambdas",
        default=DEFAULT_EXPECTED_LAMBDAS,
        help="comma-separated required sweep, or 'auto' to require identical discovered sets",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE,
        help=(
            "figure path for one dataset, or basename for multiple datasets "
            "(dataset names are inserted before the suffix)"
        ),
    )
    parser.add_argument(
        "--item-cache-dir",
        type=Path,
        default=DEFAULT_ITEM_CACHE_DIR,
        help="atomic per-item resume cache directory",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--devices",
        type=parse_cuda_devices,
        default=DEFAULT_DEVICES,
        help=(
            "comma-separated logical CUDA devices, one full resident model per "
            f"worker (default: {DEFAULT_DEVICES})"
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Transformers attention implementation (training used eager)",
    )
    parser.add_argument("--max-len", type=positive_int, default=8192)
    parser.add_argument("--progress-every", type=positive_int, default=20)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "recompute every aggregate/item score and atomically replace its "
            "compatible cache entry"
        ),
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help=(
            "audit checkpoint-training artifacts and build the full canonical "
            "no-transcript population without loading checkpoints or tokenizers"
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="render PNGs from an already complete result cache without inspecting checkpoints",
    )
    args = parser.parse_args(argv)
    if args.validate_data_only and args.plot_only:
        parser.error("--validate-data-only and --plot-only are mutually exclusive")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets contains duplicates")
    invalid = [dataset for dataset in args.datasets if dataset not in DEFAULT_DATASETS]
    if invalid:
        parser.error(
            f"this experiment is locked to {DEFAULT_DATASETS}; invalid dataset(s): {invalid}"
        )
    if args.figure.suffix.lower() != ".png":
        parser.error(f"--figure must end in .png, got {args.figure}")
    if (
        not args.validate_data_only
        and tuple(args.datasets) != DEFAULT_DATASETS
        and (args.results == DEFAULT_RESULTS or args.figure == DEFAULT_FIGURE)
    ):
        parser.error(
            "a dataset subset requires explicit --results and --figure paths; "
            "refusing to overwrite the shared two-dataset outputs"
        )
    if isinstance(args.devices, str):
        args.devices = parse_cuda_devices(args.devices)
    # Materialize/validate stable worker identity before any model path is reached.
    try:
        worker_specs_from_devices(args.devices)
    except ContractError as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    checkpoint_root = resolve_project_path(args.checkpoint_root)
    results_path = resolve_project_path(args.results)
    figure_path = resolve_project_path(args.figure)
    item_cache_path = resolve_project_path(args.item_cache_dir)
    expected_requested = parse_expected_lambdas(args.expected_qh_lambdas)

    if not args.validate_data_only:
        validate_png_runtime()

    if args.plot_only:
        if expected_requested is None:
            raise ContractError("--plot-only requires explicit --expected-qh-lambdas")
        cache = load_result_cache(results_path)
        entries = cache_entries(cache)
        selected = plot_complete_results(
            entries, args.datasets, expected_requested, figure_path, cache
        )
        print_loss_summaries(selected, args.datasets, expected_requested)
        return

    ft = load_ft_module()
    posterior = load_posterior_eval_module()
    data_contracts = {
        dataset: prepare_dataset_contract(ft, posterior, dataset)
        for dataset in args.datasets
    }
    if args.validate_data_only:
        print("\n[data] validation complete; no checkpoint/model was loaded.")
        return

    if args.max_len != int(ft.MAX_SEQ_LEN):
        raise ContractError(
            f"--max-len {args.max_len} differs from the checkpoint/training contract "
            f"{ft.MAX_SEQ_LEN}; this evaluator does not compare mixed truncation policies"
        )
    args.runtime_versions = validate_evaluation_runtime()

    discovered = discover_checkpoint_paths(checkpoint_root, args.datasets)
    expected_lambdas = validate_discovered_sweep(
        discovered, args.datasets, expected_requested
    )
    expected_set = set(expected_lambdas)
    discovered = [entry for entry in discovered if entry[1] in expected_set]
    print(
        f"\n[checkpoint] discovered complete sweep: {len(discovered)} directories "
        f"({', '.join(canonical_decimal(value) for value in expected_lambdas)})"
    )

    # Preflight the WHOLE sweep before loading the first 20B model.  This prevents a
    # many-hour run ending with a partial, non-plottable curve because one late shard
    # or metadata file was absent.
    specs = [
        preflight_checkpoint(dataset, qh_lambda, label, path)
        for dataset, qh_lambda, label, path in discovered
    ]
    specs.sort(key=lambda item: (args.datasets.index(item.dataset), item.qh_lambda))
    for spec in specs:
        validate_checkpoint_metadata(ft, spec, data_contracts[spec.dataset])
        print(
            f"[checkpoint] OK {spec.path.name}: {len(spec.weight_files)} weight file(s), "
            f"fingerprint={spec.fingerprint[:16]}..."
        )
    lock_sweep_contracts(specs)
    preflight_all_tokenizers(ft, specs)

    # GPT-OSS's chat template inserts the current local date.  Freeze every prompt
    # before loading the first model, then reuse those exact token IDs across the full
    # lambda sweep.  A run that crosses midnight during this short rendering phase is
    # rejected instead of silently mixing date-conditioned inputs.
    render_local_date = _datetime.date.today().isoformat()
    rendered_contracts: Dict[str, RenderedDatasetContract] = {}
    for dataset in args.datasets:
        representative = next(spec for spec in specs if spec.dataset == dataset)
        rendered_contracts[dataset] = render_dataset_contract(
            ft,
            representative,
            data_contracts[dataset],
            args,
            render_local_date,
        )
    if _datetime.date.today().isoformat() != render_local_date:
        raise ContractError(
            "the local date changed while rendering chat-template prompts; rerun so "
            "every checkpoint receives the same date-conditioned input"
        )

    evaluator_sha256 = file_sha256(Path(__file__).resolve())
    cache = load_result_cache(results_path)
    entries = cache_entries(cache)
    item_cache = ItemResultCache(item_cache_path)
    specs_by_key = {spec.key: spec for spec in specs}
    expected_run_hashes = {
        spec.key: run_contract_hash(
            spec,
            data_contracts[spec.dataset],
            rendered_contracts[spec.dataset],
            args,
        )
        for spec in specs
    }
    entries = (
        {}
        if args.force
        else filter_current_result_entries(
            entries,
            specs_by_key,
            data_contracts,
            rendered_contracts,
            args,
            evaluator_sha256,
            expected_run_hashes,
        )
    )

    for spec in specs:
        data = data_contracts[spec.dataset]
        rendered = rendered_contracts[spec.dataset]
        contract_hash = expected_run_hashes[spec.key]
        cached = entries.get(spec.key)
        if (
            not args.force
            and cached is not None
            and cached.get("run_contract_hash") == contract_hash
        ):
            print(
                f"[cache] reuse {spec.dataset} lambda={spec.qh_lambda_label}: "
                f"View2={cached['view2_qy_loss']:.6f}, "
                f"View3={cached['view3_qh_loss']:.6f}"
            )
            continue

        result = dict(
            evaluate_checkpoint(
                ft,
                spec,
                data,
                rendered,
                rendered.letter_ids,
                rendered.prefill_ids,
                contract_hash,
                item_cache,
                args,
            )
        )
        result["evaluator_contract_version"] = EVALUATOR_CONTRACT_VERSION
        result["evaluator_sha256"] = evaluator_sha256
        result["run_contract_hash"] = contract_hash
        entries[spec.key] = result
        document = build_result_document(
            args,
            args.datasets,
            expected_lambdas,
            evaluator_sha256,
            data_contracts,
            rendered_contracts,
            entries,
            specs_by_key,
            expected_run_hashes,
        )
        atomic_write_json(results_path, document)
        print(
            f"[result] {spec.dataset} lambda={spec.qh_lambda_label}: "
            f"View2={result['view2_qy_loss']:.8f}, "
            f"View3={result['view3_qh_loss']:.8f}; cache saved to {results_path}"
        )

    # Rewrite even if every point came from cache, so top-level provenance/settings
    # are canonical for this invocation.
    document = build_result_document(
        args,
        args.datasets,
        expected_lambdas,
        evaluator_sha256,
        data_contracts,
        rendered_contracts,
        entries,
        specs_by_key,
        expected_run_hashes,
    )
    atomic_write_json(results_path, document)
    selected = plot_complete_results(
        entries, args.datasets, expected_lambdas, figure_path, document
    )
    print(
        "\nDone. Reported losses are full-canonical no-transcript dual-order "
        "per-row means: no lambda_fc or qh_lambda coefficient was applied."
    )
    print()
    print_loss_summaries(selected, args.datasets, expected_lambdas)


if __name__ == "__main__":
    try:
        main()
    except (ContractError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
