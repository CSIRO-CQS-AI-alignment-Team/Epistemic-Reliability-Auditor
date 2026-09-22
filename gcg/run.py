"""Command-line runner for white-box GCG.

``--preflight-only`` performs every tokenizer-side check (scaffold shape, marker
survival, editable vocabulary, initial-suffix deployability) WITHOUT loading model
weights, so a spec can be validated on a login node before a GPU job is queued.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Mapping, Sequence

import torch

from .apply import require_deployment_metadata
from .objective import (
    SuffixValidator,
    allowed_token_ids,
    compile_task_spec,
    infer_logits_kwarg,
    load_task_spec,
    sha256_file,
    task_spec_digest,
    tokenizer_vocab_size,
)
from .optimizer import GCGConfig, GCGOptimizer, save_result

DEFAULT_INIT_SEED_TEXT = " x"


def _dtype(value: str) -> torch.dtype | None:
    choices = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return choices[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown dtype {value!r}") from exc


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested {device}, but CUDA is unavailable")
    return device


def _load_tokenizer(model_name: str) -> Any:
    """Load the tokenizer through the production loader contract when available."""

    try:
        from adversarial_transcript.score_verifier import load_text_tokenizer
        from transformers import AutoTokenizer

        tokenizer = load_text_tokenizer(AutoTokenizer, model_name)
    except ImportError:  # pragma: no cover - exercised only without transformers
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise RuntimeError("tokenizer needs a pad_token_id or eos_token_id for GCG batching")
    # Left padding is also set on the tokenizer for models that consult this field in
    # their position-id construction. objective.py pads explicitly as a second guard.
    tokenizer.padding_side = "left"
    return tokenizer


def _from_pretrained_compat(auto_cls: Any, model_name: str, dtype: torch.dtype | None,
                            **kwargs: Any) -> Any:
    """transformers renamed ``torch_dtype=`` to ``dtype=``; support both."""

    if dtype is None:
        return auto_cls.from_pretrained(model_name, **kwargs)
    try:
        return auto_cls.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return auto_cls.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def _quantization_kwargs(model_name: str) -> dict[str, Any]:
    """Match score_verifier: MXFP4 checkpoints must be dequantized to be differentiable."""

    try:
        from transformers import AutoConfig, Mxfp4Config
    except ImportError:  # pragma: no cover - Transformers build without MXFP4 support
        return {}
    try:
        config = AutoConfig.from_pretrained(model_name)
    except (OSError, ValueError):  # pragma: no cover - offline/local-only checkpoints
        return {}
    if getattr(config, "quantization_config", None) is None:
        return {}
    return {"quantization_config": Mxfp4Config(dequantize=True)}


def _load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype | None,
    device_map: str | Mapping[str, str],
    *,
    max_memory: Mapping[int | str, int | str] | None = None,
) -> Any:
    from transformers import AutoModelForCausalLM

    # Stream a single-GPU replica directly to its assigned device to avoid retaining
    # another complete checkpoint copy in host memory.
    kwargs: dict[str, Any] = {
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    kwargs.update(_quantization_kwargs(model_name))
    if isinstance(device_map, Mapping):
        # Accept a concrete device map directly. Model-parallel workers use a balanced
        # map with per-group memory limits; single-GPU callers use the selected device.
        kwargs["device_map"] = dict(device_map)
    elif device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
        kwargs["device_map"] = device_map
    else:
        kwargs["device_map"] = {"": str(device)}
    if max_memory is not None:
        kwargs["max_memory"] = dict(max_memory)
    try:
        model = _from_pretrained_compat(AutoModelForCausalLM, model_name, dtype, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = _from_pretrained_compat(AutoModelForCausalLM, model_name, dtype, **kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_init_suffix(tokenizer: Any, text: str | None, suffix_len: int) -> torch.Tensor:
    if suffix_len < 1:
        raise ValueError("suffix length must be positive")

    def _ids(raw: str) -> torch.Tensor:
        encoded = tokenizer(raw, add_special_tokens=False, return_tensors="pt")
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if ids.ndim == 2:
            ids = ids[0]
        return ids.to(dtype=torch.long, device="cpu")

    if text is not None:
        ids = _ids(text)
        if ids.numel() != suffix_len:
            raise ValueError(
                f"--init-text encodes to {ids.numel()} tokens but --suffix-len is {suffix_len}; "
                "set the exact token length or change --suffix-len"
            )
        return ids

    # A printable, whitespace-led seed keeps the deployed fragment separated from the
    # preceding word and survives the decode/re-encode round trip on BPE tokenizers.
    seed_ids = _ids(DEFAULT_INIT_SEED_TEXT)
    if seed_ids.numel() == 0:
        raise ValueError(f"default GCG seed {DEFAULT_INIT_SEED_TEXT!r} tokenized to no ids")
    if seed_ids.numel() != 1:
        # Keep the default deterministic even for tokenizers that split the seed.
        seed_ids = seed_ids[:1]
    return seed_ids.repeat(suffix_len)


def resolve_editable_vocabulary(
    tokenizer: Any,
    *,
    allow_special_tokens: bool = False,
    allow_non_ascii: bool = False,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Return ``(allowed_ids, tokenizer_vocab_limit)`` for one tokenizer.

    The editable vocabulary depends only on the tokenizer and the two relaxation
    flags, never on the task spec, but building it decodes and re-encodes the whole
    vocabulary. A batch runner that preflights hundreds of specs against the SAME
    tokenizer can therefore compute this once and hand the tensor to every
    :func:`preflight` call instead of repeating the scan (see ``gcg.run_all``).
    """

    limit = vocab_size if vocab_size is not None else tokenizer_vocab_size(tokenizer)
    if limit is None:
        raise ValueError(
            "tokenizer does not report a vocabulary size; pass an explicit vocab_size"
        )
    allowed = allowed_token_ids(
        tokenizer, int(limit),
        allow_special_tokens=allow_special_tokens,
        allow_non_ascii=allow_non_ascii,
    )
    return allowed, int(limit)


def preflight(
    tokenizer: Any,
    spec: Mapping[str, Any],
    *,
    suffix_len: int,
    init_text: str | None = None,
    allow_special_tokens: bool = False,
    allow_non_ascii: bool = False,
    vocab_size: int | None = None,
    allowed_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Tokenizer-only validation shared by ``--preflight-only`` and the real run.

    Raises on any condition that would make the run unsound or its output
    un-deployable: a mis-shaped answer slot, an edit marker that does not survive
    tokenization, missing per-candidate deployment metadata, an empty editable
    vocabulary, or an initial suffix that cannot be deployed as text.

    ``allowed_ids`` is an optional pre-built editable vocabulary from
    :func:`resolve_editable_vocabulary`. It is a pure cache of a tokenizer-only
    computation: passing it skips the vocabulary scan but changes nothing else, and
    the seed suffix is still checked against it here exactly as if it had just been
    built. Pass it only for a tokenizer/flag combination it was built from.
    """

    # Same guard gcg.apply enforces: refuse a spec whose result could never be
    # written back, before spending the search.
    require_deployment_metadata(spec)
    tasks, choice_ids, prefill = compile_task_spec(spec, tokenizer)
    meta = spec.get("meta", {}) if isinstance(spec.get("meta"), Mapping) else {}
    validator = SuffixValidator(
        tokenizer, tasks, edit_marker=str(meta.get("edit_marker", "")) or None
    )
    if allowed_ids is None:
        allowed, limit = resolve_editable_vocabulary(
            tokenizer,
            allow_special_tokens=allow_special_tokens,
            allow_non_ascii=allow_non_ascii,
            vocab_size=vocab_size,
        )
    else:
        limit = vocab_size if vocab_size is not None else tokenizer_vocab_size(tokenizer)
        if limit is None:
            raise ValueError(
                "tokenizer does not report a vocabulary size; pass an explicit vocab_size"
            )
        allowed = allowed_ids.detach().to(dtype=torch.long, device="cpu")
        if allowed.ndim != 1 or allowed.numel() == 0:
            raise ValueError("allowed_ids must be a non-empty one-dimensional tensor")
    suffix_ids = encode_init_suffix(tokenizer, init_text, suffix_len)
    outside = suffix_ids[~torch.isin(suffix_ids, allowed)]
    if outside.numel():
        raise ValueError(
            f"initial suffix uses token id(s) {sorted(set(int(x) for x in outside.tolist()))} "
            "that are excluded from the editable vocabulary; pick another --init-text"
        )
    calibration = validator.calibrate(suffix_ids)
    reason = validator.text_reason(suffix_ids)
    if reason is not None:
        raise ValueError(f"initial suffix is not a deployable text fragment: {reason}")
    validator.verify_exact(suffix_ids)
    return {
        "tasks": tasks,
        "choice_ids": choice_ids,
        "prefill": prefill,
        "validator": validator,
        "allowed_ids": allowed,
        "suffix_ids": suffix_ids,
        "report": {
            "task_count": len(tasks),
            "choice_token_ids": dict(choice_ids),
            "editable_vocab_size": int(allowed.numel()),
            "tokenizer_vocab_size": int(limit),
            "initial_suffix_ids": [int(x) for x in suffix_ids.tolist()],
            "initial_suffix_text": validator.decode(suffix_ids),
            **calibration,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Greedy Coordinate Gradient against a frozen verifier. "
            "Tasks must be produced by python -m gcg.prepare."
        )
    )
    parser.add_argument("--model", required=True, help="HF model/checkpoint used as the frozen verifier")
    parser.add_argument("--tasks", required=True, help="gcg-task-v2 JSON")
    parser.add_argument("--out", help="gcg-result-v2 JSON (not required with --preflight-only)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=256,
                        help="exactly evaluated substitutions per step")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="max FORWARD ROWS per model call (candidates x tasks are "
                             "flattened first, so this is a hard batch bound)")
    parser.add_argument("--suffix-len", type=int, default=16)
    parser.add_argument("--init-text", default=None,
                        help="optional exact-length initial suffix text")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-special-tokens", action="store_true",
                        help="allow GCG to propose chat/control/padding tokens")
    parser.add_argument("--allow-non-ascii", action="store_true",
                        help="allow non-ASCII tokens in the public suffix")
    parser.add_argument("--no-gradient-normalization", action="store_true",
                        help="disable per-task unit-norm gradient aggregation")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run tokenizer-side checks and exit without loading the model")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.preflight_only and not args.out:
        raise SystemExit("--out is required unless --preflight-only is given")
    if args.out and os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing result: {args.out}")
    for name in ("steps", "top_k", "candidates", "eval_batch_size", "suffix_len"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    spec = load_task_spec(args.tasks)
    tokenizer = _load_tokenizer(args.model)
    # Every tokenizer-side check runs BEFORE the weights are loaded, so a mis-shaped
    # spec, a destroyed edit marker or an undeployable seed costs seconds, not a 20B
    # model load. GCGOptimizer re-checks the ids against the real embedding matrix.
    checked = preflight(
        tokenizer, spec,
        suffix_len=args.suffix_len,
        init_text=args.init_text,
        allow_special_tokens=args.allow_special_tokens,
        allow_non_ascii=args.allow_non_ascii,
    )
    report = checked["report"]
    if args.preflight_only:
        print(f"[gcg.run] preflight OK for {args.tasks}")
        for key in sorted(report):
            print(f"[gcg.run]   {key}: {report[key]!r}")
        return 0

    device = _resolve_device(args.device)
    dtype = _dtype(args.dtype)
    if dtype is None and device.type == "cuda":
        dtype = torch.bfloat16
    model = _load_model(args.model, device, dtype, args.device_map)
    tasks = checked["tasks"]
    logits_kwarg = infer_logits_kwarg(model)
    config = GCGConfig(
        steps=args.steps,
        top_k=args.top_k,
        candidates=args.candidates,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        normalize_per_task=not args.no_gradient_normalization,
    )
    optimizer = GCGOptimizer(
        model,
        tokenizer,
        tasks,
        checked["choice_ids"],
        pad_token_id=int(tokenizer.pad_token_id),
        config=config,
        logits_kwarg=logits_kwarg,
        allowed_ids=checked["allowed_ids"],
        validator=checked["validator"],
    )
    result = optimizer.optimize(checked["suffix_ids"])
    metadata = {
        "model": args.model,
        "device": str(device),
        "device_map": args.device_map,
        "dtype": str(dtype or "model-default"),
        "task_spec": os.path.abspath(args.tasks),
        # Content digest: what gcg.apply requires, stable across JSON formatting.
        "task_spec_digest": task_spec_digest(spec),
        # Byte digest of the file as read, kept purely as provenance.
        "task_spec_sha256": sha256_file(args.tasks),
        "task_count": len(tasks),
        "prefill": checked["prefill"],
        "choice_token_ids": checked["choice_ids"],
        "logits_kwarg": logits_kwarg,
        "suffix_separator": str(spec.get("meta", {}).get("suffix_separator", "")),
        "preflight": report,
        "editable_suffix_is_public": True,
        "warning": (
            "Validate the decoded suffix as a public transcript fragment and rerun "
            "the production quote/correctness/leakage filters before using it."
        ),
    }
    save_result(args.out, result, metadata=metadata)
    print(
        f"[gcg.run] {len(tasks)} task(s), {args.steps} step(s); "
        f"loss {result.initial.loss:.6f} -> {result.final.loss:.6f}; wrote {args.out}"
    )
    print(f"[gcg.run] suffix: {result.suffix_text!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
