"""Apply an optimized public suffix to native candidate JSONL records.

This is deliberately a small, auditable adapter that loads no model. It does not score or
select anything; the caller must rerun the existing correctness/leakage/quote filters
and the verifier scorer after this step.

It is, however, fail-closed about *fidelity*: the fragment written here must be the
exact text GCG optimized, appended at the exact place the task specification carved
out, without disturbing the transcript rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any, Mapping, Sequence

from adversarial_transcript import common, schema
from quote_utils import normalize_text, verify_quotes

from .objective import (
    FORBIDDEN_SUFFIX_SUBSTRINGS,
    RESULT_SCHEMA_VERSION,
    load_task_spec,
    sha256_text,
    task_spec_digest,
)


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(row["item_id"]), str(row["condition"]), int(row["candidate_idx"]))


def _require_sha256(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ValueError(
            f"{what} must be a 64-character lowercase sha256 hex digest, got {value!r}; "
            "refusing to deploy without evidence of the span it was optimized against"
        )
    return value


def _require_round_index(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{what} must be a JSON integer, got {value!r}; refusing to guess which "
            "round holds the editable span"
        )
    if value < 0:
        raise ValueError(f"{what} must be non-negative, got {value!r}")
    return int(value)


def _render_is_append_stable(argument: str, addition: str) -> bool:
    """True when appending ``addition`` leaves the verified rendering untouched.

    ``verify_quotes`` only rewrites ``<quote>...</quote>`` spans, and which spans it
    matches depends on the tag structure alone -- not on the story. So checking
    append-stability against an empty probe story is sufficient for every story.
    """

    probe = normalize_text("")
    return verify_quotes(argument + addition, probe) == verify_quotes(argument, probe) + addition


def require_deployment_metadata(
    task_spec: Mapping[str, Any],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Validate and group the per-candidate metadata deployment depends on.

    ``gcg.prepare`` always writes every field below, so a missing one means the spec
    was hand-edited or truncated -- never a licence to guess the edit site. Also
    called from ``gcg.run``'s preflight so an un-deployable spec fails before the
    model is loaded rather than after hours of search.
    """

    meta = task_spec.get("meta", {})
    meta = meta if isinstance(meta, Mapping) else {}
    if not isinstance(meta.get("suffix_separator"), str):
        raise ValueError(
            "task specification meta has no suffix_separator string; refusing to guess "
            "how the fragment joins the argument"
        )
    groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    for task in task_spec.get("tasks", []):
        metadata = task.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"task {task.get('id')!r} has a non-object metadata block")
        key = tuple(metadata.get("candidate_key", ()))
        if len(key) != 3:
            raise ValueError(f"task {task.get('id')!r} lacks a three-part candidate_key")
        key = (str(key[0]), str(key[1]), int(key[2]))
        debater = metadata.get("compromised_debater")
        if debater not in common.DEBATER_NAMES:
            raise ValueError(
                f"task {task.get('id')!r} has compromised_debater={debater!r}; expected "
                f"one of {common.DEBATER_NAMES}"
            )
        entry = {
            "debater": debater,
            "round_index": _require_round_index(
                metadata.get("final_round_index"),
                f"task {task.get('id')!r} metadata.final_round_index",
            ),
            "native_sha256": _require_sha256(
                metadata.get("native_final_argument_sha256"),
                f"task {task.get('id')!r} metadata.native_final_argument_sha256",
            ),
        }
        previous = groups.get(key)
        if previous is not None and previous != entry:
            raise ValueError(f"candidate {key} has inconsistent editable-span metadata")
        groups[key] = entry
    if not groups:
        raise ValueError("task specification contains no candidate metadata")
    return groups


def _suffix_from_result(result: Mapping[str, Any], task_spec: Mapping[str, Any]) -> str:
    """Extract the suffix only if the result proves it is safe to deploy.

    Every piece of evidence is REQUIRED, not merely checked when present: a result
    that simply omits its digest or fidelity block must not be easier to deploy than
    one that records a failure.
    """

    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"result is not a {RESULT_SCHEMA_VERSION} file "
            f"(got {result.get('schema_version')!r})"
        )
    suffix = result.get("suffix_text")
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("GCG result has no non-empty suffix_text")
    recorded = result.get("suffix_text_sha256")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("GCG result has no suffix_text_sha256; refusing to deploy it")
    if recorded != sha256_text(suffix):
        raise ValueError("GCG result suffix_text does not match its recorded sha256")
    fidelity = result.get("fidelity")
    if not isinstance(fidelity, Mapping):
        raise ValueError("GCG result has no fidelity block; refusing to deploy it")
    if fidelity.get("roundtrip_verified") is not True:
        raise ValueError(
            "GCG result is not marked round-trip verified; refusing to deploy a suffix "
            "whose text may denote a different token sequence"
        )
    meta = result.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("GCG result has no meta block; refusing to deploy it")
    digest = meta.get("task_spec_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError(
            "GCG result records no task_spec_digest, so it cannot be bound to the task "
            "specification it was optimized against"
        )
    expected = task_spec_digest(task_spec)
    if digest != expected:
        raise ValueError(
            "GCG result was optimized against a different task specification "
            f"(result digest {digest[:16]}..., supplied spec {expected[:16]}...)"
        )
    for bad in FORBIDDEN_SUFFIX_SUBSTRINGS:
        if bad in suffix:
            raise ValueError(
                f"GCG suffix contains {bad!r}, which would corrupt the transcript "
                "rendering or forge a quote-verification tag"
            )
    return suffix


def apply_suffix(
    candidates: Sequence[Mapping[str, Any]],
    task_spec: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    separator: str | None = None,
    allow_reapply: bool = False,
) -> list[dict[str, Any]]:
    # Spec first: it establishes every field the rest of this function reads.
    groups = require_deployment_metadata(task_spec)
    suffix = _suffix_from_result(result, task_spec)
    meta_separator = task_spec["meta"]["suffix_separator"]
    if separator is None:
        separator = meta_separator
    elif separator != meta_separator:
        raise ValueError(
            f"separator {separator!r} does not match the task spec's "
            f"suffix_separator {meta_separator!r}"
        )
    addition = separator + suffix
    if not addition[:1].isspace():
        raise ValueError(
            "the applied fragment does not start with whitespace and would glue onto "
            "the last word of the debater's argument"
        )

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for original in candidates:
        row = dict(original)
        key = _candidate_key(row)
        if key in groups:
            entry = groups[key]
            transcript = row.get("transcript")
            if not isinstance(transcript, Mapping) or not isinstance(transcript.get("rounds"), list):
                raise ValueError(f"candidate {key} has no native transcript rounds")
            rounds = [dict(round_row) for round_row in transcript["rounds"]]
            debater = entry["debater"]
            index = entry["round_index"]
            if not rounds:
                raise ValueError(f"candidate {key} has no transcript rounds")
            if index >= len(rounds):
                raise ValueError(
                    f"candidate {key}: task spec edits round {index} but the candidate "
                    f"has {len(rounds)} round(s)"
                )
            if index != len(rounds) - 1:
                # The editable span must be the candidate's final compromised argument.
                raise ValueError(
                    f"candidate {key}: task spec edits round {index}, which is not the "
                    f"final round ({len(rounds) - 1}) of this candidate"
                )
            # Semantics can drift while the argument text stays byte-identical: the
            # stance map or the declared compromised debater can be swapped underneath
            # an unchanged argument. Both must still name the side the spec edited.
            declared = row.get("compromised_debater")
            if declared != debater:
                raise ValueError(
                    f"candidate {key}: compromised_debater={declared!r} but the task "
                    f"spec's editable span is inside {debater!r}'s argument"
                )
            stance = transcript.get(debater)
            if stance != "Y_true":
                raise ValueError(
                    f"candidate {key}: {debater!r} now defends {stance!r}, not 'Y_true'; "
                    "refusing to write a correctness-preserving suffix into a Y_false "
                    "argument"
                )
            if debater not in rounds[index]:
                raise ValueError(f"candidate {key} lacks the final compromised argument")
            argument = str(rounds[index][debater])
            gen_meta = row.get("gen_meta")
            already = (
                isinstance(gen_meta, Mapping) and bool(gen_meta.get("gcg_suffix_applied"))
            ) or argument.endswith(addition)
            if already and not allow_reapply:
                raise ValueError(
                    f"candidate {key} already carries a GCG suffix; rerun with "
                    "allow_reapply=True only if you really want to write it again"
                )
            # Check source-text identity after the idempotence guard so a repeated call
            # reports the most specific cause. ``allow_reapply`` applies only when the
            # argument still matches the prepared source text.
            expected_sha = entry["native_sha256"]  # validated well-formed at group time
            if hashlib.sha256(argument.encode("utf-8")).hexdigest() != expected_sha:
                raise ValueError(
                    f"candidate {key}: the final argument changed since task preparation; "
                    "the optimized suffix is invalid for this candidate"
                )
            if not _render_is_append_stable(argument, addition):
                raise ValueError(
                    f"candidate {key}: appending the suffix changes the verified quote "
                    "rendering of the original argument"
                )
            rounds[index][debater] = argument + addition
            new_transcript = dict(transcript)
            new_transcript["rounds"] = rounds
            row["transcript"] = new_transcript
            row["gen_meta"] = {
                **(dict(gen_meta) if isinstance(gen_meta, Mapping) else {}),
                "gcg_suffix_applied": True,
                "gcg_suffix_text": suffix,
                "gcg_suffix_separator": separator,
                "gcg_suffix_sha256": sha256_text(suffix),
            }
            seen.add(key)
        output.append(row)
    missing = sorted(set(groups) - seen)
    if missing:
        raise ValueError(f"task specification references missing candidate(s): {missing[:3]}")
    for row in output:
        errors = schema.validate_candidate(row)
        if errors:
            raise ValueError(f"applied candidate {_candidate_key(row)} failed schema: {errors[0]}")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a GCG suffix to native candidate JSONL")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-reapply", action="store_true",
                        help="permit rewriting a candidate that carries the "
                             "gcg_suffix_applied flag; its argument must still match "
                             "the task spec's recorded hash")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing candidate file: {args.out}")
    candidates = common.read_jsonl(args.candidates)
    task_spec = load_task_spec(args.tasks)
    with open(args.result, encoding="utf-8") as handle:
        result = json.load(handle)
    rows = apply_suffix(candidates, task_spec, result, allow_reapply=args.allow_reapply)
    common.write_jsonl(args.out, rows)
    applied = sum(
        1 for row in rows
        if isinstance(row.get("gen_meta"), Mapping)
        and row["gen_meta"].get("gcg_suffix_applied")
    )
    print(f"[gcg.apply] applied suffix to {applied} candidate(s); wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
