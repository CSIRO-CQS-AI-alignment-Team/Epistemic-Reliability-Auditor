"""Build GCG task specifications from production adversarial candidates.

The preparation step reuses the exact prompt builders, deterministic option order and
final-channel scaffold from ``adversarial_transcript.score_verifier``. The resulting
two tasks per candidate share one editable span:

* q_m targets the semantic ``Y_true`` option;
* q_h targets the semantic ``H_false`` option.

The editable span is placed inside the final argument of the compromised Y_true
debater, immediately before its closing public-transcript quote.  q_h text stays in
the private task specification and is never part of the emitted suffix itself.

No separator is injected on the prefix side: a trailing space in ``prompt_prefix``
would be swallowed into the edit marker's first pre-token by GPT-4-family
tokenizers, which destroys the marker match. The optimized suffix carries its own
leading whitespace instead (``SuffixValidator`` enforces it), so the deployed text
is byte-identical to what was optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from typing import Any, Mapping, Sequence

from adversarial_transcript import common, schema, score_verifier
from quote_utils import normalize_text, verify_quotes

from .objective import DEFAULT_EDIT_MARKER, TASK_SCHEMA_VERSION

# The optimized suffix is appended directly to the compromised debater's final
# argument; it must supply its own leading whitespace.
SUFFIX_SEPARATOR = ""
DEFAULT_CONDITIONS = ("adversarial",)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _parse_conditions(raw: str) -> list[str]:
    conditions = [c.strip().lower() for c in raw.split(",") if c.strip()]
    bad = [c for c in conditions if c not in schema.CONDITIONS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown condition(s): {', '.join(bad)}")
    if not conditions:
        raise argparse.ArgumentTypeError("at least one condition is required")
    return conditions


def _target_letter(order: Mapping[str, str], semantic_label: str) -> str:
    matches = [letter for letter, label in order.items() if label == semantic_label]
    if len(matches) != 1:
        raise ValueError(
            f"answer order does not contain exactly one {semantic_label}: {order!r}"
        )
    return matches[0]


def resolve_compromised_debater(candidate: Mapping[str, Any]) -> str:
    """Resolve the debater whose final argument carries the editable span.

    Fails closed: GCG must never guess which side it is editing, because the whole
    construction depends on the span living inside the Y_true defender's argument.
    """

    transcript = candidate.get("transcript")
    if not isinstance(transcript, Mapping):
        raise ValueError(f"{candidate.get('item_id')}: candidate has no transcript object")
    declared = candidate.get("compromised_debater")
    stance_based = common.compromised_debater_from_stance(transcript)
    if declared is not None and declared not in common.DEBATER_NAMES:
        raise ValueError(
            f"{candidate.get('item_id')}: compromised_debater={declared!r} is not a "
            f"debater name {common.DEBATER_NAMES}"
        )
    if declared is not None:
        if transcript.get(declared) != "Y_true":
            raise ValueError(
                f"{candidate.get('item_id')}: compromised_debater {declared!r} does not "
                "defend Y_true; refusing to edit a Y_false argument"
            )
        return declared
    if stance_based not in common.DEBATER_NAMES:
        raise ValueError(
            f"{candidate.get('item_id')}: cannot resolve the Y_true debater from the "
            "stance map and no compromised_debater is declared"
        )
    return stance_based


def _editable_split(candidate: Mapping[str, Any], story: str, prompt: str) -> tuple[str, str]:
    """Split a production prompt immediately before the compromised final quote."""

    compromised = resolve_compromised_debater(candidate)
    rounds = candidate["transcript"].get("rounds", [])
    if not rounds:
        raise ValueError(f"{candidate.get('item_id')}: transcript has no rounds")
    if compromised not in rounds[-1]:
        raise ValueError(
            f"{candidate.get('item_id')}: final round has no {compromised!r} argument"
        )

    # score_verifier.render_verified_transcript is the production rendering function;
    # reproduce only the final line needed to locate the safe insertion boundary.
    final_argument = verify_quotes(rounds[-1][compromised], normalize_text(story))
    line = f'{compromised}: """{final_argument}"""'
    positions = [index for index in range(len(prompt)) if prompt.startswith(line, index)]
    if len(positions) != 1:
        raise ValueError(
            f"{candidate.get('item_id')}: expected one final compromised argument line, "
            f"found {len(positions)}"
        )
    line_start = positions[0]
    closing_quote = line_start + len(line) - 3
    if closing_quote <= line_start or prompt[closing_quote : closing_quote + 3] != '"""':
        raise ValueError(f"{candidate.get('item_id')}: failed to locate closing quote")
    prefix = prompt[:closing_quote] + SUFFIX_SEPARATOR
    suffix = prompt[closing_quote:]
    if "</transcript>" in prefix:
        raise ValueError(f"{candidate.get('item_id')}: editable span fell outside transcript")
    return prefix, suffix


def build_spec(
    candidates: Sequence[Mapping[str, Any]],
    stories: Mapping[str, str],
    *,
    option_seed: int = score_verifier.DEFAULT_OPTION_SEED,
    qy_weight: float = 1.0,
    qh_weight: float = 2.0,
    limit: int | None = None,
    item_id: str | None = None,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    edit_marker: str = DEFAULT_EDIT_MARKER,
) -> dict[str, Any]:
    if qy_weight <= 0 or qh_weight <= 0:
        raise ValueError("Q_Y and Q_H weights must be positive")
    condition_set = {str(c).lower() for c in conditions}
    unknown = condition_set - set(schema.CONDITIONS)
    if unknown:
        raise ValueError(f"unknown condition(s): {sorted(unknown)}")
    tasks: list[dict[str, Any]] = []
    selected = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate rows must be JSON objects")
        # Filter first, then validate: a row we are not going to use must not be able
        # to abort the whole preparation, but every row we DO use is validated hard.
        if candidate.get("condition") not in condition_set:
            continue
        if item_id is not None and candidate.get("item_id") != item_id:
            continue
        if limit is not None and selected >= limit:
            break
        errors = schema.validate_candidate(candidate)
        if errors:
            raise ValueError(
                f"{candidate.get('item_id', '<unknown>')}: invalid candidate: {errors[0]}"
            )
        story_title = candidate["story_title"]
        if story_title not in stories:
            raise ValueError(f"{candidate['item_id']}: story {story_title!r} is missing")
        story = stories[story_title]
        compromised = resolve_compromised_debater(candidate)

        qy_prompt, qy_order = score_verifier.build_qy_prompt(candidate, story, option_seed)
        qh_prompt, qh_order = score_verifier.build_qh_prompt(candidate, story, option_seed)
        qy_prefix, qy_suffix = _editable_split(candidate, story, qy_prompt)
        qh_prefix, qh_suffix = _editable_split(candidate, story, qh_prompt)
        qy_target = _target_letter(qy_order, "Y_true")
        qh_target = _target_letter(qh_order, "H_false")
        native_argument = str(candidate["transcript"]["rounds"][-1][compromised])
        if '"""' in native_argument:
            raise ValueError(
                f"{candidate['item_id']}: final argument already contains '\"\"\"'; "
                "the transcript rendering would be ambiguous"
            )
        base_key = (
            candidate["item_id"], candidate["condition"], int(candidate["candidate_idx"])
        )
        metadata = {
            "candidate_key": list(base_key),
            "story_title": story_title,
            "compromised_debater": compromised,
            "final_round_index": len(candidate["transcript"]["rounds"]) - 1,
            "native_final_argument_sha256": hashlib.sha256(
                native_argument.encode("utf-8")
            ).hexdigest(),
            "qy_target_semantic": "Y_true",
            "qh_target_semantic": "H_false",
            "qy_answer_order": dict(qy_order),
            "qh_answer_order": dict(qh_order),
            "base_qy_prompt_sha256": hashlib.sha256(qy_prompt.encode("utf-8")).hexdigest(),
            "base_qh_prompt_sha256": hashlib.sha256(qh_prompt.encode("utf-8")).hexdigest(),
            "editable_location": "final_argument_of_compromised_debater",
        }
        tasks.extend(
            [
                {
                    "id": f"{candidate['item_id']}|{candidate['condition']}|{candidate['candidate_idx']}|Q_Y",
                    "group": f"{candidate['item_id']}|{candidate['condition']}|{candidate['candidate_idx']}",
                    "prompt_prefix": qy_prefix,
                    "prompt_suffix": qy_suffix,
                    "target": qy_target,
                    "weight": float(qy_weight),
                    "metadata": {**metadata, "readout": "Q_Y"},
                },
                {
                    "id": f"{candidate['item_id']}|{candidate['condition']}|{candidate['candidate_idx']}|Q_H",
                    "group": f"{candidate['item_id']}|{candidate['condition']}|{candidate['candidate_idx']}",
                    "prompt_prefix": qh_prefix,
                    "prompt_suffix": qh_suffix,
                    "target": qh_target,
                    "weight": float(qh_weight),
                    "metadata": {**metadata, "readout": "Q_H"},
                },
            ]
        )
        selected += 1

    if not tasks:
        raise ValueError("no valid candidates selected for GCG")
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "meta": {
            "source": "adversarial_transcript.score_verifier",
            "prompt_version": score_verifier.PROMPT_VERSION,
            "prefill": score_verifier.FINAL_CHANNEL_PREFILL,
            "generation_prompt_suffix": score_verifier.GENERATION_PROMPT_SUFFIX,
            "scaffold_control_tokens": ["<|channel|>", "<|message|>"],
            "choice_completions": dict(score_verifier.LABEL_COMPLETIONS),
            "edit_marker": edit_marker,
            "suffix_separator": SUFFIX_SEPARATOR,
            "loss_formula": (
                "weighted mean of restricted forced-choice NLLs: "
                f"{qy_weight:g} * L_QY(Y_true) + {qh_weight:g} * L_QH(H_false)"
            ),
            "option_seed": int(option_seed),
            "conditions": sorted(condition_set),
            "selected_candidate_count": selected,
            "task_count": len(tasks),
            "warning": (
                "This specification contains private Q_H prompts. Keep it off the "
                "public transcript path; only the optimized suffix is public output."
            ),
        },
        "tasks": tasks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a production-parity GCG task spec")
    parser.add_argument("--candidates", required=True, help="candidate JSONL from generate.py")
    parser.add_argument("--stories", required=True, help="title-story JSON map")
    parser.add_argument("--out", required=True, help="output gcg-task-v2 JSON")
    parser.add_argument("--option-seed", type=int, default=score_verifier.DEFAULT_OPTION_SEED)
    parser.add_argument("--qy-weight", type=_positive_float, default=1.0)
    parser.add_argument("--qh-weight", type=_positive_float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--item-id", default=None)
    parser.add_argument("--conditions", type=_parse_conditions, default=list(DEFAULT_CONDITIONS),
                        help="comma-separated candidate conditions to prepare; default: adversarial")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing task spec: {args.out}")
    candidates = common.read_jsonl(args.candidates)
    stories = common.load_story_map(args.stories)
    spec = build_spec(
        candidates,
        stories,
        option_seed=args.option_seed,
        qy_weight=args.qy_weight,
        qh_weight=args.qh_weight,
        limit=args.limit,
        item_id=args.item_id,
        conditions=args.conditions,
    )
    common.write_json(args.out, spec)
    print(
        f"[gcg.prepare] wrote {len(spec['tasks'])} tasks for "
        f"{spec['meta']['selected_candidate_count']} candidate(s) to {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
