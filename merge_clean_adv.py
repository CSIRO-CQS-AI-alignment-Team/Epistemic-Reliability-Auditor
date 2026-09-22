#!/usr/bin/env python3
"""Merge honest and selected adversarial transcripts for QuALITY-H or GPQA."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
    return rows


def index_from_row(row: dict[str, Any], fallback_idx: int) -> int:
    adv = row.get("adversarial_selected")
    if isinstance(adv, dict) and isinstance(adv.get("dataset_index"), int):
        return adv["dataset_index"]

    item_id = row.get("item_id", "")
    if isinstance(item_id, str) and item_id.startswith("qh_") and item_id[3:].isdigit():
        return int(item_id[3:])

    return fallback_idx


def transcript_is_well_formed(transcript: Any) -> bool:
    if not isinstance(transcript, dict):
        return False
    if transcript.get("Debater A") not in {"Y_true", "Y_false"}:
        return False
    if transcript.get("Debater B") not in {"Y_true", "Y_false"}:
        return False
    rounds = transcript.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 3:
        return False
    for debate_round in rounds:
        if not isinstance(debate_round, dict):
            return False
        if not isinstance(debate_round.get("Debater A"), str):
            return False
        if not isinstance(debate_round.get("Debater B"), str):
            return False
    return True


def build_selected_index(selected_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for fallback_idx, row in enumerate(selected_rows):
        idx = index_from_row(row, fallback_idx)
        if idx in by_index:
            raise ValueError(f"duplicate selected dataset index: {idx}")
        by_index[idx] = row
    return by_index


def merge_items(
    honest_items: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(honest_items) != len(selected_rows):
        raise ValueError(
            f"count mismatch: honest={len(honest_items)} selected={len(selected_rows)}"
        )

    by_index = build_selected_index(selected_rows)
    expected_indexes = set(range(len(honest_items)))
    actual_indexes = set(by_index)
    if actual_indexes != expected_indexes:
        missing = sorted(expected_indexes - actual_indexes)
        extra = sorted(actual_indexes - expected_indexes)
        raise ValueError(f"selected index coverage mismatch: missing={missing} extra={extra}")

    merged: list[dict[str, Any]] = []
    for idx, item in enumerate(honest_items):
        row = by_index[idx]
        if row.get("story_title") != item.get("story_title"):
            raise ValueError(
                f"story_title mismatch at index {idx}: "
                f"selected={row.get('story_title')!r} honest={item.get('story_title')!r}"
            )

        qy = item.get("Q_Y")
        if not isinstance(qy, dict):
            raise ValueError(f"missing Q_Y at index {idx}")
        if "transcript" not in qy:
            raise ValueError(f"missing Q_Y.transcript at index {idx}")

        new_item = copy.deepcopy(item)
        new_qy = new_item["Q_Y"]
        new_qy["honest_transcript"] = new_qy.pop("transcript")

        if not transcript_is_well_formed(new_qy["honest_transcript"]):
            raise ValueError(f"malformed honest transcript at index {idx}")

        if row.get("status") == "ok":
            adv = row.get("adversarial_selected")
            if not isinstance(adv, dict):
                raise ValueError(f"ok row has no adversarial_selected at index {idx}")
            if adv.get("dataset_index") != idx:
                raise ValueError(
                    f"adversarial_selected dataset_index mismatch at index {idx}: "
                    f"{adv.get('dataset_index')!r}"
                )
            if adv.get("story_title") != item.get("story_title"):
                raise ValueError(f"adversarial_selected story_title mismatch at index {idx}")
            if adv.get("condition") != "adversarial":
                raise ValueError(f"selected condition is not adversarial at index {idx}")

            expected_qy = adv.get("qy")
            if isinstance(expected_qy, dict):
                for key in ("question", "Y_true", "Y_false"):
                    if expected_qy.get(key) != qy.get(key):
                        raise ValueError(f"selected qy.{key} mismatch at index {idx}")

            transcript = adv.get("transcript")
            if not transcript_is_well_formed(transcript):
                raise ValueError(f"malformed adversarial transcript at index {idx}")
            new_qy["adversarial_transcript"] = transcript
        else:
            new_qy["adversarial_transcript"] = None

        merged.append(new_item)

    return merged


def validate_merge(
    honest_items: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    merged: list[dict[str, Any]],
) -> None:
    if len(honest_items) != len(selected_rows) or len(honest_items) != len(merged):
        raise ValueError("merged, honest, and selected counts differ")

    by_index = build_selected_index(selected_rows)
    for idx, (honest_item, merged_item) in enumerate(zip(honest_items, merged)):
        qy = merged_item.get("Q_Y")
        honest_qy = honest_item.get("Q_Y")
        if not isinstance(qy, dict) or not isinstance(honest_qy, dict):
            raise ValueError(f"missing Q_Y at index {idx}")
        if "transcript" in qy:
            raise ValueError(f"leftover Q_Y.transcript at index {idx}")
        if qy.get("honest_transcript") != honest_qy.get("transcript"):
            raise ValueError(f"honest transcript mismatch at index {idx}")

        row = by_index[idx]
        actual_adv = qy.get("adversarial_transcript")
        if row.get("status") == "ok":
            expected_adv = row["adversarial_selected"]["transcript"]
            if actual_adv != expected_adv:
                raise ValueError(f"adversarial transcript mismatch at index {idx}")
        elif actual_adv is not None:
            raise ValueError(f"non-ok row has non-null adversarial transcript at index {idx}")


def sample_audit(
    selected_rows: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> None:
    if sample_size <= 0:
        return
    if sample_size > len(merged):
        raise ValueError(f"sample_size={sample_size} exceeds item count={len(merged)}")

    rng = random.Random(seed)
    sample = sorted(rng.sample(range(len(merged)), sample_size))
    by_index = build_selected_index(selected_rows)
    print(f"Random audit seed={seed} sample_size={sample_size}")
    for idx in sample:
        row = by_index[idx]
        qy = merged[idx]["Q_Y"]
        expected = (
            row.get("adversarial_selected", {}).get("transcript")
            if row.get("status") == "ok"
            else None
        )
        if qy["adversarial_transcript"] != expected:
            raise ValueError(f"sample audit failed at index {idx}")
        transcript = qy["adversarial_transcript"]
        rounds = None if transcript is None else len(transcript.get("rounds", []))
        candidate_idx = None
        if isinstance(row.get("adversarial_selected"), dict):
            candidate_idx = row["adversarial_selected"].get("candidate_idx")
        print(
            f"{idx:03d} {row.get('item_id')} status={row.get('status')} "
            f"candidate={candidate_idx} adv_null={transcript is None} "
            f"adv_rounds={rounds} title={merged[idx].get('story_title')}"
        )
    print("Sample audit passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge QuALITY-H honest transcripts with selected adversarial transcripts."
    )
    parser.add_argument("--dataset", type=str, default="QuALITY-H")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260703)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--gcg", action="store_true", help="Use GCG-style selected.jsonl format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_path = None
    honest_path = Path(f"dataset/{args.dataset}/{args.dataset}-with-honest-transcripts.json")
    if args.gcg:
        selected_path = Path(f"runs/adversarial/{args.dataset}/selected-gcg.jsonl")
    else:
        selected_path = Path(f"runs/adversarial/{args.dataset}/selected.jsonl")
    out_path = Path(f"dataset/{args.dataset}/{args.dataset}.json")

    honest_items = load_json(honest_path)
    selected_rows = load_jsonl(selected_path)
    if not isinstance(honest_items, list):
        raise ValueError(f"{honest_path} must contain a JSON list")

    merged = merge_items(honest_items, selected_rows)
    validate_merge(honest_items, selected_rows, merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=args.indent, ensure_ascii=False) + "\n")

    honest_count = sum(x["Q_Y"].get("honest_transcript") is not None for x in merged)
    adv_count = sum(x["Q_Y"].get("adversarial_transcript") is not None for x in merged)
    adv_null_count = sum(x["Q_Y"].get("adversarial_transcript") is None for x in merged)

    print(f"Wrote {out_path}")
    print(f"items={len(merged)}")
    print(f"honest_transcript={honest_count}")
    print(f"adversarial_transcript={adv_count}")
    print(f"adversarial_transcript_null={adv_null_count}")
    print("Full validation passed")

    sample_audit(selected_rows, merged, args.sample_size, args.sample_seed)


if __name__ == "__main__":
    main()
