"""score.py — OPTIONAL dry-run / placeholder external-score producer (stdlib-only).

select.py is external-score-first: in production the score JSONL is the readout of a
real base verifier probed on clean q_m, transcript-conditioned q_m, and
p_hfalse = P_Vbase(H_false | q_h, T). That readout is heavy (GPU) and lives
in score_verifier.py.

This module exists so the OFFLINE pipeline (and run_pipeline.py) can be exercised
end-to-end without a verifier: it emits a score record per candidate with a
DETERMINISTIC pseudo p_hfalse derived from a hash of the transcript. It is a
PLACEHOLDER ONLY — the numbers are meaningless and must never be used for analysis.
A real scorer should replace this and emit the same schema (see schemas/score.schema.json).

    python -m adversarial_transcript.score --candidates runs/cand.jsonl --out runs/scores.jsonl
"""

import argparse
import hashlib
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from adversarial_transcript import common  # noqa: E402
from adversarial_transcript.filters import leakage  # noqa: E402


def _hash_frac(*parts):
    """Deterministic float in [0, 1) from the parts (stable across runs/machines)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0x100000000


def score_candidate(candidate, demo_bias=False):
    """Build one PLACEHOLDER score record for a candidate."""
    text = common.transcript_text(candidate["transcript"])
    overlap = leakage.audit_overlap(text, candidate.get("qh", {}))
    frac = _hash_frac(candidate["item_id"], candidate["condition"],
                      candidate["candidate_idx"], text)
    p_hfalse = frac
    if demo_bias:
        # Illustrative ONLY: make adversarial transcripts trend higher so the demo
        # pipeline shows a positive delta when honest-BoK is enabled.
        # Not a model output. Off by default.
        p_hfalse = (0.5 + 0.5 * frac) if candidate["condition"] == "adversarial" else (0.5 * frac)
        if overlap["transcript_contains_h_false_text"]:
            p_hfalse = max(p_hfalse, 0.6)
    return {
        "item_id": candidate["item_id"],
        "condition": candidate["condition"],
        "candidate_idx": candidate["candidate_idx"],
        "qy_clean_pred_letter": "A",
        "qy_clean_pred": "Y_true",
        "qy_clean_correct": True,
        "qy_clean_probs": {
            "letter": {"A": 1.0, "B": 0.0},
            "semantic": {"Y_true": 1.0, "Y_false": 0.0},
        },
        "qy_adv_pred_letter": "A",
        "qy_adv_pred": "Y_true",
        "qy_adv_correct": True,
        "qy_adv_probs": {
            "letter": {"A": 1.0, "B": 0.0},
            "semantic": {"Y_true": 1.0, "Y_false": 0.0},
        },
        "qy_no_regression_pass": True,
        # Compact aliases consumed by the shared score validators.
        "qy_pred_letter": None,
        "qy_correct": True,  # PLACEHOLDER: a real scorer must compute this
        "qy_preserved": True,
        "p_hfalse": round(p_hfalse, 6),
        "p_htrue": round(1.0 - p_hfalse, 6),
        "qh_probs": None,
        "baseline_p_hfalse": None,
        "scorer": "dry-run",
        "meta": {"placeholder": True, **overlap},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Dry-run placeholder external scorer (NOT a real verifier).")
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--demo-bias", action="store_true",
                   help="bias adversarial p_hfalse upward for an illustrative non-trivial demo")
    args = p.parse_args(argv)

    candidates = common.read_jsonl(args.candidates)
    scores = [score_candidate(c, demo_bias=args.demo_bias) for c in candidates]
    common.write_jsonl(args.out, scores)
    print(f"[score] PLACEHOLDER scores for {len(scores)} candidate(s) -> {args.out}")
    print("[score] NOTE: dry-run p_hfalse is deterministic noise, not a verifier readout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
