"""Small end-to-end smoke runner for adversarial transcript selection.

This runner wires together:

  generate.py -> score.py (placeholder, optional) -> select.py

Production CTI should replace score.py with a real verifier readout JSONL. The
runner refuses to use placeholder scores on real model outputs unless explicitly
requested.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from adversarial_transcript import generate, score, select  # noqa: E402


def positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate, score, and select debate candidates.")
    p.add_argument("--out-dir", required=True, help="directory for candidates/scores/selections")
    p.add_argument("--items", default=generate.DEFAULT_ITEMS)
    p.add_argument("--stories", default=generate.DEFAULT_STORIES)
    p.add_argument("--k", type=positive_int, default=8)
    p.add_argument("--conditions", default="adversarial",
                   help="comma-separated subset to generate/select; default: adversarial")
    p.add_argument("--num-rounds", "--rounds", dest="num_rounds", type=positive_int, default=3)
    p.add_argument("--start-index", type=generate.nonnegative_int, default=0)
    p.add_argument("--limit", type=positive_int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="generate placeholders and use placeholder scores")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--backend", choices=["api", "transformers"], default=generate.DEFAULT_BACKEND)
    p.add_argument("--model", default=generate.DEFAULT_MODEL)
    p.add_argument("--api-key", default=generate.DEFAULT_API_KEY)
    p.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"],
                   default="auto")
    p.add_argument("--device-map", default=generate.DEFAULT_DEVICE_MAP)
    p.add_argument("--scores", default=None,
                   help="external verifier score JSONL; if omitted, placeholder scoring is used only for --dry-run or --placeholder-score")
    p.add_argument("--placeholder-score", action="store_true",
                   help="allow dry-run score.py even for real generated candidates")
    p.add_argument("--demo-bias", action="store_true",
                   help="pass --demo-bias to placeholder score.py")
    p.add_argument("--quote-max-unverified-frac", type=float, default=0.0)
    p.add_argument("--allow-missing-stories", action="store_true",
                   help="debug only: pass through to select.py")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    candidates_path = os.path.join(args.out_dir, "candidates.jsonl")
    generated_scores_path = os.path.join(args.out_dir, "scores.placeholder.jsonl")
    selections_path = os.path.join(args.out_dir, "selected.jsonl")
    report_path = os.path.join(args.out_dir, "report.json")

    gen_argv = [
        "--items", args.items,
        "--stories", args.stories,
        "--out", candidates_path,
        "--k", str(args.k),
        "--conditions", args.conditions,
        "--num-rounds", str(args.num_rounds),
        "--start-index", str(args.start_index),
        "--seed", str(args.seed),
        "--backend", args.backend,
        "--model", args.model,
        "--api-key", args.api_key,
        "--torch-dtype", args.torch_dtype,
        "--device-map", args.device_map,
    ]
    if args.limit is not None:
        gen_argv += ["--limit", str(args.limit)]
    if args.dry_run:
        gen_argv.append("--dry-run")
    if args.overwrite:
        gen_argv.append("--overwrite")
    generate.main(gen_argv)

    scores_path = args.scores
    if scores_path is None:
        if not args.dry_run and not args.placeholder_score:
            raise SystemExit(
                "real generation requires --scores from an external verifier, "
                "or pass --placeholder-score for a smoke test only"
            )
        score_argv = ["--candidates", candidates_path, "--out", generated_scores_path]
        if args.demo_bias:
            score_argv.append("--demo-bias")
        score.main(score_argv)
        scores_path = generated_scores_path

    select_argv = [
        "--candidates", candidates_path,
        "--scores", scores_path,
        "--stories", args.stories,
        "--out", selections_path,
        "--report", report_path,
        "--conditions", args.conditions,
        "--quote-max-unverified-frac", str(args.quote_max_unverified_frac),
    ]
    if args.allow_missing_stories:
        select_argv.append("--allow-missing-stories")
    select.main(select_argv)
    print(f"[pipeline] candidates: {candidates_path}")
    print(f"[pipeline] scores: {scores_path}")
    print(f"[pipeline] selections: {selections_path}")
    print(f"[pipeline] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
