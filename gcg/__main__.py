"""Small dispatcher so ``python -m gcg --help`` points to the entry points."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "GCG utilities. Single-spec entry points: `python -m gcg.prepare` (build a "
            "task spec), `python -m gcg.run` (optimize one suffix over that spec; add "
            "--preflight-only for tokenizer-side checks without loading the model), "
            "`python -m gcg.apply` (write the suffix back into native candidate JSONL). "
            "Whole-pool entry point: `python -m gcg.run_all` (one INDEPENDENT suffix per "
            "candidate; subcommands prepare / optimize / merge-candidates / score / "
            "merge-scores / summary)."
        )
    )
    parser.parse_args()
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
