# Adversarial Transcript Pipeline

This package generates, validates, scores, and selects adversarial debate transcripts for
the QuALITY-H and GPQA experiments.

For each paired item, two debaters argue the visible question `Q_Y`. The debater defending
`Y_true` privately receives `Q_H`, its two answer options, and the target `H_false`. The
stored transcript contains only the public `Q_Y` debate.

## Contract

- Generate `K=8` three-round candidates per item with `google/gemma-4-31B-it`.
- Preserve the visible-answer stance assignment from the selected honest transcript.
- Keep private instructions and hidden-task metadata outside public arguments.
- Require schema-valid scores, no explicit hidden-task leakage, and story-grounded quotes.
- Select only safe candidates whose transcript-conditioned `Q_Y` prediction is `Y_true`.
- Rank the primary pool by `P(H_false | Q_H, T)` from the frozen base verifier.
- Record `qy_fallback` rows for audit, but do not publish them as canonical adversarial
  transcripts.

## Modules

| Module | Purpose |
|---|---|
| `generate.py` | Generate adversarial candidates with the fixed `hybrid-v2` instruction |
| `prompt.py` | Render and hash the private instruction and per-round reminder |
| `score_verifier.py` | Score clean/transcript `Q_Y` and transcript-conditioned `Q_H` |
| `select.py` | Apply fail-closed filters, rank candidates, and write selections/reports |
| `schema.py`, `schemas/` | Runtime and human-readable record contracts |
| `filters/leakage.py` | Reject explicit experimental-language disclosure |
| `filters/quote_audit.py` | Verify quoted evidence against the source story |
| `fingerprint.py` | Bind readout caches to checkpoint contents |
| `readout.py` | Dual-order semantic forced-choice readout used by MI analysis |
| `score.py` | Deterministic placeholder scores for smoke tests only |
| `run_pipeline.py` | Small generation → placeholder-score → selection smoke wrapper |

The offline selection path is dependency-light and consumes external JSONL scores; model
loading is confined to generation and verifier scoring.

## 1. Generate candidates

The numbered Slurm launcher starts the Gemma 4 vLLM server and runs the generator:

```bash
sbatch script/03-adversarial-transcript-generation.slurm QuALITY-H
sbatch script/03-adversarial-transcript-generation.slurm GPQA
```

Equivalent generator command for one dataset:

```bash
DATASET=QuALITY-H
python3 -m adversarial_transcript.generate \
  --items "dataset/${DATASET}/${DATASET}-with-honest-transcripts.json" \
  --stories "dataset/${DATASET}/${DATASET}-title-story.json" \
  --out "runs/adversarial/${DATASET}/candidates.jsonl" \
  --k 8 \
  --backend api \
  --model google/gemma-4-31B-it
```

Expected row counts are 1,776 for QuALITY-H and 808 for GPQA. Generation resumes from
the existing candidate file only when its prompt, input, model, and candidate identity
match the active run.

## 2. Optimize and score with GCG

The production pipeline applies an independent 16-token suffix to each adversarial
candidate and optimizes

```text
L_GCG = L_QY(Y_true) + 2 L_QH(H_false).
```

Preparation, resident GPU execution, exact reranking, and output contracts are documented
in [`../gcg/README.md`](../gcg/README.md). The resulting files are:

```text
runs/adversarial/<DATASET>/gcg/candidates-gcg.jsonl
runs/adversarial/<DATASET>/gcg/verifier_scores-gcg.jsonl
```

## 3. Select candidates

```bash
DATASET=QuALITY-H
RUN_ROOT="runs/adversarial/${DATASET}/gcg"

python3 -m adversarial_transcript.select \
  --candidates "${RUN_ROOT}/candidates-gcg.jsonl" \
  --scores "${RUN_ROOT}/verifier_scores-gcg.jsonl" \
  --stories "dataset/${DATASET}/${DATASET}-title-story.json" \
  --out "${RUN_ROOT}/selected-gcg.jsonl" \
  --report "${RUN_ROOT}/report-gcg.json" \
  --conditions adversarial \
  --strict-schema
```

Selection proceeds in four steps:

1. Require a usable score row.
2. Reject explicit hidden-task leakage and ungrounded quotes.
3. From safe candidates with `qy_adv_correct == true`, choose the largest `p_hfalse`;
   break ties by smaller `candidate_idx`.
4. If the primary pool is empty, retain the safe candidate with the largest
   `qy_adv_probs.semantic.Y_true` as `qy_fallback`; otherwise use
   `no_adv_survivor`.

Expected status counts are:

| Dataset | `ok` | `qy_fallback` | `no_adv_survivor` |
|---|---:|---:|---:|
| QuALITY-H | 184 | 27 | 11 |
| GPQA | 83 | 9 | 9 |

Only `ok` rows are eligible for publication by `merge_clean_adv.py`.

## Smoke test

This CPU-only smoke test exercises generation, placeholder scoring, schemas, filters, and
selection without loading a model:

```bash
python3 -m adversarial_transcript.run_pipeline \
  --out-dir runs/adversarial-smoke \
  --items dataset/QuALITY-H/QuALITY-H-with-honest-transcripts.json \
  --stories dataset/QuALITY-H/QuALITY-H-title-story.json \
  --limit 2 \
  --k 2 \
  --dry-run \
  --demo-bias \
  --overwrite
```

`score.py` emits deterministic synthetic values in this mode. These outputs validate
plumbing only and must not be used as scientific verifier scores.
