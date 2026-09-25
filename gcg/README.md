# GCG Suffix Optimization

This package applies Greedy Coordinate Gradient (GCG) search to the adversarial debate
candidates. Each candidate receives its own editable 16-token suffix in the compromised
`Y_true` defender's final public argument.

The frozen base verifier optimizes the Section 6 objective

```text
L_GCG = L_QY(Y_true) + 2 L_QH(H_false).
```

`L_QY` preserves the visible answer, while `L_QH` elicits steering toward the targeted
hidden answer. Candidate tokens are proposed from input gradients and accepted only after
an exact forward-loss evaluation.

## Pipeline

```text
prepare candidate-specific q_m/q_h tasks
  -> optimize one suffix per candidate
  -> apply suffixes to native transcripts
  -> merge the complete candidate pool
  -> rescore q_m and q_h with the resident verifier replicas
  -> fail-closed selection and summary
```

GCG is white-box: it backpropagates through the verifier's input embeddings and reads the
embedding matrix. An inference endpoint that exposes only text or log probabilities cannot
replace the local Hugging Face model in this stage.

## Modules

| Module | Purpose |
|---|---|
| `prepare.py` | Build production-aligned `q_m` and `q_h` task specifications |
| `objective.py` | Compile prompts, compute restricted A/B NLL, gradients, and exact scores |
| `optimizer.py` | Propose, filter, and rerank token substitutions |
| `run.py` | Optimize one task specification or run tokenizer-only preflight |
| `apply.py` | Write a verified suffix back to a native candidate |
| `run_all.py` | Manage whole-pool manifests, merges, scoring, and summaries |
| `resident_pool.py` | Reuse persistent verifier replicas across optimization and scoring |

## Core invariants

1. **Production readout position.** Both tasks score the next `" A"`/`" B"` token at the
   Harmony final-channel `Answer:` slot used by the verifier scorer.
2. **Text fidelity.** Every proposed suffix must survive decode/encode and exact in-context
   retokenization. The text written into the transcript must denote the exact token sequence
   evaluated by GCG.
3. **Independent candidates.** `run_all` creates one two-task specification per source
   candidate; suffixes are never shared across candidates.
4. **Complete-pool finalization.** Merge, scoring, and selection require full manifest
   coverage and reject missing, duplicate, stale, or mismatched artifacts.

## 1. Prepare the run root

Run once per dataset after adversarial candidate generation:

```bash
DATASET=QuALITY-H
RUN_ROOT="runs/adversarial/${DATASET}/gcg"

python3 -m gcg.run_all prepare \
  --source-candidates "runs/adversarial/${DATASET}/candidates.jsonl" \
  --stories "dataset/${DATASET}/${DATASET}-title-story.json" \
  --run-root "${RUN_ROOT}" \
  --conditions adversarial \
  --expected-k 8
```

Preparation validates the complete `K=8` pool and writes:

```text
<RUN_ROOT>/manifest.jsonl
<RUN_ROOT>/prepare-config.json
<RUN_ROOT>/native/
<RUN_ROOT>/tasks/
```

Each task specification contains exactly one candidate's `q_m` and `q_h` readouts, both
sharing the same editable suffix.

## 2. Optional tokenizer-only preflight

Validate one prepared specification before requesting the full GPU allocation:

```bash
python3 -m gcg.run \
  --model checkpoints/gpt-oss-20b-bf16-base \
  --tasks "${RUN_ROOT}/tasks/000000.json" \
  --suffix-len 16 \
  --preflight-only
```

This checks the final-channel scaffold, marker location, editable vocabulary, and initial
suffix without loading model weights.

## 3. Run one resident-worker layout

The provided launchers allocate four GPUs. Choose one layout per dataset:

```bash
# Four workers, one complete verifier replica per GPU
sbatch script/04-gcg-generated-adv.slurm QuALITY-H
sbatch script/04-gcg-generated-adv.slurm GPQA

# OR two workers, one verifier replica sharded across each two-GPU group
sbatch script/05-gcg-generated-adv-mulGPU.slurm QuALITY-H
sbatch script/05-gcg-generated-adv-mulGPU.slurm GPQA
```

Both launchers run `python -m gcg.resident_pool` with the same logical defaults:

| Parameter | Value |
|---|---:|
| optimization steps | 8 |
| gradient proposal top-k | 32 |
| exact candidates per step | 64 |
| exact-evaluation batch size | 8 |
| suffix length | 16 tokens |
| random seed | 0 |

Do not run both layouts against the same run root concurrently.

## Outputs

A complete resident run writes:

```text
<RUN_ROOT>/results/
<RUN_ROOT>/applied/
<RUN_ROOT>/done/
<RUN_ROOT>/scores/
<RUN_ROOT>/candidates-gcg.jsonl
<RUN_ROOT>/verifier_scores-gcg.jsonl
```

Each done marker is bound to the source candidate, story, task specification, result, and
applied transcript. A resumed run revalidates those bindings before reusing an artifact.
Changing the source pool, stories, model, prompt contract, `K`, or optimization settings
requires a new run root or an explicit rebuild.

## 4. Select and summarize

```bash
python3 -m adversarial_transcript.select \
  --candidates "${RUN_ROOT}/candidates-gcg.jsonl" \
  --scores "${RUN_ROOT}/verifier_scores-gcg.jsonl" \
  --stories "dataset/${DATASET}/${DATASET}-title-story.json" \
  --out "${RUN_ROOT}/selected-gcg.jsonl" \
  --report "${RUN_ROOT}/report-gcg.json" \
  --conditions adversarial \
  --strict-schema

python3 -m gcg.run_all summary --run-root "${RUN_ROOT}"
```

The selector requires valid scores, no explicit hidden-task leakage, grounded quotes, and
`q_m -> Y_true` for the primary branch. It then chooses the candidate with the largest
`P(H_false | q_h, T)`. Fallback rows remain audit records and are not canonical attacks.

Before `merge_clean_adv.py --gcg`, copy the validated selected file to the dataset-level
path consumed by that merger:

```bash
cp "${RUN_ROOT}/selected-gcg.jsonl" \
  "runs/adversarial/${DATASET}/selected-gcg.jsonl"
python3 merge_clean_adv.py --dataset "${DATASET}" --gcg
```

The summary and merge should run only after the full manifest has passed validation.
