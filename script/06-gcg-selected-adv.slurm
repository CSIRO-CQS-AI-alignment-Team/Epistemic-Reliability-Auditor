#!/bin/bash
#SBATCH --job-name=GCG-Adv-Sel
#SBATCH --time=8:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512g
#SBATCH --gres=gpu:1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

module load cuda/12.8.1
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2

echo "[GCG] datasets=QuALITY-H,GPQA model=checkpoints/gpt-oss-20b-bf16-base"
echo "[GCG] run roots=runs/adversarial/{QuALITY-H,GPQA}/gcg"

cp runs/adversarial/QuALITY-H/gcg/candidates-gcg.jsonl runs/adversarial/QuALITY-H/
cp runs/adversarial/QuALITY-H/gcg/verifier_scores-gcg.jsonl runs/adversarial/QuALITY-H/

cp runs/adversarial/GPQA/gcg/candidates-gcg.jsonl runs/adversarial/GPQA/
cp runs/adversarial/GPQA/gcg/verifier_scores-gcg.jsonl runs/adversarial/GPQA/

echo "[GCG] apply fail-closed score, leakage, quote, and Q_Y selection checks"
python3 -m adversarial_transcript.select --candidates "runs/adversarial/QuALITY-H/candidates-gcg.jsonl" --scores "runs/adversarial/QuALITY-H/verifier_scores-gcg.jsonl" --stories "dataset/QuALITY-H/QuALITY-H-title-story.json" --out "runs/adversarial/QuALITY-H/selected-gcg.jsonl" --report "runs/gcg/QuALITY-H/report-gcg.json" --conditions adversarial --strict-schema
python3 -m adversarial_transcript.select --candidates "runs/adversarial/GPQA/candidates-gcg.jsonl" --scores "runs/adversarial/GPQA/verifier_scores-gcg.jsonl" --stories "dataset/GPQA/GPQA-title-story.json" --out "runs/adversarial/GPQA/selected-gcg.jsonl" --report "runs/gcg/GPQA/report-gcg.json" --conditions adversarial --strict-schema

echo "[GCG] validate final artifacts and write summaries"
python3 -m gcg.run_all summary --run-root "runs/gcg/QuALITY-H/independent-two-task"
python3 -m gcg.run_all summary --run-root "runs/gcg/GPQA/independent-two-task"

echo "[GCG] merge canonical adversarial transcripts into the paired datasets"
python3 merge_clean_adv.py --dataset QuALITY-H --gcg
python3 merge_clean_adv.py --dataset GPQA --gcg