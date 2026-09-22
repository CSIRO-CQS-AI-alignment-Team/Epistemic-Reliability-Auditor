#!/bin/bash
#SBATCH --job-name=GCG-Adv-Sel
#SBATCH --time=2:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64g
#SBATCH --gres=gpu:4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

DATASET=$1

if [[ ! "$DATASET" =~ ^(BoolQ|TruthfulQA|GPQA|QuALITY-H|QuALITY-H-50|QuALITY-H-250)$ ]]; then
    echo "Invalid dataset name. Supported values: BoolQ, TruthfulQA, GPQA, QuALITY-H, QuALITY-H-50, QuALITY-H-250."
    exit 1
fi

module load cuda/12.8.1
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2

echo "[GCG] prepare per-candidate Q_Y/Q_H task specifications"
python3 -m gcg.run_all prepare --source-candidates "runs/adversarial/${DATASET}/candidates.jsonl" --stories "dataset/${DATASET}/${DATASET}-title-story.json" --run-root "runs/adversarial/${DATASET}/gcg" --conditions adversarial --expected-k 8

echo "[GCG] optimize and score $DATASET with four single-GPU resident workers"
python3 -m gcg.resident_pool --run-root "runs/adversarial/${DATASET}/gcg" --stories "dataset/${DATASET}/${DATASET}-title-story.json" --model "checkpoints/gpt-oss-20b-bf16-base" --workers 4 --dtype bfloat16