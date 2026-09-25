#!/bin/bash
#SBATCH --job-name=GCG-Adv-Sel
#SBATCH --time=2:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64g
#SBATCH --gres=gpu:4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

DATASET=$1

if [[ ! "$DATASET" =~ ^(QuALITY-H|QuALITY-H-50|QuALITY-H-250|GPQA)$ ]]; then
    echo "Invalid dataset name. Supported values: QuALITY-H, QuALITY-H-50, QuALITY-H-250, GPQA."
    exit 1
fi

module load cuda/12.8.1
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2

echo "[GCG] prepare per-candidate Q_Y/Q_H task specifications"
python3 -m gcg.run_all prepare --source-candidates "runs/adversarial/${DATASET}/candidates.jsonl" --stories "dataset/${DATASET}/${DATASET}-title-story.json" --run-root "runs/adversarial/${DATASET}/gcg" --conditions adversarial --expected-k 8

echo "[GCG] optimize and score $DATASET with two two-GPU resident workers (This is for CUDA out-of-memory issue on 4-GPU workers)"
python3 -m gcg.resident_pool --run-root "runs/adversarial/${DATASET}/gcg" --stories "dataset/${DATASET}/${DATASET}-title-story.json" --model "checkpoints/gpt-oss-20b-bf16-base" --workers 2 --gpus-per-worker 2 --dtype bfloat16