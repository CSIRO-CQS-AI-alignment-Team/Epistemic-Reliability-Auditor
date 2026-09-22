#!/bin/bash
#SBATCH --job-name=V-FT-grounded
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512g
#SBATCH --gres=gpu:4

module load cuda/12.8.1
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2

DATASET=$1

if [[ ! "$DATASET" =~ ^(BoolQ|TruthfulQA|GPQA|QuALITY-H|QuALITY-H-50|QuALITY-H-250)$ ]]; then
    echo "Invalid dataset name. Supported values: BoolQ, TruthfulQA, GPQA, QuALITY-H, QuALITY-H-50, QuALITY-H-250."
    exit 1
fi

accelerate launch --config_file config/accelerate-zero3.yaml ft-verifier-oss-full.py \
    --dataset "$DATASET" --supervision grounded --epochs 3 \
    --lambda-lm 1.0 --lambda-fc 2.0 --grounded-unresolved answer-only --output-dir checkpoints/gpt-oss-20b-verifier-fullft-$DATASET
