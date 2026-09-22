#!/bin/bash
#SBATCH --job-name=V-FT-adv
#SBATCH --time=8:00:00
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

echo "Fine-tuning verifier for dataset: $DATASET with grounded LM, Q_Y, and Q_H objectives (weights 1, 2, and 2)"
accelerate launch --config_file config/accelerate-zero3.yaml ft-verifier-oss-full.py \
    --dataset "$DATASET" --mode adversarial --supervision grounded --epochs 3 \
    --lambda-lm 1.0 --lambda-fc 2.0 --grounded-unresolved answer-only --qh-aux --qh-lambda 2 --output-dir checkpoints/gpt-oss-20b-verifier-fullft-adversarial-$DATASET

# echo "Fine-tuning verifier for dataset: $DATASET under adversarial grounded three-view objective (1*L_view1 + 2*L_view2 + $QH_LAMBDA*L_view3)"
# accelerate launch --config_file config/accelerate-zero3.yaml ft-verifier-oss-full.py \
#     --dataset "$DATASET" --mode adversarial --supervision grounded --epochs 2 \
#     --lambda-lm 1.0 --lambda-fc 2.0 --grounded-unresolved answer-only --qh-aux --qh-lambda $QH_LAMBDA --output-dir checkpoints/gpt-oss-20b-verifier-fullft-adversarial-$DATASET-w-QHLoss-fc2-QHLambda-$QH_LAMBDA