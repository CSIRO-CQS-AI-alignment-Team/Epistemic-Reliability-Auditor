#!/bin/bash
#SBATCH --job-name=D-ss-hdt-bok
#SBATCH --time=8:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=256g
#SBATCH --gres=gpu:1

module load cuda/12.8.1

DATASET=$1
if [[ ! "$DATASET" =~ ^(BoolQ|TruthfulQA|GPQA|QuALITY-H|QuALITY-H-50|QuALITY-H-250)$ ]]; then
    echo "Invalid dataset name. Supported values: BoolQ, TruthfulQA, GPQA, QuALITY-H, QuALITY-H-50, QuALITY-H-250."
    exit 1
fi

echo "==Score and Select Honest Debate Transcript (Dataset: $DATASET) (BoK=8)=="
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2
python3 honest_select.py --stage all --dataset $DATASET