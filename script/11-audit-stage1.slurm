#!/bin/bash
#SBATCH --job-name=audit-stage1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512g
#SBATCH --gres=gpu:4

module load cuda/12.9.1
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2

MODEL=${1:?Usage: sbatch 11-audit-stage1.slurm MODEL GPQA_EPSILON QUALITY_EPSILON}
GPQA_EPSILON=${2:?Missing GPQA epsilon}
QUALITY_EPSILON=${3:?Missing QuALITY-H epsilon}

# Preserve the published submission spelling while passing the exact Python CLI choice.
if [[ "$MODEL" == gpt-oss-20B ]]; then
    MODEL=gpt-oss-20b
fi
case "$MODEL" in
    gpt-oss-20b|qwen3.5-9B|llama3.1-8b-it) ;;
    *) echo "Invalid model: $MODEL" >&2; exit 2 ;;
esac

# Run serially: each Stage 1 process uses all allocated GPUs by default.
status=0
python3 audit-stage1.py --dataset GPQA --k 0.001 --model "$MODEL" --epsilon "$GPQA_EPSILON" || status=1
python3 audit-stage1.py --dataset QuALITY-H --k 0.001 --model "$MODEL" --epsilon "$QUALITY_EPSILON" || status=1
exit "$status"
