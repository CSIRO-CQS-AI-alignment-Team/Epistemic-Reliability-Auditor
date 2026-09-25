#!/bin/bash
#SBATCH --job-name=V-post-FC
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512g
#SBATCH --gres=gpu:2

START_TIME=$SECONDS

module load cuda/12.9.1
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2

MODEL=${1:?Missing model}
DATASET=${2:?Missing dataset}
HONEST=${3:?Expected no or honest}
case "$MODEL" in
    gpt-oss-20b|qwen3.5-9B|llama3.1-8b-it) ;;
    *) echo "Invalid model: $MODEL" >&2; exit 2 ;;
esac
case "$DATASET" in
    QuALITY-H|GPQA) ;;
    *) echo "Invalid dataset: $DATASET" >&2; exit 2 ;;
esac
case "$HONEST" in
    no|honest) ;;
    *) echo "Expected no or honest, got: $HONEST" >&2; exit 2 ;;
esac

if [ "$HONEST" = "honest" ]; then
    python3 audit-stage2.py --model "$MODEL" --dataset "$DATASET" --warm-cache --mtest 100 --honest --devices 0,1
else
    python3 audit-stage2.py --model "$MODEL" --dataset "$DATASET" --warm-cache --mtest 100 --devices 0,1
fi
status=$?

ELAPSED_SECONDS=$(( $SECONDS - $START_TIME ))

# 3. Rewrite as HH:MM:SS format
HOURS=$(( ELAPSED_SECONDS / 3600 ))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECS=$(( ELAPSED_SECONDS % 60 ))

# 4. print the total execution time
echo "=========================================="
echo "Total execution time: ${ELAPSED_SECONDS} seconds"
printf "Formatted time: %02dh:%02dm:%02ds\n" "$HOURS" "$MINUTES" "$SECS"
echo "=========================================="
exit "$status"