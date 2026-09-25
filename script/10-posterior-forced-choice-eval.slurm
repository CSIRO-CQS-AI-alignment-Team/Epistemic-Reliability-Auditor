#!/bin/bash
#SBATCH --job-name=V-post-FC
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512g
#SBATCH --gres=gpu:3

module load cuda/12.8.1
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2

DATASET=$1

if [[ ! "$DATASET" =~ ^(QuALITY-H|QuALITY-H-50|QuALITY-H-250|GPQA)$ ]]; then
    echo "Invalid dataset name. Supported values: QuALITY-H, QuALITY-H-50, QuALITY-H-250, GPQA."
    exit 1
fi

mkdir -p runs/posterior_shift/$DATASET

echo "Starting base posterior eval on GPU 0"
python3 verifier-posterior-eval.py \
  --dataset dataset/$DATASET/$DATASET-no-debate.json \
  --model checkpoints/gpt-oss-20b-bf16-base \
  --model-label base \
  --output runs/posterior_shift/$DATASET/base.json \
  --device 0 \
  --questions both \
  > runs/posterior_shift/$DATASET/base.log 2>&1 &
BASE_PID=$!

echo "Starting honest posterior eval on GPU 1"
python3 verifier-posterior-eval.py \
  --dataset dataset/$DATASET/$DATASET-no-debate.json \
  --model checkpoints/gpt-oss-20b-verifier-fullft-$DATASET \
  --model-label honest \
  --output runs/posterior_shift/$DATASET/honest.json \
  --device 1 \
  --questions both \
  > runs/posterior_shift/$DATASET/honest.log 2>&1 &
HONEST_PID=$!

echo "Starting adversarial posterior eval on GPU 2"
python3 verifier-posterior-eval.py \
  --dataset dataset/$DATASET/$DATASET-no-debate.json \
  --model checkpoints/gpt-oss-20b-verifier-fullft-adversarial-$DATASET \
  --model-label adv \
  --output runs/posterior_shift/$DATASET/adv.json \
  --device 2 \
  --questions both \
  > runs/posterior_shift/$DATASET/adv.log 2>&1 &
ADV_PID=$!

BASE_STATUS=0
HONEST_STATUS=0
ADV_STATUS=0

wait "${BASE_PID}" || BASE_STATUS=$?
wait "${HONEST_PID}" || HONEST_STATUS=$?
wait "${ADV_PID}" || ADV_STATUS=$?

echo "base posterior eval exit status: ${BASE_STATUS}"
echo "honest posterior eval exit status: ${HONEST_STATUS}"
echo "adv posterior eval exit status: ${ADV_STATUS}"

if [[ "${BASE_STATUS}" -ne 0 || "${HONEST_STATUS}" -ne 0 || "${ADV_STATUS}" -ne 0 ]]; then
  echo "At least one posterior eval failed. Check runs/posterior_shift/$DATASET/*.log" >&2
  exit 1
fi

echo "All posterior eval jobs completed successfully."
