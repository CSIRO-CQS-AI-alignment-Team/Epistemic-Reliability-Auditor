#!/bin/bash
#SBATCH --job-name=Debate-BoK
#SBATCH --time=12:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=256g
#SBATCH --gres=gpu:4

module load cuda/12.8.1

DATASET=$1
if [[ ! "$DATASET" =~ ^(QuALITY-H|QuALITY-H-50|QuALITY-H-250|GPQA)$ ]]; then
    echo "Invalid dataset name. Supported values: QuALITY-H, QuALITY-H-50, QuALITY-H-250, GPQA."
    exit 1
fi

echo "==START the Debater LLM=="
source "${GEMMA_VLLM_ENV_ACTIVATE:?Set GEMMA_VLLM_ENV_ACTIVATE to the environment activation script}" || exit 2
vllm serve google/gemma-4-31B-it --host 127.0.0.1 --port 18888 --max-model-len 262144 --tensor-parallel-size 4 --enable-chunked-prefill --dtype bfloat16 &

sleep 20m

echo "== Start Generate Honest Debate Transcript($DATASET) (BoK=8) =="
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2
python3 debate-bok.py --base-url http://localhost:18888/v1 --dataset $DATASET
