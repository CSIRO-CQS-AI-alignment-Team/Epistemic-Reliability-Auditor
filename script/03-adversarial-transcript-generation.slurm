#!/bin/bash
#SBATCH --job-name=adv-trans-generate
#SBATCH --time=36:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512g
#SBATCH --gres=gpu:4

module load cuda/12.8.1

DATASET=$1

if [[ ! "$DATASET" =~ ^(BoolQ|TruthfulQA|GPQA|QuALITY-H|QuALITY-H-50|QuALITY-H-250)$ ]]; then
    echo "Invalid dataset name. Supported values: BoolQ, TruthfulQA, GPQA, QuALITY-H, QuALITY-H-50, QuALITY-H-250."
    exit 1
fi

echo "==Start the VLLM Services=="
source "${GEMMA_VLLM_ENV_ACTIVATE:?Set GEMMA_VLLM_ENV_ACTIVATE to the environment activation script}" || exit 2
vllm serve google/gemma-4-31B-it --host 127.0.0.1 --port 18888 --max-model-len 262144 --tensor-parallel-size 4 --enable-chunked-prefill --dtype bfloat16 &

sleep 15m

echo "Generating adversarial transcripts for dataset: $DATASET"
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2
python -m adversarial_transcript.generate --items dataset/${DATASET}/${DATASET}-with-honest-transcripts.json --stories dataset/${DATASET}/${DATASET}-title-story.json --out runs/adversarial/${DATASET}/candidates.jsonl --k 8 --model google/gemma-4-31B-it --backend api