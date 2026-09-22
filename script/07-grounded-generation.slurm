#!/bin/bash
#SBATCH --job-name=G-G-HA
#SBATCH --time=8:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512g
#SBATCH --gres=gpu:4

module load cuda/12.8.1
source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2


# 1. Base audit artifacts (torch-free; shared with the other arms).
python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --check-data
python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --check-data
python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --mode adversarial --check-data
python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --mode adversarial --check-data


# 2. Offline grounded targets. Generation is POLICY-INDEPENDENT: it always finishes the
#    split and writes every record, unresolved ones included with their per-attempt
#    failure reasons, so a fail-closed run still leaves a fully inspectable artifact.
source "${GPTOSS_VLLM_ENV_ACTIVATE:?Set GPTOSS_VLLM_ENV_ACTIVATE to the environment activation script}" || exit 2
vllm serve ./checkpoints/gpt-oss-20b-bf16-base --host 127.0.0.1 --port 18888 --max-model-len 131072 --tensor-parallel-size 4 --enable-chunked-prefill --dtype bfloat16 --compilation-config.pass_config.fuse_allreduce_rms false &
sleep 15m


source "${AISI_ENV_ACTIVATE:?Set AISI_ENV_ACTIVATE to the environment activation script}" || exit 2
run_5_times() {
    local cmd="$1"
    local task_name="$2"
    for i in {1..5}; do
        echo "[$task_name] Start to execute $i/5 times..."
        eval "$cmd"
    done
    echo "[$task_name] 5 times loop completed!"
}

echo "Starting Generated Grounded (Honest) in parallel..."
CMD_QUALITY_TRAIN="python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --generate-grounded --split train --teacher-backend api --teacher-model ./checkpoints/gpt-oss-20b-bf16-base --teacher-base-url http://127.0.0.1:18888/v1 --generate-grounded-concurrency 8"
CMD_QUALITY_EVAL="python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --generate-grounded --split eval --teacher-backend api --teacher-model ./checkpoints/gpt-oss-20b-bf16-base --teacher-base-url http://127.0.0.1:18888/v1 --generate-grounded-concurrency 8"
CMD_GPQA_TRAIN="python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --generate-grounded --split train --teacher-backend api --teacher-model ./checkpoints/gpt-oss-20b-bf16-base --teacher-base-url http://127.0.0.1:18888/v1 --generate-grounded-concurrency 8"
CMD_GPQA_EVAL="python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --generate-grounded --split eval --teacher-backend api --teacher-model ./checkpoints/gpt-oss-20b-bf16-base --teacher-base-url http://127.0.0.1:18888/v1 --generate-grounded-concurrency 8"

run_5_times "$CMD_QUALITY_TRAIN" "QuALITY-H-Train" &
PID1=$!

run_5_times "$CMD_QUALITY_EVAL" "QuALITY-H-Eval" &
PID2=$!

run_5_times "$CMD_GPQA_TRAIN" "GPQA-Train" &
PID3=$!

run_5_times "$CMD_GPQA_EVAL" "GPQA-Eval" &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "Honest grounded-generation wait finished; inspect each task result in this log."

echo "Starting Generated Grounded (Adversarial) in parallel..."

CMD_QUALITY_TRAIN_ADV="python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --mode adversarial --generate-grounded --split train --teacher-backend api --teacher-model ./checkpoints/gpt-oss-20b-bf16-base --teacher-base-url http://127.0.0.1:18888/v1 --generate-grounded-concurrency 8"
CMD_QUALITY_EVAL_ADV="python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --mode adversarial --generate-grounded --split eval --teacher-backend api --teacher-model ./checkpoints/gpt-oss-20b-bf16-base --teacher-base-url http://127.0.0.1:18888/v1 --generate-grounded-concurrency 8"
CMD_GPQA_TRAIN_ADV="python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --mode adversarial --generate-grounded --split train --teacher-backend api --teacher-model ./checkpoints/gpt-oss-20b-bf16-base --teacher-base-url http://127.0.0.1:18888/v1 --generate-grounded-concurrency 8"
CMD_GPQA_EVAL_ADV="python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --mode adversarial --generate-grounded --split eval --teacher-backend api --teacher-model ./checkpoints/gpt-oss-20b-bf16-base --teacher-base-url http://127.0.0.1:18888/v1 --generate-grounded-concurrency 8"

run_5_times "$CMD_QUALITY_TRAIN_ADV" "QuALITY-H-Train" &
PID5=$!

run_5_times "$CMD_QUALITY_EVAL_ADV" "QuALITY-H-Eval" &
PID6=$!

run_5_times "$CMD_GPQA_TRAIN_ADV" "GPQA-Train" &
PID7=$!

run_5_times "$CMD_GPQA_EVAL_ADV" "GPQA-Eval" &
PID8=$!

wait $PID5 $PID6 $PID7 $PID8
echo "Adversarial grounded-generation wait finished; inspect each task result in this log."


echo "Grounded artifacts and manifests are written, including unresolved attempts."
echo "Inspect dataset/<DATASET>/*-grounded-manifest.json; rerun this job to retry unresolved items."

# 3. Audit report + tokenizer contracts.
python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --check-grounded --grounded-unresolved "fail"
python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --check-tokenizer --grounded-unresolved "fail"

python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --check-grounded --grounded-unresolved "fail"
python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --check-tokenizer --grounded-unresolved "fail"

python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --mode adversarial --check-grounded --grounded-unresolved "fail"
python3 ft-verifier-oss-full.py --dataset QuALITY-H --supervision grounded --mode adversarial --check-tokenizer --grounded-unresolved "fail"

python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --mode adversarial --check-grounded --grounded-unresolved "fail"
python3 ft-verifier-oss-full.py --dataset GPQA --supervision grounded --mode adversarial --check-tokenizer --grounded-unresolved "fail"


