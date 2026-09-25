#!/bin/bash
#SBATCH --job-name=audit-stage2-eval
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=72
#SBATCH --mem=256g

START_TIME=$SECONDS
# This cache-only sweep needs the project's CPU-side audit dependencies.
source "${AUDIT_STAGE2_EVAL_ENV_ACTIVATE:?Set AUDIT_STAGE2_EVAL_ENV_ACTIVATE to the environment activation script}" || exit 2

mkdir -p logs/audit-stage2-eval 
LOG_DIR=$(mktemp -d "logs/audit-stage2-eval/job-${SLURM_JOB_ID:-manual}.XXXXXX")
printf 'Logs: %s\nBackground groups: 8; serial audits/group: 18; CPU thread cap/process: %s\n' "$LOG_DIR"


run_one() {
    local rc logfile
    group_attempted=$((group_attempted + 1))
    printf -v logfile '%s/%s-%02d.log' "$LOG_DIR" "$group_name" "$group_attempted"
    printf '[%s %02d/18] START log=%s\n' "$group_name" "$group_attempted" "$logfile"
    if "$@" >"$logfile" 2>&1; then
        rc=0
    else
        rc=$?
        group_failed=$((group_failed + 1))
    fi
    printf '%s\t%s\t%s\n' "$group_attempted" "$rc" "$logfile" >>"$LOG_DIR/$group_name.status.tsv"
    printf '[%s %02d/18] END exit=%d\n' "$group_name" "$group_attempted" "$rc"
    return 0
}

pids=()
names=()

# Group 1/8: runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json
(
    group_name=qwen3.5-9B__QuALITY-H__adv
    group_attempted=0
    group_failed=0
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 10 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.2-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 10 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.15-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 10 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.1-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 10 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.05-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 10 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.01-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 10 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.001-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 25 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.2-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 25 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.15-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 25 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.1-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 25 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.05-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 25 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.01-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 25 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.001-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 50 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.2-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 50 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.15-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 50 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.1-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 50 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.05-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 50 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.01-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --p 50 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/QuALITY-H-0.001-50/
    printf "[%s] GROUP FINISHED attempted=%d failed=%d\n" "$group_name" "$group_attempted" "$group_failed"
    (( group_attempted == 18 && group_failed == 0 ))
) &
pids+=("$!")
names+=(qwen3.5-9B__QuALITY-H__adv)

# Group 2/8: runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json
(
    group_name=qwen3.5-9B__QuALITY-H__honest
    group_attempted=0
    group_failed=0
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 10 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.2-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 10 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.15-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 10 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.1-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 10 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.05-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 10 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.01-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 10 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.001-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 25 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.2-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 25 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.15-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 25 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.1-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 25 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.05-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 25 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.01-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 25 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.001-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 50 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.2-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 50 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.15-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 50 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.1-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 50 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.05-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 50 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.01-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset QuALITY-H --honest --p 50 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/QuALITY-H-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/QuALITY-H-0.001-50/
    printf "[%s] GROUP FINISHED attempted=%d failed=%d\n" "$group_name" "$group_attempted" "$group_failed"
    (( group_attempted == 18 && group_failed == 0 ))
) &
pids+=("$!")
names+=(qwen3.5-9B__QuALITY-H__honest)

# Group 3/8: runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json
(
    group_name=qwen3.5-9B__GPQA__adv
    group_attempted=0
    group_failed=0
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 10 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.2-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 10 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.15-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 10 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.1-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 10 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.05-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 10 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.01-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 10 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.001-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 25 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.2-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 25 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.15-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 25 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.1-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 25 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.05-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 25 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.01-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 25 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.001-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 50 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.2-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 50 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.15-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 50 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.1-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 50 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.05-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 50 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.01-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --p 50 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-strict.json --output-dir runs/audit-stage2-qwen3.5-9B/GPQA-0.001-50/
    printf "[%s] GROUP FINISHED attempted=%d failed=%d\n" "$group_name" "$group_attempted" "$group_failed"
    (( group_attempted == 18 && group_failed == 0 ))
) &
pids+=("$!")
names+=(qwen3.5-9B__GPQA__adv)

# Group 4/8: runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json
(
    group_name=qwen3.5-9B__GPQA__honest
    group_attempted=0
    group_failed=0
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 10 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.2-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 10 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.15-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 10 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.1-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 10 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.05-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 10 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.01-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 10 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.001-10/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 25 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.2-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 25 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.15-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 25 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.1-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 25 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.05-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 25 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.01-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 25 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.001-25/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 50 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.2-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 50 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.15-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 50 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.1-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 50 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.05-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 50 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.01-50/
    run_one python3 -u audit-stage2.py --model qwen3.5-9B --dataset GPQA --honest --p 50 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/qwen3.5-9B/GPQA-honest.json --output-dir runs/audit-stage2-qwen3.5-9B-honest/GPQA-0.001-50/
    printf "[%s] GROUP FINISHED attempted=%d failed=%d\n" "$group_name" "$group_attempted" "$group_failed"
    (( group_attempted == 18 && group_failed == 0 ))
) &
pids+=("$!")
names+=(qwen3.5-9B__GPQA__honest)

# Group 5/8: runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json
(
    group_name=llama3.1-8b-it__QuALITY-H__adv
    group_attempted=0
    group_failed=0
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 10 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.2-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 10 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.15-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 10 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.1-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 10 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.05-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 10 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.01-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 10 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.001-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 25 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.2-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 25 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.15-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 25 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.1-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 25 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.05-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 25 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.01-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 25 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.001-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 50 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.2-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 50 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.15-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 50 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.1-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 50 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.05-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 50 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.01-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --p 50 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/QuALITY-H-0.001-50/
    printf "[%s] GROUP FINISHED attempted=%d failed=%d\n" "$group_name" "$group_attempted" "$group_failed"
    (( group_attempted == 18 && group_failed == 0 ))
) &
pids+=("$!")
names+=(llama3.1-8b-it__QuALITY-H__adv)

# Group 6/8: runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json
(
    group_name=llama3.1-8b-it__QuALITY-H__honest
    group_attempted=0
    group_failed=0
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 10 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.2-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 10 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.15-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 10 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.1-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 10 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.05-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 10 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.01-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 10 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.001-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 25 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.2-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 25 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.15-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 25 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.1-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 25 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.05-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 25 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.01-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 25 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.001-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 50 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.2-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 50 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.15-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 50 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.1-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 50 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.05-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 50 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.01-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset QuALITY-H --honest --p 50 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/QuALITY-H-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/QuALITY-H-0.001-50/
    printf "[%s] GROUP FINISHED attempted=%d failed=%d\n" "$group_name" "$group_attempted" "$group_failed"
    (( group_attempted == 18 && group_failed == 0 ))
) &
pids+=("$!")
names+=(llama3.1-8b-it__QuALITY-H__honest)

# Group 7/8: runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json
(
    group_name=llama3.1-8b-it__GPQA__adv
    group_attempted=0
    group_failed=0
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 10 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.2-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 10 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.15-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 10 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.1-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 10 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.05-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 10 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.01-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 10 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.001-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 25 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.2-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 25 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.15-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 25 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.1-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 25 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.05-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 25 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.01-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 25 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.001-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 50 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.2-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 50 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.15-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 50 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.1-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 50 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.05-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 50 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.01-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --p 50 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-strict.json --output-dir runs/audit-stage2-llama3.1-8b-it/GPQA-0.001-50/
    printf "[%s] GROUP FINISHED attempted=%d failed=%d\n" "$group_name" "$group_attempted" "$group_failed"
    (( group_attempted == 18 && group_failed == 0 ))
) &
pids+=("$!")
names+=(llama3.1-8b-it__GPQA__adv)

# Group 8/8: runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json
(
    group_name=llama3.1-8b-it__GPQA__honest
    group_attempted=0
    group_failed=0
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 10 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.2-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 10 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.15-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 10 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.1-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 10 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.05-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 10 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.01-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 10 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.001-10/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 25 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.2-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 25 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.15-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 25 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.1-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 25 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.05-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 25 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.01-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 25 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.001-25/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 50 --eta 0.2 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.2-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 50 --eta 0.15 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.15-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 50 --eta 0.1 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.1-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 50 --eta 0.05 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.05-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 50 --eta 0.01 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.01-50/
    run_one python3 -u audit-stage2.py --model llama3.1-8b-it --dataset GPQA --honest --p 50 --eta 0.001 --mtest 100 --cache-only --score-cache runs/audit-stage2-score-cache/llama3.1-8b-it/GPQA-honest.json --output-dir runs/audit-stage2-llama3.1-8b-it-honest/GPQA-0.001-50/
    printf "[%s] GROUP FINISHED attempted=%d failed=%d\n" "$group_name" "$group_attempted" "$group_failed"
    (( group_attempted == 18 && group_failed == 0 ))
) &
pids+=("$!")
names+=(llama3.1-8b-it__GPQA__honest)

# Wait for EVERY group, including when an earlier group failed.
# A bare `wait` is insufficient to reliably aggregate all child exit statuses.
overall_status=0
failed_groups=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'GROUP OK: %s\n' "${names[$index]}"
    else
        rc=$?
        overall_status=1
        failed_groups=$((failed_groups + 1))
        printf 'GROUP FAILED: %s (exit=%d); inspect %s/%s.status.tsv\n' \
            "${names[$index]}" "$rc" "$LOG_DIR" "${names[$index]}" >&2
    fi
done

ELAPSED_SECONDS=$((SECONDS - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
SECS=$((ELAPSED_SECONDS % 60))
echo '=========================================='
printf 'All 8 background groups have exited; failed groups: %d\n' "$failed_groups"
printf 'Total execution time: %d seconds\n' "$ELAPSED_SECONDS"
printf 'Formatted time: %02dh:%02dm:%02ds\n' "$HOURS" "$MINUTES" "$SECS"
printf 'Per-command logs and exit codes: %s\n' "$LOG_DIR"
echo '=========================================='
exit "$overall_status"
