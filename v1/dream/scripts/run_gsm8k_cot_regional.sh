#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Match the Dream Instruct zero-shot GSM8K-CoT protocol used by the original
# regional pilot. The very large per-region deferral cap reproduces the old
# uncapped behavior; override it explicitly for capped-deferral ablations.
export MODEL=${MODEL:-Dream-org/Dream-v0-Instruct-7B}
export TASK=gsm8k_cot
export NUM_FEWSHOT=0
export LENGTH=${LENGTH:-256}
export DTYPE=${DTYPE:-bfloat16}
export TEMPERATURE=${TEMPERATURE:-0.1}
export TOP_P=${TOP_P:-0.9}
export APPLY_CHAT_TEMPLATE=true
export ESCAPE_UNTIL=false
export OUTPUT_ROOT=${OUTPUT_ROOT:-evals_results/regional_gsm8k_cot}
export REGION_SIZE=${REGION_SIZE:-32}
export LOCAL_STEPS=${LOCAL_STEPS:-32}
export MAX_PROGRESS_GAP=${MAX_PROGRESS_GAP:-4}
export DEFERRAL_THRESHOLD=${DEFERRAL_THRESHOLD:-0.4}
export DEFERRAL_UNTIL_REVEALED=${DEFERRAL_UNTIL_REVEALED:-2}
export MAX_REGION_DEFERRALS=${MAX_REGION_DEFERRALS:-1000000}
export MAX_GLOBAL_DEFERRALS=${MAX_GLOBAL_DEFERRALS:-4}
export STOP_FILTER_THRESHOLD=${STOP_FILTER_THRESHOLD:-0.7}

exec bash "${SCRIPT_DIR}/run_gsm8k_regional.sh" "$@"
