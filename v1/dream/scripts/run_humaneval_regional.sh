#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export TASK=humaneval
export NUM_FEWSHOT=${NUM_FEWSHOT:-0}
export ESCAPE_UNTIL=true
export CONFIRM_RUN_UNSAFE_CODE=true
export OUTPUT_ROOT=${OUTPUT_ROOT:-evals_results/regional_humaneval}

exec bash "${SCRIPT_DIR}/run_gsm8k_regional.sh" "$@"
