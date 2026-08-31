#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:-regional}
LIMIT=${LIMIT:-50}
MODEL=${MODEL:-Dream-org/Dream-v0-Base-7B}
OUTPUT_ROOT=${OUTPUT_ROOT:-evals_results/regional_pilot}
LENGTH=${LENGTH:-256}
REGION_SIZE=${REGION_SIZE:-32}
LOCAL_STEPS=${LOCAL_STEPS:-32}
MAX_PROGRESS_GAP=${MAX_PROGRESS_GAP:-4}
DEFERRAL_THRESHOLD=${DEFERRAL_THRESHOLD:-0.4}
DEFERRAL_UNTIL_REVEALED=${DEFERRAL_UNTIL_REVEALED:-2}
MAX_GLOBAL_DEFERRALS=${MAX_GLOBAL_DEFERRALS:-4}
STOP_FILTER_THRESHOLD=${STOP_FILTER_THRESHOLD:-0.7}
RUN_TAG=${RUN_TAG:-}

REGIONAL_ARGS="regional_region_size=${REGION_SIZE},regional_local_steps=${LOCAL_STEPS},regional_max_progress_gap=${MAX_PROGRESS_GAP},regional_deferral_threshold=${DEFERRAL_THRESHOLD},regional_deferral_until_revealed=${DEFERRAL_UNTIL_REVEALED},regional_max_global_deferrals=${MAX_GLOBAL_DEFERRALS},regional_commit_policy=entropy"

case "$METHOD" in
  vanilla)
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=${LENGTH},add_bos_token=true,alg=entropy,use_cache=false"
    ;;
  fast)
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=8,add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=false"
    ;;
  fast_cache)
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=8,add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true"
    ;;
  regional)
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=${LENGTH},add_bos_token=true,alg=regional_balanced,use_cache=false,${REGIONAL_ARGS},regional_tail_guard=true"
    ;;
  regional_filter)
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=${LENGTH},add_bos_token=true,alg=regional_balanced,use_cache=false,${REGIONAL_ARGS},regional_stop_mode=filter,regional_stop_filter_threshold=${STOP_FILTER_THRESHOLD}"
    ;;
  regional_defer)
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=${LENGTH},add_bos_token=true,alg=regional_balanced,use_cache=false,${REGIONAL_ARGS},regional_stop_mode=defer,regional_stop_filter_threshold=${STOP_FILTER_THRESHOLD}"
    ;;
  *)
    echo "usage: $0 {vanilla|fast|fast_cache|regional|regional_filter|regional_defer}" >&2
    exit 2
    ;;
esac

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

accelerate launch eval.py \
  --model dream \
  --model_args "$MODEL_ARGS" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 1 \
  --limit "$LIMIT" \
  --seed 1234 \
  --output_path "${OUTPUT_ROOT}/${METHOD}_${LIMIT}${RUN_TAG:+_${RUN_TAG}}" \
  --log_samples
