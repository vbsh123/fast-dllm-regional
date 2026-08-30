#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:-regional}
LIMIT=${LIMIT:-50}
MODEL=${MODEL:-Dream-org/Dream-v0-Base-7B}
OUTPUT_ROOT=${OUTPUT_ROOT:-evals_results/regional_pilot}
LENGTH=${LENGTH:-256}

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
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=${LENGTH},add_bos_token=true,alg=regional_balanced,use_cache=false,regional_region_size=32,regional_local_steps=32,regional_max_progress_gap=4,regional_deferral_threshold=0.4,regional_deferral_until_revealed=2,regional_max_global_deferrals=4,regional_tail_guard=true,regional_commit_policy=entropy"
    ;;
  regional_filter)
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=${LENGTH},add_bos_token=true,alg=regional_balanced,use_cache=false,regional_region_size=32,regional_local_steps=32,regional_max_progress_gap=4,regional_deferral_threshold=0.4,regional_deferral_until_revealed=2,regional_max_global_deferrals=4,regional_stop_mode=filter,regional_stop_filter_threshold=0.7,regional_commit_policy=entropy"
    ;;
  regional_defer)
    MODEL_ARGS="pretrained=${MODEL},max_new_tokens=${LENGTH},diffusion_steps=${LENGTH},add_bos_token=true,alg=regional_balanced,use_cache=false,regional_region_size=32,regional_local_steps=32,regional_max_progress_gap=4,regional_deferral_threshold=0.4,regional_deferral_until_revealed=2,regional_max_global_deferrals=4,regional_stop_mode=defer,regional_stop_filter_threshold=0.7,regional_commit_policy=entropy"
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
  --output_path "${OUTPUT_ROOT}/${METHOD}_${LIMIT}" \
  --log_samples
