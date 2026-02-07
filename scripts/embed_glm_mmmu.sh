#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_paths=(
  "zai-org/GLM-4.1V-9B-Thinking"
)

data_path="Data/Questions/MMMU_Validation.jsonl"
images_root="Data/Images/MMMU_Val"
save_path="Data/Representation/MMMU_Validation"

input_mode="mixed"
batch_size=2

for model_path in "${model_paths[@]}"; do
  echo "Processing model: ${model_path}"
  python embed_glm_mmmu.py \
    --reasoning True \
    --model_path "${model_path}" \
    --data_path "${data_path}" \
    --images_root "${images_root}" \
    --save_path "${save_path}" \
    --input_mode "${input_mode}" \
    --batch_size "${batch_size}"
done
