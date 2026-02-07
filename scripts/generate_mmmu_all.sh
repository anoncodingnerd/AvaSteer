#!/bin/bash

model_path="zai-org/GLM-4.1V-9B-Thinking"
# model_path="Qwen/Qwen3-VL-2B-Thinking"
vote_num=1
tensor_parallel_size=1
steering_strength=0.0
steering_vector_type="all"
adaptive_steering="True"
adaptive_reduce="last"
adaptive_eps="1e-6"
# calibration_vector_path="./Data/Representation/MathV_split/Qwen3-VL-2B-Thinking/calibration_vectors.npy"
# calibration_vector_path="./Data/Representation/MMMU_Validation/Qwen3-VL-2B-Thinking/calibration_vectors.npy"
calibration_vector_path="./Data/Representation/MathV_split/GLM-4.1V-9B-Thinking/calibration_vectors.npy"
batch_size=64
max_tokens=40960

# for adaptive_alpha in {0.35,0.5,0.65,0.8,0.95,1.1}; do
for adaptive_alpha in {1.0,1.5,2.0,2.5}; do
    echo "Waiting for previous process to fully exit..."
    sleep 10  # Wait for 10 seconds to ensure that the previous process has completely exited.
    echo "Running adaptive_alpha=${adaptive_alpha}"
    bash scripts/generate_and_evaluate_mmmu.sh \
        "$model_path" \
        "$vote_num" \
        "$tensor_parallel_size" \
        "$steering_strength" \
        "$steering_vector_type" \
        "$adaptive_steering" \
        "$adaptive_reduce" \
        "$adaptive_eps" \
        "$adaptive_alpha" \
        "$calibration_vector_path" \
        "$batch_size" \
        "$max_tokens"
done
