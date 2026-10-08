#!/bin/bash
# Generate teacher reasoning data on the test splits of the three held-out datasets.
# Run from the repository root. Writes JSONL files to
# $GREEDY_PRUNER_DATA_DIR/latest-data/teacher-generated/.

set -e

base_data_dir="${GREEDY_PRUNER_DATA_DIR:-$(pwd)/data}/latest-data/teacher-generated"

# Each entry: dataset_name|split|model
declare -a DATASET_MODEL_SPLIT_LIST=(
    "iamjanvijay/gsm8k|test|meta-llama/Llama-3.1-8B-Instruct"
    "iamjanvijay/openaimath|test|meta-llama/Llama-3.1-8B-Instruct"
    "iamjanvijay/MMLU-Pro|test|meta-llama/Llama-3.1-8B-Instruct"
    "iamjanvijay/gsm8k|test|Qwen/Qwen2.5-7B-Instruct"
    "iamjanvijay/openaimath|test|Qwen/Qwen2.5-7B-Instruct"
    "iamjanvijay/MMLU-Pro|test|Qwen/Qwen2.5-7B-Instruct"
)

for entry in "${DATASET_MODEL_SPLIT_LIST[@]}"; do
    IFS='|' read -r DATASET_NAME DATASET_SPLIT MODEL <<< "$entry"
    echo "Running: dataset=${DATASET_NAME}, split=${DATASET_SPLIT}, model=${MODEL}"
    python3 code/gen_data_from_teacher.py \
        --model "${MODEL}" \
        --dataset_name "${DATASET_NAME}" \
        --dataset_split "${DATASET_SPLIT}" \
        --output_dir "$base_data_dir" \
        --temperature 0.0 \
        --n 1
done

echo "All teacher data generation tasks completed!"
