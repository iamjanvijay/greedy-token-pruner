#!/bin/bash
# Script to generate teacher-generated reasoning data from various models and datasets
# Usage: bash gen_data_from_teacher.sh

set -e  # Exit on error

base_data_dir="${GREEDY_PRUNER_DATA_DIR:-$(pwd)/data}/latest-data/teacher-generated"

# Datasets, splits, and models to loop over
declare -a DATASET_MODEL_SPLIT_LIST=(
    # # Format: dataset_name|split|model n=10 and temperature=0.7
    # "openai/gsm8k|train|meta-llama/Llama-3.1-8B-Instruct"
    # "openai/gsm8k|test|meta-llama/Llama-3.1-8B-Instruct"
    # "openai/gsm8k|train|Qwen/Qwen2.5-7B-Instruct"
    # "openai/gsm8k|test|Qwen/Qwen2.5-7B-Instruct"
    # "simplescaling/openaimath|train|meta-llama/Llama-3.1-8B-Instruct"
    # "simplescaling/openaimath|test|meta-llama/Llama-3.1-8B-Instruct"
    # "simplescaling/openaimath|train|Qwen/Qwen2.5-7B-Instruct"
    # "simplescaling/openaimath|test|Qwen/Qwen2.5-7B-Instruct"
    # "TIGER-Lab/MMLU-Pro|test|meta-llama/Llama-3.1-8B-Instruct"
    # "TIGER-Lab/MMLU-Pro|test|Qwen/Qwen2.5-7B-Instruct"

    # To get zero shot performance of the models on the datasets.
    # # For Llama-2-7b-chat-hf.
    # "iamjanvijay/gsm8k|test|meta-llama/Llama-2-7b-chat-hf"
    # "iamjanvijay/openaimath|test|meta-llama/Llama-2-7b-chat-hf"
    # "iamjanvijay/MMLU-Pro|test|meta-llama/Llama-2-7b-chat-hf"
    # # For Qwen-2-7B-Instruct.
    # "iamjanvijay/gsm8k|test|Qwen/Qwen2-7B-Instruct"
    # "iamjanvijay/openaimath|test|Qwen/Qwen2-7B-Instruct"
    # "iamjanvijay/MMLU-Pro|test|Qwen/Qwen2-7B-Instruct"
    # # For Mistral-7B-Instruct-v0.3.
    # "iamjanvijay/gsm8k|test|mistralai/Mistral-7B-Instruct-v0.3"
    # "iamjanvijay/openaimath|test|mistralai/Mistral-7B-Instruct-v0.3"
    # "iamjanvijay/MMLU-Pro|test|mistralai/Mistral-7B-Instruct-v0.3"
    # # For mistralai/Mathstral-7b-v0.1
    # "iamjanvijay/gsm8k|test|mistralai/Mathstral-7b-v0.1"
    # "iamjanvijay/openaimath|test|mistralai/Mathstral-7b-v0.1"
    # "iamjanvijay/MMLU-Pro|test|mistralai/Mathstral-7b-v0.1"

    # For Llama-3.1-8B-Instruct.
    "iamjanvijay/gsm8k|test|meta-llama/Llama-3.1-8B-Instruct"
    "iamjanvijay/openaimath|test|meta-llama/Llama-3.1-8B-Instruct"
    "iamjanvijay/MMLU-Pro|test|meta-llama/Llama-3.1-8B-Instruct"
    # For Qwen-2.5-7B-Instruct.
    "iamjanvijay/gsm8k|test|Qwen/Qwen2.5-7B-Instruct"
    "iamjanvijay/openaimath|test|Qwen/Qwen2.5-7B-Instruct"
    "iamjanvijay/MMLU-Pro|test|Qwen/Qwen2.5-7B-Instruct"
)

for entry in "${DATASET_MODEL_SPLIT_LIST[@]}"; do
    IFS='|' read -r DATASET_NAME DATASET_SPLIT MODEL <<< "$entry"
    echo "Running: dataset=${DATASET_NAME}, split=${DATASET_SPLIT}, model=${MODEL}"
    python3 data_generation/gen_data_from_teacher.py \
        --model "${MODEL}" \
        --dataset_name "${DATASET_NAME}" \
        --dataset_split "${DATASET_SPLIT}" \
        --output_dir "$base_data_dir" \
        --temperature 0.0 \
        --n 1
done

echo "All teacher data generation tasks completed!"



