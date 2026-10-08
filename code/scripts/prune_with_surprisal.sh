#!/bin/bash
# Surprisal pruning experiments. Run from the repository root.

# GSM8K train - Qwen teacher, Qwen pruner
export CUDA_VISIBLE_DEVICES=0
for split_idx in {0..99}; do
    python3 code/prune_with_surprisal.py qwen-2.5-7b gsm8k train qwen-2.5-7b --num-splits 100 --split-idx $split_idx
done

# GSM8K train - Llama-3-8B teacher, Llama-3-8B pruner
export CUDA_VISIBLE_DEVICES=1
for split_idx in {0..99}; do
    python3 code/prune_with_surprisal.py llama-3-8b gsm8k train llama-3-8b --num-splits 100 --split-idx $split_idx
done

# Math500 train - Qwen teacher, Qwen pruner
export CUDA_VISIBLE_DEVICES=2
for split_idx in {0..199}; do
    python3 code/prune_with_surprisal.py qwen-2.5-7b math500 train qwen-2.5-7b --num-splits 200 --split-idx $split_idx
done

# MMLU test - Qwen teacher, Qwen pruner
export CUDA_VISIBLE_DEVICES=3
for split_idx in {0..199}; do
    python3 code/prune_with_surprisal.py qwen-2.5-7b mmlu test qwen-2.5-7b --num-splits 200 --split-idx $split_idx
done
