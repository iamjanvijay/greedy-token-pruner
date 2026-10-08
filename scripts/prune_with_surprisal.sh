#!/bin/bash
# surprisal pruning experiments

# GSM8K train - qwen teacher, qwen pruner | qwen with dataset 1 | 500 seconds per example. roughly 7500 examples. 
export CUDA_VISIBLE_DEVICES=0
for split_idx in {0..99}; do
    python3 baselines/prune_with_surprisal.py qwen-2.5-7b gsm8k train qwen-2.5-7b --num-splits 100 --split-idx $split_idx
done

# GSM8K train - Llama-3-8B, Llama-3-8B pruner | additionl model with one dataset for pruning | 500 seconds per example. roughly 7500 example | won't run this as I already have dataset from this?
export CUDA_VISIBLE_DEVICES=1
for split_idx in {0..99}; do
    python3 baselines/prune_with_surprisal.py llama-3-8b gsm8k train llama-3-8b --num-splits 100 --split-idx $split_idx
done

# Math500 train - qwen teacher, qwen pruner | qwen with dataset 2 | 1000 seconds per example. roughly 9500 examples.
export CUDA_VISIBLE_DEVICES=2
for split_idx in {0..199}; do
    python3 baselines/prune_with_surprisal.py qwen-2.5-7b math500 train qwen-2.5-7b --num-splits 200 --split-idx $split_idx
done

# MMLU test - qwen teacher, Qwen pruner | qwen with dataset 3 | 1000 seconds per example. roughly 9500 examples. 
export CUDA_VISIBLE_DEVICES=3
for split_idx in {0..199}; do
    python3 baselines/prune_with_surprisal.py qwen-2.5-7b mmlu test qwen-2.5-7b --num-splits 200 --split-idx $split_idx
done



