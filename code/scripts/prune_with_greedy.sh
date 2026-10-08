#!/bin/bash
# Greedy pruning experiments. Run from the repository root.

# GSM8K train - Qwen teacher, Qwen pruner
python3 code/prune_with_greedy.py qwen-2.5-7b gsm8k train qwen-2.5-7b --best-cand-criteria reason-answer --keep-ratio 0.68 --num-splits 4 --split-idx 0

# Math500 train - Qwen teacher, Qwen pruner
python3 code/prune_with_greedy.py qwen-2.5-7b math500 train qwen-2.5-7b --best-cand-criteria reason-answer --keep-ratio 0.45 --num-splits 4 --split-idx 0 --verbose --filter-percentile 90

# MMLU test - Qwen teacher, Qwen pruner
python3 code/prune_with_greedy.py qwen-2.5-7b mmlu test qwen-2.5-7b --best-cand-criteria reason-answer --keep-ratio 0.45 --num-splits 4 --split-idx 0 --verbose --filter-percentile 90

# GSM8K train - Qwen teacher, Llama-2 pruner (cross-model setting)
python3 code/prune_with_greedy.py llama-2-7b gsm8k train qwen-2.5-7b --best-cand-criteria reason-answer --keep-ratio 0.45 --num-splits 4 --split-idx 0

# GSM8K train - Llama-3-8B teacher, Llama-3-8B pruner
python3 code/prune_with_greedy.py llama-3-8b gsm8k train llama-3-8b --best-cand-criteria reason-answer --keep-ratio 0.45 --num-splits 4 --split-idx 0

# GSM8K train - Qwen teacher, Qwen pruner (ablation: answer-only pruning criterion)
python3 code/prune_with_greedy.py qwen-2.5-7b gsm8k train qwen-2.5-7b --best-cand-criteria answer --keep-ratio 0.45 --num-splits 4 --split-idx 0
