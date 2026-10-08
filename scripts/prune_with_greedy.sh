#!/bin/bash
# Greedy pruning experiments

# python3 pruning/prune_with_greedy.py pruner_model dataset split teacher_model --best-cand-criteria reason-answer --keep-ratio 0.45

# GSM8K train - qwen teacher, qwen pruner | qwen with dataset 1 | 500 seconds per example. roughly 7500 examples. 
python3 pruning/prune_with_greedy.py qwen-2.5-7b gsm8k train qwen-2.5-7b --best-cand-criteria reason-answer --keep-ratio 0.68 --num-splits 4 --split-idx 0

# Math500 train - qwen teacher, qwen pruner | qwen with dataset 2 | 1000 seconds per example. roughly 9500 examples. 
python3 pruning/prune_with_greedy.py qwen-2.5-7b math500 train qwen-2.5-7b --best-cand-criteria reason-answer --keep-ratio 0.45 --num-splits 4 --split-idx 0 --verbose --filter-percentile 90

# MMLU test - qwen teacher, Qwen pruner | qwen with dataset 3 | 1000 seconds per example. roughly 9500 examples. 
python3 pruning/prune_with_greedy.py qwen-2.5-7b mmlu test qwen-2.5-7b --best-cand-criteria reason-answer --keep-ratio 0.45 --num-splits 4 --split-idx 0 --verbose --filter-percentile 90

# GSM8K train - qwen teacher, Llama-2 pruner | student is the pruner model | 500 seconds per example. roughly 7500 examples. 
python3 pruning/prune_with_greedy.py llama-2-7b gsm8k train qwen-2.5-7b --best-cand-criteria reason-answer --keep-ratio 0.45 --num-splits 4 --split-idx 0

# GSM8K train - Llama-3-8B, Llama-3-8B pruner | additionl model with one dataset for pruning | 500 seconds per example. roughly 7500 example | won't run this as I already have dataset from this?
python3 pruning/prune_with_greedy.py llama-3-8b gsm8k train llama-3-8b --best-cand-criteria reason-answer --keep-ratio 0.45 --num-splits 4 --split-idx 0

# GSM8K train - Qwen teacher, Qwen pruner | pruning with answer as pruning criterion | 500 seconds per example. roughly 7500 examples. 
python3 pruning/prune_with_greedy.py qwen-2.5-7b gsm8k train qwen-2.5-7b --best-cand-criteria answer --keep-ratio 0.45 --num-splits 4 --split-idx 0



