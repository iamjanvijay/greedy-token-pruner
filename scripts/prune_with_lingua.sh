#!/bin/bash
# Usage: python prune-with-lingua.py <student_model> <dataset> <split> <teacher_model> [--keep-fractions ...]

# # target model: llama-2-7b

# # (1) llama-3-8b + gsm8k; # setting with additional pruning ratios for comparison
# python3 baselines/prune_with_lingua.py llama-2-7b gsm8k train llama-3-8b --keep-fractions 0.30 0.50 0.70 0.80 0.90
# # (2) qwen-2.5-7b + gsm8k;
# python3 baselines/prune_with_lingua.py llama-2-7b gsm8k train qwen-2.5-7b --keep-fractions 0.70 0.80 0.90
# # (3) qwen-2.5-7b + math500;
# python3 baselines/prune_with_lingua.py llama-2-7b math500 train qwen-2.5-7b --keep-fractions 0.70 0.80 0.90
# # (4) llama-3-8b + mmlu;
# python3 baselines/prune_with_lingua.py llama-2-7b mmlu test llama-3-8b --keep-fractions 0.70 0.80 0.90

# # target model: mistral-7b

# # (1) llama-3-8b + gsm8k; # setting with additional pruning ratios for comparison
# python3 baselines/prune_with_lingua.py mistral-7b gsm8k train llama-3-8b --keep-fractions 0.30 0.50 0.70 0.80 0.90
# # (2) qwen-2.5-7b + gsm8k;
# python3 baselines/prune_with_lingua.py mistral-7b gsm8k train qwen-2.5-7b --keep-fractions 0.70 0.80 0.90
# # (3) qwen-2.5-7b + math500;
# python3 baselines/prune_with_lingua.py mistral-7b math500 train qwen-2.5-7b --keep-fractions 0.70 0.80 0.90
# # (4) llama-3-8b + mmlu;
# python3 baselines/prune_with_lingua.py mistral-7b mmlu test llama-3-8b --keep-fractions 0.70 0.80 0.90

python3 baselines/prune_with_lingua.py llama-3-8b gsm8k train llama-3-8b --keep-fractions 0.70 0.80 0.90