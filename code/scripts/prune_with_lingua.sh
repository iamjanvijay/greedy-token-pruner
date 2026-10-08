#!/bin/bash
# LLMLingua-2 baseline. Run from the repository root.
# Usage: python baselines/prune_with_lingua.py <student_model> <dataset> <split> <teacher_model> [--keep-fractions ...]

python3 code/prune_with_lingua.py llama-3-8b gsm8k train llama-3-8b --keep-fractions 0.70 0.80 0.90
