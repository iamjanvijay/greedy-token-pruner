import jsonlines
import sys
import argparse
import os

BASE_DIR = os.environ.get("GREEDY_PRUNER_DATA_DIR", os.path.join(os.getcwd(), "data"))

# Suppress vLLM progress bars if not verbose
if '--verbose' not in sys.argv:
    os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
    os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'

from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer
from termcolor import colored
import time
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt
import numpy as np

def plot_token_length_histograms(data, pruner_tokenizer, output_path):
    percentile_values = [25, 50, 75, 90, 95, 96, 97, 98, 99, 100]   

    for item in data:
        item['user_prompt_num_tokens'] = len(pruner_tokenizer.encode(item['user_prompt'], add_special_tokens=False))
        item['response_reason_num_tokens'] = len(pruner_tokenizer.encode(item['response_reason'], add_special_tokens=False))
        item['response_answer_num_tokens'] = len(pruner_tokenizer.encode(item['response_answer'], add_special_tokens=False))
    
    user_prompt_num_tokens = [item['user_prompt_num_tokens'] for item in data]
    response_reason_num_tokens = [item['response_reason_num_tokens'] for item in data]
    response_answer_num_tokens = [item['response_answer_num_tokens'] for item in data]
    
    def get_legend_labels(lengths, percentiles):
        sorted_lengths = np.sort(lengths)
        return [f"{p}th: {int(np.round(pv))} (≤: {np.sum(sorted_lengths <= pv)})" 
                for p, pv in zip(percentile_values, percentiles)]

    plt.figure(figsize=(15, 5))
    
    for idx, (lengths, title, color) in enumerate([
        (user_prompt_num_tokens, 'Question Lengths', 'blue'),
        (response_reason_num_tokens, 'Reason Lengths', 'green'),
        (response_answer_num_tokens, 'Answer Lengths', 'orange')
    ], 1):
        plt.subplot(1, 3, idx)
        plt.hist(lengths, bins=30, color=color, alpha=0.7)
        plt.title(title)
        plt.xlabel('Number of tokens')
        plt.ylabel('Frequency')
        percentiles = np.percentile(lengths, percentile_values)
        for perc in percentiles:
            plt.axvline(perc, color='red', linestyle='dashed', linewidth=1)
        plt.legend(get_legend_labels(lengths, percentiles), title="Percentiles (Tokens, ≤Count)", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    
    r_percentiles = np.percentile(response_reason_num_tokens, percentile_values)
    a_percentiles = np.percentile(response_answer_num_tokens, percentile_values)
    u_percentiles = np.percentile(user_prompt_num_tokens, percentile_values)

    return {perc: float(perc_val) for perc, perc_val in zip(percentile_values, r_percentiles)}, {perc: float(perc_val) for perc, perc_val in zip(percentile_values, a_percentiles)}, {perc: float(perc_val) for perc, perc_val in zip(percentile_values, u_percentiles)}

def get_logprobs_vllm(llm_model, list_of_prompt_token_ids, args):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    llm_input_dicts = [{"prompt_token_ids": et} for et in list_of_prompt_token_ids]
    list_of_sampling_params = [sampling_params for _ in range(len(llm_input_dicts))]
    outputs = llm_model.generate(llm_input_dicts, sampling_params=list_of_sampling_params, use_tqdm=args.verbose)

    all_token_logprobs = []
    all_tokens = []
    for idx, output in enumerate(outputs):
        prompt_logprob = output.prompt_logprobs
        assert len(list_of_prompt_token_ids[idx]) == len(prompt_logprob), "Length of list_of_prompt_token_ids and prompt_logprob must be equal."
        
        tokens, token_logprobs = [], []
        for token_idx, (prompt_token_id, token_logprob_dict) in enumerate(zip(list_of_prompt_token_ids[idx], prompt_logprob)):
            if token_idx == 0:
                token, token_logprob = "", 0.0  # First token has no logprob in vLLM
            else:
                token, token_logprob = token_logprob_dict[prompt_token_id].decoded_token, token_logprob_dict[prompt_token_id].logprob
            token_logprobs.append(token_logprob)
            tokens.append(token)
        
        all_token_logprobs.append(token_logprobs)
        all_tokens.append(tokens)
    
    return {"token_logprobs": all_token_logprobs, "tokens": all_tokens}

def render_with_mask(tokenizer, token_ids, keep_mask):
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    assert len(tokens) == len(keep_mask), "Length of tokens and keep_mask must be equal."

    rendered = []
    for token, keep in zip(tokens, keep_mask):
        if keep:
            rendered.append(colored(token, "green"))
        else:
            rendered.append(colored(token, "red"))
    print("|".join(rendered))

def perplexity_from_logprobs(log_probs):
    """
    perplexity computation: exp(-mean(log_probs))
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.size == 0:
        return 1e32

    mean_log = log_probs.mean()  # this is safe; sum won't overflow for typical sequence lengths
    return min(float(np.exp(-mean_log)), 1e32)

def get_surprisal_pruned_question_reason_answer(pruner_tokenizer, pruner_marker_token_to_id, llm_model, item, verbose=False, args=None):
    pre_reason_token_ids, reason_token_ids, post_reason_token_ids = [], [], []
    is_pre_reason, is_reason, is_post_reason = True, False, False
    for token_id in item["chat_template_token_ids"]:
        if token_id == pruner_marker_token_to_id["<reason>"]:
            is_pre_reason, is_reason = False, True
            continue
        if token_id == pruner_marker_token_to_id["</reason>"]:
            is_reason, is_post_reason = False, True
            continue
        assert sum([is_pre_reason, is_reason, is_post_reason]) == 1, "Exactly one of is_pre_reason, is_reason, is_post_reason must be True"
        if is_pre_reason:
            pre_reason_token_ids.append(token_id)
        elif is_reason:
            reason_token_ids.append(token_id)
        else:
            post_reason_token_ids.append(token_id)
    post_reason_num_tokens = len(post_reason_token_ids)
    original_reason_num_tokens = len(reason_token_ids)

    # Initialize or load existing pruning data
    if "surprisal_prunings" not in item:
        item["surprisal_prunings"] = {}
        item["surprisal_prunings"]["reason_token_ids"] = reason_token_ids
        item["surprisal_prunings"]["del_pos_rank"] = []
        item["surprisal_prunings"]["reason_answer_perplexity"] = []
        item["surprisal_prunings"]["answer_perplexity"] = []
        item["surprisal_prunings"]["reason_perplexity"] = []

    keep_reason_token_ids = [1 for _ in range(len(reason_token_ids))] # 1 means keep, 0 means delete
    
    if verbose:
        render_with_mask(pruner_tokenizer, reason_token_ids, keep_reason_token_ids)

    all_logprobs = get_logprobs_vllm(llm_model, [pre_reason_token_ids + reason_token_ids + post_reason_token_ids], args)
    reason_logprobs = all_logprobs["token_logprobs"][0][-original_reason_num_tokens-post_reason_num_tokens:-post_reason_num_tokens]
    item["surprisal_prunings"]["original_reason_perplexity"] = perplexity_from_logprobs(reason_logprobs)
    reason_answer_logprobs = all_logprobs["token_logprobs"][0][-original_reason_num_tokens-post_reason_num_tokens:]
    item["surprisal_prunings"]["original_reason_answer_perplexity"] = perplexity_from_logprobs(reason_answer_logprobs)
    answer_logprobs = all_logprobs["token_logprobs"][0][-post_reason_num_tokens:]
    item["surprisal_prunings"]["original_answer_perplexity"] = perplexity_from_logprobs(answer_logprobs)

    # highest log-prob first, means token with least importance appears first, means tokens appearing first can be deleted
    importance_order = sorted(zip(range(len(reason_logprobs)), reason_logprobs), key=lambda x: x[1], reverse=True) 

    for del_idx, reason_logprob in importance_order:
        best_del_idx, best_reason_answer_perplexity = -1, float('inf')
        keep_reason_token_ids[del_idx] = 0 # delete this token
        pruned_reason_token_ids = [reason_token_id for keep_reason_token_id, reason_token_id in zip(keep_reason_token_ids, reason_token_ids) if keep_reason_token_id == 1]
        list_of_prompt_token_ids = [deepcopy(pre_reason_token_ids + pruned_reason_token_ids + post_reason_token_ids)]

        all_logprobs = get_logprobs_vllm(llm_model, list_of_prompt_token_ids, args)
        token_logprobs, tokens = all_logprobs["token_logprobs"][0], all_logprobs["tokens"][0]
        
        keep_reason_num_tokens = sum(keep_reason_token_ids)
        reason_answer_perplexity = perplexity_from_logprobs(token_logprobs[-keep_reason_num_tokens-post_reason_num_tokens:])
        answer_perplexity = perplexity_from_logprobs(token_logprobs[-post_reason_num_tokens:])
        reason_perplexity = perplexity_from_logprobs(token_logprobs[-keep_reason_num_tokens-post_reason_num_tokens:-post_reason_num_tokens])

        best_del_idx = del_idx
        item["surprisal_prunings"]["del_pos_rank"].append(best_del_idx)
        item["surprisal_prunings"]["reason_answer_perplexity"].append(reason_answer_perplexity)
        item["surprisal_prunings"]["answer_perplexity"].append(answer_perplexity)
        item["surprisal_prunings"]["reason_perplexity"].append(reason_perplexity)

        if verbose:
            render_with_mask(pruner_tokenizer, reason_token_ids, keep_reason_token_ids)

    return item

def build_tokenizer_with_markers(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = [
        "<reason>", "</reason>"
    ]
    special_tokens_dict = {
        "additional_special_tokens": special_tokens
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    # Build a mapping from marker token string to its token ID
    marker_token_to_id = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens}
    return tokenizer, marker_token_to_id

def load_teacher_generated_data(filepath):
    """
    Load teacher-generated reasoning data from JSONL file.
    
    Args:
        filepath: path to the JSONL file
    
    Returns:
        list of data dictionaries
    """
    dataset = []
    for data_dict in tqdm(jsonlines.open(filepath), desc="Loading data"):
        assert "system_prompt" in data_dict or "model_name" in data_dict, "Invalid data format"
        
        # Skip metadata lines (contain model_name instead of actual data)
        if "model_name" in data_dict or \
            data_dict["is_correct"] == False or len(data_dict["response_reason"]) < 100:
            continue

        if "split_type" in data_dict and data_dict["split_type"] != "train":
            continue
            
        dataset.append(data_dict)
    return dataset

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prune reasoning chains from teacher model using surprisal token deletion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prune-with-surprisal.py llama-3-8b gsm8k train qwen-2.5-7b
  python prune-with-surprisal.py mistral-7b mmlu validation llama-3-8b --num-splits 4 --split-idx 0
  python prune-with-surprisal.py llama-2-7b gsm8k train qwen-2.5-7b --best-cand-criteria answer
        """
    )
    
    parser.add_argument(
        "pruner_model",
        type=str,
        choices=["llama-2-7b", "mistral-7b", "llama-3-8b", "qwen-2.5-7b"],
        help="Student model to train (target model for pruned data)"
    )
    
    parser.add_argument(
        "dataset",
        type=str,
        choices=["gsm8k", "mmlu", "math500"],
        help="Dataset to process"
    )
    
    parser.add_argument(
        "split",
        type=str,
        choices=["train", "test", "validation"],
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "teacher_model",
        type=str,
        choices=["llama-2-7b", "mistral-7b", "llama-3-8b", "qwen-2.5-7b"],
        help="Teacher model that generated the reasoning chains"
    )
    
    parser.add_argument(
        "--num-splits",
        type=int,
        default=1,
        help="Number of splits to divide the dataset into for parallel processing (default: 1)"
    )
    
    parser.add_argument(
        "--split-idx",
        type=int,
        default=0,
        help="Index of the split to process (0-indexed, must be < num-splits) (default: 0)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging of each deletion step (default: False)"
    )
    
    parser.add_argument(
        "--filter-percentile",
        type=int,
        default=99,
        choices=[25, 50, 75, 90, 95, 96, 97, 98, 99, 100],
        help="Percentile threshold for filtering examples by token length (default: 95)"
    )
    
    return parser.parse_args()

# Model name mapping
MODEL_NAME_MAP = {
    "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct"
}

# Dataset name mapping
DATASET_NAME_MAP = {
    "mmlu": "TIGER-Lab/MMLU-Pro",
    "gsm8k": "openai/gsm8k",
    "math500": "simplescaling/openaimath"
}

if __name__ == "__main__":
    args = parse_args()

    assert args.teacher_model == args.pruner_model, "Teacher model and pruner model must be the same"

    pruner_model = MODEL_NAME_MAP[args.pruner_model]
    dataset = DATASET_NAME_MAP[args.dataset]
    dataset_split = args.split
    teacher_model = MODEL_NAME_MAP[args.teacher_model]

    pruner_tokenizer, pruner_marker_token_to_id = build_tokenizer_with_markers(pruner_model)

    source_filepath = os.path.join(
        BASE_DIR, "teacher-generated",
        f"{dataset.replace('/', '-')}_{dataset_split}_{teacher_model.replace('/', '-')}.jsonl"
    )
    dest_path = f"{BASE_DIR}/surprisal-pruned"

    save_folder = os.path.join(dest_path, teacher_model.replace('/', '-'), dataset.replace('/', '-'), dataset_split, pruner_model.replace('/', '-'))
    os.makedirs(save_folder, exist_ok=True)

    teacher_dataset = load_teacher_generated_data(source_filepath)

    # filter in the examples where reason length is greater than the percentile
    filter_percentile = args.filter_percentile
    reason_percentiles, answer_percentiles, user_prompt_percentiles = plot_token_length_histograms(teacher_dataset, pruner_tokenizer, os.path.join(save_folder, f"token_length_histograms.png"))   
    filtered_teacher_dataset = []
    for item in teacher_dataset:
        if item['response_reason_num_tokens'] <= reason_percentiles[filter_percentile] and \
            item['response_answer_num_tokens'] <= answer_percentiles[filter_percentile] and \
                item['user_prompt_num_tokens'] <= user_prompt_percentiles[filter_percentile]:
            filtered_teacher_dataset.append(item)
    teacher_dataset = filtered_teacher_dataset
    print(f">>> Filtered {len(teacher_dataset)} examples where reason length is smaller than the {filter_percentile}th percentile")

    # Create num_splits splits of teacher_dataset for parallel processing
    num_splits, split_idx = args.num_splits, args.split_idx
    
    if split_idx >= num_splits:
        raise ValueError(f"split_idx ({split_idx}) must be less than num_splits ({num_splits})")

    split_size = (len(teacher_dataset) + num_splits - 1) // num_splits  # ceiling division
    teacher_dataset = [teacher_dataset[i * split_size:(i + 1) * split_size] for i in range(num_splits)][split_idx]
    print(f">>> Processing split {split_idx}/{num_splits} with {len(teacher_dataset)} examples...")

    for item in teacher_dataset:
        messages = [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["user_prompt"]},
            {"role": "assistant", "content": "<reason>" + item["response_reason"] + "</reason>" + item["response_answer"]}
        ]
        item["chat_template"] = pruner_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        item["chat_template_token_ids"] = pruner_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    max_chat_template_token_ids_length = max(len(item["chat_template_token_ids"]) for item in teacher_dataset)
    print(f">>> Max chat template token ids length: {max_chat_template_token_ids_length}")

    # Initialize vLLM model for direct Python API usage (MUCH faster than HTTP)
    print(">>> Initializing vLLM model...")
    llm_model = LLM(
        model=pruner_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_chat_template_token_ids_length+1,
        disable_log_stats=not args.verbose,
    )

    print("vLLM model initialized!")

    save_filepath = os.path.join(save_folder, f"surprisal_pruned_split_{split_idx}.jsonl")

    # Load existing pruning data to check progress
    existing_pruning_data = {}
    if os.path.exists(save_filepath):
        with jsonlines.open(save_filepath) as reader:
            for item in reader:
                if "user_prompt" in item and "surprisal_prunings" in item:
                    existing_pruning_data[item["user_prompt"]] = item

    # Print configuration before starting pruning
    print(f"\n{'='*80}")
    print("surprisal PRUNING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Teacher Model:       {teacher_model}")
    print(f"Pruner Model:        {pruner_model}")
    print(f"Dataset:             {dataset}")
    print(f"Split:               {dataset_split}")
    print(f"Verbose Logging:     {args.verbose}")
    print(f"Split Index:         {split_idx}/{num_splits}")
    print(f"Total Examples:      {len(teacher_dataset)}")
    print(f"Existing Pruned:     {len(existing_pruning_data)}")
    print(f"Save Path:           {save_filepath}")
    print(f"Max Model Length:    {max_chat_template_token_ids_length}")
    print(f"{'='*80}\n")

    # Surprisal pruning: delete reasoning tokens in order of ascending surprisal
    # (equivalently, descending log-prob under the teacher model).
    print("Starting surprisal pruning...")
    total_time, total_examples = 0, 0
    skipped_examples = 0
    
    for idx, item in enumerate(teacher_dataset):
        user_prompt = item["user_prompt"]

        time_start = time.perf_counter()
        item = get_surprisal_pruned_question_reason_answer(
            pruner_tokenizer, pruner_marker_token_to_id, llm_model, item, 
            verbose=args.verbose, args=args
        )
        time_end = time.perf_counter()
        time_taken = time_end - time_start

        total_time += time_taken
        total_examples += 1

        # Save current keep_ratio in the item
        item["total_reason_tokens"] = len(item["surprisal_prunings"]["reason_token_ids"])
        item["tokens_deleted"] = len(item["surprisal_prunings"]["del_pos_rank"])
        item["tokens_remaining"] = item["total_reason_tokens"] - item["tokens_deleted"]

        # Calculate progress statistics
        avg_time_per_example = total_time / total_examples
        examples_remaining = len(teacher_dataset) - (idx + 1)
        estimated_time_remaining = avg_time_per_example * examples_remaining
        
        print(f">>> Processed {idx + 1}/{len(teacher_dataset)} examples | Skipped: {skipped_examples} | "
              f"Avg time: {avg_time_per_example:.2f}s | Remaining: {examples_remaining} | "
              f"Est. time left: {estimated_time_remaining/60:.1f}m ({estimated_time_remaining:.0f}s)")
        
        # New item, append
        with jsonlines.open(save_filepath, "a") as writer:
            writer.write(item)
