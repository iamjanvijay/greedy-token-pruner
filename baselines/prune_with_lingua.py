import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import jsonlines

from llmlingua import PromptCompressor
class LLMLingua:
    def __init__(self) -> None:
        self.llmlingua_path="microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        self._llm_lingua = None
        
    # Ensure Singleton instance
    def get_llm_lingua_instance(self):
            if not self._llm_lingua:
                self._llm_lingua = PromptCompressor(
                    model_name=self.llmlingua_path,
                    use_llmlingua2=True,  # Whether to use llmlingua-2

                )
                return self._llm_lingua
            else:
                return self._llm_lingua
            
def compress_reasoning_with_lingua(chunked_reasonings, lingua_compression_rate, llm_lingua):
    """
    Compress reasoning text chunks using LLMLingua-2.
    
    Args:
        chunked_reasonings: list of lists, each inner list contains text chunks from one reasoning chain
        lingua_compression_rate: float, compression rate parameter for LLM-Lingua
        llm_lingua: LLMLingua instance
    
    Returns:
        list of lists, compressed text chunks
    """
    compressed_chunks_list = []
    for reasoning_chunks in chunked_reasonings:
        compressed_chunks = []
        for chunk in reasoning_chunks:
            compressed = llm_lingua.get_llm_lingua_instance().compress_prompt(
                chunk, 
                rate=lingua_compression_rate,
                force_reserve_digit=False,
                drop_consecutive=True
            )
            compressed_chunks.append(compressed["compressed_prompt"])
        compressed_chunks_list.append(compressed_chunks)
    return compressed_chunks_list

def split_reasonings_into_chunks(reasoning_texts, max_chunk_tokens, lingua_tokenizer):
    """
    Split long reasoning texts into smaller chunks for LLMLingua processing.
    
    Args:
        reasoning_texts: list of strings, each is a reasoning chain
        max_chunk_tokens: int, maximum tokens per chunk
        lingua_tokenizer: tokenizer for chunking
    
    Returns:
        list of lists, each inner list contains text chunks from one reasoning
    """
    chunked_reasonings = []
    for reasoning_text in reasoning_texts:
        tokens = lingua_tokenizer.encode(reasoning_text, add_special_tokens=False, verbose=False)
        chunks = []
        start_idx = 0
        while start_idx < len(tokens):
            chunk_text = lingua_tokenizer.decode(tokens[start_idx:start_idx + max_chunk_tokens])
            chunks.append(chunk_text)
            start_idx += max_chunk_tokens
        chunked_reasonings.append(chunks)
    return chunked_reasonings

def compute_compression_ratios(original_texts, compressed_texts, target_tokenizer):
    """
    Calculate compression ratios using the target model's tokenizer.
    
    Args:
        original_texts: list of original reasoning texts
        compressed_texts: list of compressed reasoning texts
        target_tokenizer: tokenizer of the model that will use this data
    
    Returns:
        tuple: (average_ratio, list of individual ratios)
    """
    assert len(original_texts) == len(compressed_texts), "Text lists must have same length"
    
    ratios = []
    for original, compressed in zip(original_texts, compressed_texts):
        original_tokens = target_tokenizer(original, add_special_tokens=False)["input_ids"]
        compressed_tokens = target_tokenizer(compressed, add_special_tokens=False)["input_ids"]
        ratio = float(len(compressed_tokens)) / float(len(original_tokens))
        ratios.append(ratio)
    
    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio, ratios

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
            
        dataset.append(data_dict)
    return dataset

def find_optimal_lingua_compression_rate(dataset, llm_lingua, llm_lingua_tokenizer, target_tokenizer, 
                                          max_chunk_tokens, target_keep_fraction):
    """
    Use binary search to find the optimal LLMLingua compression rate that achieves
    the target compression ratio when measured with the target model's tokenizer.
    
    Args:
        dataset: list of data dictionaries with 'response_reason' field
        llm_lingua: LLMLingua instance
        llm_lingua_tokenizer: tokenizer for LLMLingua
        target_tokenizer: tokenizer of target model
        max_chunk_tokens: max tokens per chunk for LLMLingua
        target_keep_fraction: desired compression ratio (e.g., 0.30 means keep 30% of tokens)
    
    Returns:
        list of data dictionaries with added 'pruned_response_reason' and 'keep_frac' fields
    """
    # Extract all reasoning texts
    reasoning_texts = [item["response_reason"] for item in dataset]
    
    # Split into chunks for LLMLingua processing
    chunked_reasonings = split_reasonings_into_chunks(reasoning_texts, max_chunk_tokens, llm_lingua_tokenizer)

    # Binary search for optimal LLMLingua compression rate
    min_rate, max_rate = 0.0, 1.0
    optimal_rate = None
    max_iterations = 30  # Prevent infinite loops
    min_range_threshold = 1e-6  # Stop if search range becomes too small
    
    for iteration in range(max_iterations):
        mid_rate = (min_rate + max_rate) / 2
        
        # Compress with current rate
        compressed_chunks = compress_reasoning_with_lingua(chunked_reasonings, mid_rate, llm_lingua)
        
        # Reconstruct full compressed texts
        compressed_texts = [" ".join(chunks) for chunks in compressed_chunks]
        
        # Measure actual compression ratio with target tokenizer
        avg_compression_ratio, _ = compute_compression_ratios(reasoning_texts, compressed_texts, target_tokenizer)
        
        print(f"Iter {iteration+1}/{max_iterations} | Target: {target_keep_fraction:.3f} | "
              f"Range: [{min_rate:.4f}, {max_rate:.4f}] | Trying: {mid_rate:.4f} | Actual: {avg_compression_ratio:.3f}")

        # Check convergence conditions
        if abs(target_keep_fraction - avg_compression_ratio) < 5e-3:
            optimal_rate = mid_rate
            print(f"Converged: Target reached within tolerance")
            break
        
        # Check if search range is too small to make progress
        if (max_rate - min_rate) < min_range_threshold:
            optimal_rate = mid_rate
            print(f"Converged: Search range too small ({max_rate - min_rate:.8f})")
            break
        
        # Adjust search range
        if avg_compression_ratio < target_keep_fraction:
            min_rate = mid_rate
        else:
            max_rate = mid_rate
    else:
        # Loop completed without break (max iterations reached)
        optimal_rate = mid_rate
        print(f"Warning: Max iterations reached. Using best approximation.")

    # Apply optimal compression
    final_compressed_chunks = compress_reasoning_with_lingua(chunked_reasonings, optimal_rate, llm_lingua)
    final_compressed_texts = [" ".join(chunks) for chunks in final_compressed_chunks]
    final_avg_ratio, individual_ratios = compute_compression_ratios(reasoning_texts, final_compressed_texts, target_tokenizer)

    print(f">>> OPTIMAL FOUND: Target={target_keep_fraction:.3f}, Actual={final_avg_ratio:.3f}, "
          f"LinguaRate={optimal_rate:.4f}")
    
    # Create output dataset with pruned reasonings
    pruned_dataset = []
    for idx, item in enumerate(dataset):
        item["pruned_response_reason"] = final_compressed_texts[idx]
        item["keep_frac"] = individual_ratios[idx]
        pruned_dataset.append(item)
    
    return pruned_dataset

# ================ Configuration =================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prune reasoning chains from teacher model using LLMLingua-2 for student model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prune-with-lingua.py llama-3-8b gsm8k train qwen-2.5-7b
  python prune-with-lingua.py mistral-7b mmlu validation llama-3-8b
        """
    )
    
    parser.add_argument(
        "student_model",
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
        "--keep-fractions",
        type=float,
        nargs="+",
        default=[0.30, 0.50, 0.70, 0.90],
        help="Target compression ratios (default: 0.30 0.50 0.70 0.90)"
    )
    
    parser.add_argument(
        "--max-lingua-length",
        type=int,
        default=500,
        help="Maximum tokens per chunk for LLMLingua processing (default: 500)"
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

# Parse arguments
args = parse_args()

# Map short names to full model paths
student_model = MODEL_NAME_MAP[args.student_model]
teacher_model = MODEL_NAME_MAP[args.teacher_model]
dataset = DATASET_NAME_MAP[args.dataset]
split = args.split

# Load tokenizer
student_tokenizer = AutoTokenizer.from_pretrained(student_model)

BASE_DIR = os.environ.get("GREEDY_PRUNER_DATA_DIR", os.path.join(os.getcwd(), "data"))

# Input/Output paths
source_filepath = os.path.join(
    BASE_DIR, "latest-data/teacher-generated",
    f"{dataset.replace('/', '-')}_{split}_{teacher_model.replace('/', '-')}.jsonl"
)
dest_path = os.path.join(BASE_DIR, "latest-data/lingua-pruned")

# Print configuration
print(f"\n{'='*80}")
print("Configuration:")
print(f"  Teacher Model: {teacher_model}")
print(f"  Student Model: {student_model}")
print(f"  Dataset: {dataset}")
print(f"  Split: {split}")
print(f"  Keep Fractions: {args.keep_fractions}")
print(f"  Max Lingua Length: {args.max_lingua_length}")
print(f"{'='*80}")
print(f"\nSource: {source_filepath}")
print(f"Destination: {dest_path}")

# LLMLingua setup
llm_lingua = LLMLingua()
llm_lingua_tokenizer = AutoTokenizer.from_pretrained("microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
max_lingua_length = args.max_lingua_length

# ===============================================

# Compression targets
target_keep_fractions = args.keep_fractions

# Prepare output directory
os.makedirs(dest_path, exist_ok=True)

# Load teacher-generated data
print(f"\n{'='*80}")
print(f"Processing: {source_filepath}")
print(f"{'='*80}")

teacher_dataset = load_teacher_generated_data(source_filepath)
print(f"Loaded {len(teacher_dataset)} examples")
    
# Generate pruned versions at different compression levels
for keep_fraction in target_keep_fractions:
    print(f"\n--- Compressing to {keep_fraction:.0%} ---")
    
    # Find optimal compression and apply it
    pruned_dataset = find_optimal_lingua_compression_rate(
        teacher_dataset, 
        llm_lingua, 
        llm_lingua_tokenizer, 
        student_tokenizer, 
        max_chunk_tokens=max_lingua_length, 
        target_keep_fraction=keep_fraction
    )
    
    # Save pruned dataset in hierarchical structure
    # Structure: dest_path/teacher_model/dataset/split/student_model/pruned_keepfrac_X.XX.jsonl
    output_folder = os.path.join(
        dest_path,
        teacher_model.replace('/', '-'),
        dataset.replace('/', '-'),
        split,
        student_model.replace('/', '-')
    )
    os.makedirs(output_folder, exist_ok=True)
    
    output_filename = f"pruned_keepfrac_{keep_fraction:.2f}.jsonl"
    output_filepath = os.path.join(output_folder, output_filename)
    
    with jsonlines.open(output_filepath, "w") as writer:
        writer.write_all(pruned_dataset)
    
    print(f"Saved to: {output_filepath}")

print(f"\n{'='*80}")
print(f"All files saved to: {dest_path}")
print(f"{'='*80}")















