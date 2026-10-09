import jsonlines
from math_verify import parse, verify
from datasets import load_dataset
import os
import argparse
from vllm import LLM, SamplingParams
from json_repair import repair_json
import re

def load_model_tokenizer(model_name, max_tokens):
    model = LLM(model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.80, max_model_len=max_tokens+2048, dtype="bfloat16", trust_remote_code=True)
    tokenizer = model.get_tokenizer()
    return model, tokenizer

def get_user_prompt_and_answer(dataset_name, dataset, example_idx):
    if "gsm8k" in dataset_name:
        user_prompt = dataset[example_idx]["question"]
        original_answer = dataset[example_idx]["answer"].split("####")[1].strip()
    elif "openaimath" in dataset_name:
        if "iamjanvijay" in dataset_name:
            user_prompt = dataset[example_idx]["question"]
            original_answer = dataset[example_idx]["answer"]
        else:
            user_prompt = dataset[example_idx]["problem"]
            original_answer = dataset[example_idx]["answer"]
    elif "MMLU-Pro" in dataset_name:
        if "iamjanvijay" in dataset_name:
            user_prompt = dataset[example_idx]["question"]
            original_answer = dataset[example_idx]["answer"]
        else:
            question = dataset[example_idx]["question"]
            options = dataset[example_idx]["options"]
            options_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # Extend as needed
            options_str = "\n".join([f"{options_labels[i]}. {option}" for i, option in enumerate(options)])
            user_prompt = f"{question}\n\nOptions:\n{options_str}"
            original_answer = dataset[example_idx]["answer"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return user_prompt, original_answer

def match_responses(dataset_name, pred_answer, gold_answer):
    if "MMLU-Pro" in dataset_name:
        return str(pred_answer).strip().lower() == str(gold_answer).strip().lower()
    else: # for the gsm8k and openaimath datasets
        pred_answer = parse(pred_answer)   
        gold_answer = parse(gold_answer)
        return verify(gold_answer, pred_answer) # order is important!

def batch_generate_responses(model_name, system_prompt, dataset_name, question_answer_pairs, max_tokens, temperature, n):
    # load the vllm models.
    model, tokenizer = load_model_tokenizer(model_name, max_tokens)

    # get all the questions and their answers.
    questions = [qa_pair[0] for qa_pair in question_answer_pairs]
    answers = [qa_pair[1] for qa_pair in question_answer_pairs]

    # create the prompts.
    prompts = [tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        tokenize=False,
        add_generation_prompt=True
    ) for question in questions]

    sampling_params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens)
    responses = model.generate(prompts, sampling_params)
    responses = [[response.outputs[idx].text for idx in range(len(response.outputs))] for response in responses]

    return questions, answers, responses

def parse_response_gsm8k(response):
    pattern = re.compile(
        r"""
        \{                      # opening brace
        \s*                     # optional whitespace / newlines
        ['"]answer['"]          # "answer" or 'answer'
        \s*:\s*                 # colon with optional whitespace
        """,
        re.VERBOSE,
    )

    matches = list(pattern.finditer(response))
    marker_index = matches[-1].start() if matches else -1

    if marker_index == -1:
        return "", "", "", False

    response_reason = response[:marker_index]
    response_answer = response[marker_index:]

    try:        
        # Attempt to repair/parse the JSON
        parsed = repair_json(response_answer, return_objects=True)
    except Exception:
        # If any exception occurs, parsing failed
        return "", "", response, False

    def extract_answer_from_json(obj):
        """Helper to extract answer from dict or list."""
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"])
        if isinstance(obj, list):
            # Search the list backwards for a dict containing an 'answer' key
            for item in reversed(obj):
                if isinstance(item, dict) and "answer" in item:
                    return str(item["answer"])
        return None

    answer = extract_answer_from_json(parsed)
    if answer is not None:
        return answer, response_reason, response_answer, True

    # If parsed successfully but answer not found
    return "", response_reason, response_answer, False


def parse_response_openaimath(response):
    MARKERS = ['Hence, the', 'Thus, the', 'Therefore, the', 'So, the']

    marker_index = -1
    for MARKER in MARKERS:
        if MARKER in response:
            marker_index = max(marker_index, response.rindex(MARKER))

    if marker_index == -1:
        return "", "", "", False

    response_reason = response[:marker_index]
    response_answer = response[marker_index:]

    try:
        # Try to extract content within \boxed{...}
        parsed_answer = parse(response_answer)[1]
        if not parsed_answer:
            return "", response_reason, response_answer, False
    except Exception:
        # Any failure results in unsuccessful parse
        return "", response_reason, response_answer, False
    
    return parsed_answer, response_reason, response_answer, True

def parse_response_mmlu(response):
    MARKERS = ['Hence, the', 'Thus, the', 'Therefore, the', 'So, the']

    marker_index = -1
    for MARKER in MARKERS:
        if MARKER in response:
            marker_index = max(marker_index, response.rindex(MARKER))

    if marker_index == -1:
        return "", "", "", False

    response_reason = response[:marker_index]
    response_answer = response[marker_index:]

    pattern = re.compile(
        r"""
        answer\s+is\s*[:]?        # "answer is" (+ optional colon)
        \s*                      # whitespace
        (?:\n\s*)*               # allow newlines
        [^A-Ja-j]*                # any junk symbols / latex / punctuation
        ([A-J])\b                 # capture answer letter A–J
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # Try response_answer first, then response_reason
    match = pattern.search(response_answer) or pattern.search(response_reason)
    if match:
        pred_letter = match.group(1).upper()
        if pred_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            return pred_letter, response_reason, response_answer, True

    return "", response_reason, response_answer, False

def parse_response(dataset_name, response):
    if dataset_name == "openai/gsm8k" or dataset_name == "iamjanvijay/gsm8k":
        return parse_response_gsm8k(response) 
    elif dataset_name == "simplescaling/openaimath" or dataset_name == "iamjanvijay/openaimath":
        return parse_response_openaimath(response)
    elif dataset_name == "TIGER-Lab/MMLU-Pro" or dataset_name == "iamjanvijay/MMLU-Pro":
        return parse_response_mmlu(response)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--dataset_name", default="openai/gsm8k", type=str)
    parser.add_argument("--dataset_split", default="train", type=str)
    parser.add_argument("--output_dir", default="data/teacher-generated", type=str)
    parser.add_argument("--num_examples", default=-1, type=int)
    parser.add_argument("--max_tokens", default=256*6, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--n", default=10, type=int)
    return parser.parse_args()

def main():

    args = parse_args()
    model_name, dataset_name, dataset_split, max_tokens, output_dir = args.model, args.dataset_name, args.dataset_split, args.max_tokens, args.output_dir
    temperature, n = args.temperature, args.n

    # load the dataset from huggingface.
    if dataset_name == "openai/gsm8k" or dataset_name == "iamjanvijay/gsm8k":
        dataset = load_dataset(dataset_name, 'main', split=dataset_split)
        system_prompt = '''You are a helpful assistant. Please solve the following problem step by step. At the end, output the final **answer only in the JSON format**:\n\n{"answer": "[numeric answer only]"}'''
    elif dataset_name == "simplescaling/openaimath" or dataset_name == "iamjanvijay/openaimath":
        dataset = load_dataset(dataset_name, split=dataset_split)
        system_prompt = '''You are a helpful assistant expert at solving math problems. Please solve the following math problem. First, think through the problem step by step. At the end, output the final answer only in the following format: "Thus, the final answer is: \\boxed{[numeric or mathematical expression only]}"'''
    elif dataset_name == "TIGER-Lab/MMLU-Pro" or dataset_name == "iamjanvijay/MMLU-Pro":
        dataset = load_dataset(dataset_name, split=dataset_split)
        system_prompt = '''You are a helpful assistant. Please solve the following MCQ problem step by step to select the correct answer from the given options. At the end, output the final answer only in the following format: "Thus, the final answer is: [correct letter choice only]"'''
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if args.num_examples > 0:
        dataset = dataset.shuffle(seed=42).select(range(args.num_examples))

    print(f">>> Loaded dataset: {len(dataset)} examples")
    
    # setup the output filename.
    os.makedirs(output_dir, exist_ok=True)

    question_answer_pairs = [get_user_prompt_and_answer(dataset_name, dataset, idx) for idx, example in enumerate(dataset)]
    questions, answers, responses = batch_generate_responses(model_name, system_prompt, dataset_name, question_answer_pairs, max_tokens=max_tokens, temperature=temperature, n=n)

    out_fpath = os.path.join(output_dir, f"{dataset_name.replace('/', '-')}_{dataset_split.replace('/', '-')}_{model_name.replace('/', '-')}.jsonl")
    writer = jsonlines.open(out_fpath, "w")

    total, correct, unparsed = 0, 0, 0
    for question, answer, response_list in zip(questions, answers, responses):  # response_list has n samples per question
        correct_response_found = False
        first_parsed = None
        
        # Try to find a correct response
        for response in response_list:
            parsed_answer, response_reason, response_answer, is_parsed = parse_response(dataset_name, response)
            
            if is_parsed:
                if first_parsed is None:
                    first_parsed = (response, parsed_answer, response_reason, response_answer, is_parsed)
                
                is_correct = match_responses(dataset_name, parsed_answer, answer)
                if is_correct:
                    # Found a correct response, use this one
                    writer.write({"system_prompt": system_prompt, "user_prompt": question, "response": response, "response_reason": response_reason, "response_answer": response_answer, "parsed_answer": parsed_answer, "original_answer": answer, "is_correct": True, "is_parsed": True})
                    correct += 1
                    correct_response_found = True
                    break
        
        if not correct_response_found:
            # No correct response found, use the first parsed one or first response
            if first_parsed is not None:
                response, parsed_answer, response_reason, response_answer, is_parsed = first_parsed
                writer.write({"system_prompt": system_prompt, "user_prompt": question, "response": response, "response_reason": response_reason, "response_answer": response_answer, "parsed_answer": parsed_answer, "original_answer": answer, "is_correct": False, "is_parsed": True})
            else:
                # None parsed, save the first response
                response = response_list[0]
                writer.write({"system_prompt": system_prompt, "user_prompt": question, "response": response, "response_reason": "", "response_answer": "", "parsed_answer": "", "original_answer": answer, "is_correct": False, "is_parsed": False})
                unparsed += 1
        
        total += 1

    writer.write({"model_name": model_name, "dataset_name": dataset_name, "dataset_split": dataset_split, "max_tokens": max_tokens, "temperature": temperature, "n": n, "total": total, "correct": correct, "unparsed": unparsed, "accuracy": correct / total * 100, "unparsed_rate": unparsed / total * 100})
    print(f"Total: {total}, Correct: {correct}, Unparsed: {unparsed}, Accuracy: {correct / total * 100:.2f}%")
    writer.close()

if __name__ == "__main__":
    main()