import jsonlines
from vllm import LLM, SamplingParams
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_teacher import parse_response, match_responses

base_dir = os.environ.get("GREEDY_PRUNER_DATA_DIR", os.path.join(os.getcwd(), "data"))

def read_question_reason_file(file_path, use_pruned=True):
    data = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            if "model_name" in line or line["is_correct"] == False or len(line["response_reason"]) < 100:
                continue
            system_prompt, user_prompt, response_reason, response_answer = line["system_prompt"], line["user_prompt"], line["response_reason"], line["response_answer"]
            if use_pruned:
                response_reason = line["pruned_response_reason"]
            data.append({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_reason": response_reason,
                "response_answer": response_answer
            })
    return data

def load_model_tokenizer(model_name, max_tokens):
    llm_model = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_tokens,
    )
    tokenizer = llm_model.get_tokenizer()
    return llm_model, tokenizer

def model_to_eot_token(model_name):
    assert model_name in ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
    model_to_eot_token = {
        "meta-llama/Llama-3.1-8B-Instruct": "<|eot_id|>",
        "Qwen/Qwen2.5-7B-Instruct": "<|im_end|>\n"
    }
    return model_to_eot_token[model_name]

def dataset_to_answer_begin(dataset_name):
    assert dataset_name in ["openai/gsm8k", "simplescaling/openaimath", "TIGER-Lab/MMLU-Pro"]
    dataset_answer_begins = {
        "openai/gsm8k": "{\"answer\":",
        "simplescaling/openaimath": "Thus, the final answer is: ",
        "TIGER-Lab/MMLU-Pro": "Thus, the final answer is: "
    }
    if dataset_name not in dataset_answer_begins:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset_answer_begins[dataset_name]

def generate_answer_from_question_reason(model_name, dataset_name, data, max_tokens):
    # load the model, tokenizer, and eot token.
    eot_token = model_to_eot_token(model_name)
    print(f">>> EOT token (model_name: {model_name}): ", eot_token)
    answer_begin = dataset_to_answer_begin(dataset_name)
    print(f">>> Answer begin (dataset_name: {dataset_name}): ", answer_begin)
    llm_model, tokenizer = load_model_tokenizer(model_name, max_tokens)

    # for each item in the data, create a prompt to be input to the model.
    prompts = []
    for item in data:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_prompt"]},
                {"role": "assistant", "content": item["response_reason"]},
            ],
            tokenize=False,
            add_generation_prompt=False # set to False because we want to continue the last assistant message.
        )
        assert prompt[-len(eot_token):] == eot_token, "The prompt does not end with the eot token"
        prompt = prompt[:-len(eot_token)] + answer_begin # we want to continue the last assistant message.
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    outputs = llm_model.generate(prompts, sampling_params)
    answers = [answer_begin + output.outputs[0].text for output in outputs]
    return answers

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate answers from question and reason inputs using a specified model.")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the model to use.")
    parser.add_argument("--dataset-name", type=str, default="openai/gsm8k", help="Name of the dataset to use.")
    parser.add_argument("--input-path", type=str, default=os.path.join(base_dir, "lingua-pruned/meta-llama-Llama-3.1-8B-Instruct/openai-gsm8k/train/meta-llama-Llama-3.1-8B-Instruct/pruned_keepfrac_0.30.jsonl"), help="Path to the question-reason input file.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum number of tokens for the model context window.")
    parser.add_argument("--use-pruned", action="store_true", help="Use pruned response reason.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    model_name = args.model_name
    input_path = args.input_path
    max_tokens = args.max_tokens
    dataset_name = args.dataset_name
    use_pruned = args.use_pruned

    data = read_question_reason_file(input_path, use_pruned)
    print(f">>> Loaded {len(data)} examples from {input_path} {'with pruned' if use_pruned else 'without pruned'} response reason")

    pred_answers = generate_answer_from_question_reason(model_name, dataset_name, data, max_tokens)
    print(">>> First 100 predictions: ", pred_answers[:100])
    answers = [item["response_answer"] for item in data]
    print(">>> First 100 answers: ", answers[:100])

    correct, incorrect = 0, 0
    for answer, pred_answer in zip(answers, pred_answers):
        parsed_answer, parsed_pred_answer = parse_response(dataset_name, answer)[0], parse_response(dataset_name, pred_answer)[0]
        responses_matched = match_responses(dataset_name, pred_answer=parsed_pred_answer, gold_answer=parsed_answer)
        if responses_matched:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct / (correct + incorrect) * 100
    print(f">>> Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
