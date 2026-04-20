import jsonlines
import json
import sys
import random

def get_question_reason_answer(user_prompt):
    question_begin_marker = "--------------------------------------------------\n\nQuestion:\n"
    reason_begin_marker = "\n\nOutput Reason:\n"
    answer_begin_marker = "\nOutput Answer:\n"
    answer_end_marker = "\n\nOutput Reason Tokens (JSON Format)"

    split_1 = user_prompt.split(question_begin_marker)[1]
    question = split_1.split(reason_begin_marker)[0]
    split_2 = split_1.split(reason_begin_marker)[1]
    reason = split_2.split(answer_begin_marker)[0]
    split_3 = split_2.split(answer_begin_marker)[1]
    answer = split_3.split(answer_end_marker)[0]

    return question, reason, answer

def read_jsonl_file(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        data = [line for line in reader]
    return data

if __name__ == "__main__":

    setting = sys.argv[1]
    example_idx = int(sys.argv[2])

    # qwen-teacher qwen-pruner reason-answer - gsm8k train
    if int(setting) == 1:
        teacher_model, pruner_model, best_cand_criteria, dataset, split = "qwen-2.5-7b", "qwen-2.5-7b", "reason-answer", "gsm8k", "train"
        data_filepath = f"./data/category-prompt-response-{teacher_model}-{pruner_model}-{best_cand_criteria}-{dataset}-{split}.jsonl"

    # qwen-teacher qwen-pruner reason answer - gsm8k train
    if int(setting) == 2:
        teacher_model, pruner_model, best_cand_criteria, dataset, split = "qwen-2.5-7b", "qwen-2.5-7b", "answer", "gsm8k", "train"
        data_filepath = f"./data/category-prompt-response-{teacher_model}-{pruner_model}-{best_cand_criteria}-{dataset}-{split}.jsonl"

    # qwen-teacher llama-2-pruner reason answer - gsm8k train
    if int(setting) == 3:
        teacher_model, pruner_model, best_cand_criteria, dataset, split = "qwen-2.5-7b", "llama-2-7b", "reason-answer", "gsm8k", "train"
        data_filepath = f"./data/category-prompt-response-{teacher_model}-{pruner_model}-{best_cand_criteria}-{dataset}-{split}.jsonl" # file name was wrong, acutlaly this is reason-answer objective.

    # llama-3-teacher llama-3-pruner reason-answer - gsm8k train
    if int(setting) == 4:
        teacher_model, pruner_model, best_cand_criteria, dataset, split = "llama-3-8b", "llama-3-8b", "reason-answer", "gsm8k", "train"
        data_filepath = f"./data/category-prompt-response-{teacher_model}-{pruner_model}-{best_cand_criteria}-{dataset}-{split}.jsonl"

    data = read_jsonl_file(data_filepath)

    question = data[example_idx]["question"]
    user_prompt = data[example_idx]["user_prompt"]
    reason = get_question_reason_answer(user_prompt)[1]
    answer = get_question_reason_answer(user_prompt)[2]
    reason_tokens_json = data[example_idx]["reason_tokens_json"]
    max_deleted_step = max([token["deleted_step"] for token in reason_tokens_json])
    min_deleted_step = min([token["deleted_step"] for token in reason_tokens_json])
    print(f"Max deleted step: {max_deleted_step}, Min deleted step: {min_deleted_step}")
    print(f"Number of tokens: {len(reason_tokens_json)}")
    print(f"Max token position: {max([token['token_position'] for token in reason_tokens_json])}, Min token position: {min([token['token_position'] for token in reason_tokens_json])}")
    print(f"Unique token categories: {set([token['category'] for token in reason_tokens_json])}")

    token_categories = ["SYMBOLIC_MATH", "FUNCTION", "ENTITY_NAME", "META_DISCOURSE", "VERBAL_MATH", "COREFERENCE"]
    token_category_to_rgb = {
        'SYMBOLIC_MATH': (98, 160, 202),      # symbolic
        'FUNCTION': (255, 165, 85),           # grammar
        'ENTITY_NAME': (107, 188, 107),       # entity
        'META_DISCOURSE': (226, 104, 104),    # meta
        'VERBAL_MATH': (180, 148, 209),       # verbal
        'COREFERENCE': (174, 136, 129),       # coref
    }
    token_categories_to_display_name = {
        'SYMBOLIC_MATH': 'SymbMath',
        'FUNCTION': 'Grammar',
        'ENTITY_NAME': 'EntName',
        'META_DISCOURSE': 'MetaDisc',
        'VERBAL_MATH': 'VerbalMath',
        'COREFERENCE': 'CoRef'
    }
    # \definecolor{symbolic}{RGB}{98, 160, 202}
    # \definecolor{grammar}{RGB}{255, 165, 85}
    # \definecolor{entity}{RGB}{107, 188, 107}
    # \definecolor{meta}{RGB}{226, 104, 104}
    # \definecolor{verbal}{RGB}{180, 148, 209}
    # \definecolor{coref}{RGB}{174, 136, 129}

    # print(reason_tokens_json[0].keys()) # ['token_position', 'token_text', 'deleted_step', 'category', 'justification'] # token_position
    # token_position ranges from [0 to N-1], indicating the position of the token in the reason.
    # deleted_step ranges from [1 to N], indicating the step at which the token was deleted. at step 0 all tokens are kept.
    # token catgories is one of token_categories.