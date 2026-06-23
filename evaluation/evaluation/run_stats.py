import json
import os

import tiktoken
from transformers import AutoTokenizer

from data_loading import get_termination_value, process_single_round
from judge import is_correct_judgement


def _load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH", ""))
    except Exception:
        return tiktoken.encoding_for_model("gpt-4o")


def count_tokens_with_tokenizer(text, tokenizer):
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text) // 4


def aggregate_statistics(round1_file, round2_file, round3_file):
    round1_stats = single_round_statistics(round1_file)
    round2_stats = single_round_statistics(round2_file)
    round3_stats = single_round_statistics(round3_file)

    keys = round1_stats.keys()
    avg_stats = {}
    for key in keys:
        if isinstance(round1_stats[key], dict):
            avg_stats[key] = {}
            all_keys = set(round1_stats[key].keys()) | set(round2_stats[key].keys()) | set(round3_stats[key].keys())
            for nested_key in all_keys:
                val1 = round1_stats[key].get(nested_key, 0)
                val2 = round2_stats[key].get(nested_key, 0)
                val3 = round3_stats[key].get(nested_key, 0)
                avg_stats[key][nested_key] = round((val1 + val2 + val3) / 3, 3)
        else:
            avg_stats[key] = round((round1_stats[key] + round2_stats[key] + round3_stats[key]) / 3, 3)

    return avg_stats


def single_round_statistics(input_file):
    contents = process_single_round(input_file)

    num_invalid, num_extra = 0, 0

    tool_use_cnt, visit_tool_cnt, search_tool_cnt, other_tool_cnt = [], [], [], []

    all_ans_lengths, all_think_lengths = [], []

    all_tool_calls_per_question = []
    all_assistant_tokens_per_question = []
    all_assistant_tokens_per_message = []
    termination_counts = {}

    tokenizer = _load_tokenizer()

    for item in contents:
        messages = item["messages"]
        final_msg = messages[-1]["content"] if len(messages) else ""

        if "<answer>" not in final_msg or "</answer>" not in final_msg:
            num_invalid += 1
            answer_length = 0
        else:
            answer_length = len(final_msg.split("<answer>")[1].split("</answer>")[0].strip())

        num_tool_use, num_visit_tool, num_search_tool, num_other_tool = 0, 0, 0, 0
        think_lengths = []
        question_assistant_tokens = 0

        for msg in messages:
            if msg['role'] == 'assistant':
                reasoning = None
                if "reasoning" in msg:
                    reasoning = msg["reasoning"]
                elif "reasoning_content" in msg:
                    reasoning = msg["reasoning_content"]
                content = msg['content'] if msg['content'] is not None else ""
                content = "<think>" + reasoning + "<\\think>" + content if reasoning is not None else content

                if 'tool_calls' in msg:
                    tool_calls = msg['tool_calls']
                    if tool_calls is not None:
                        for tool_call in tool_calls:
                            num_tool_use += 1
                            tool_name = tool_call["function"]["name"]
                            if tool_name == 'google_search':
                                num_search_tool += 1
                            elif 'read_page' in tool_name:
                                num_visit_tool += 1
                            else:
                                num_other_tool += 1
                else:
                    remaining_content = content

                    while ("<tool_call>" in remaining_content and "</tool_call>" in remaining_content) or ("<tool_name>" in remaining_content and "</tool_name>" in remaining_content):
                        if "<tool_call>" in remaining_content and "</tool_call>" in remaining_content:
                            start_idx = remaining_content.find("<tool_call>")
                            end_idx = remaining_content.find("</tool_call>")

                            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                tool_call_content = remaining_content[start_idx + 11:end_idx].strip()
                                if tool_call_content:
                                    num_tool_use += 1

                                    try:
                                        tool_call = json.loads(tool_call_content)
                                        tool_name = tool_call.get('name', '')
                                        if tool_name == 'google_search':
                                            num_search_tool += 1
                                        elif 'read_page' in tool_name:
                                            num_visit_tool += 1
                                        else:
                                            num_other_tool += 1
                                    except Exception:
                                        if "visit" in tool_call_content or "scrape" in tool_call_content:
                                            num_visit_tool += 1
                                        elif "search" in tool_call_content:
                                            num_search_tool += 1
                                        else:
                                            num_other_tool += 1

                                remaining_content = remaining_content[end_idx + 12:]
                            else:
                                break
                        else:
                            start_idx = remaining_content.find("<tool_name>")
                            end_idx = remaining_content.find("</tool_name>")

                            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                tool_call_content = remaining_content[start_idx + 11:end_idx].strip()
                                if tool_call_content:
                                    num_tool_use += 1

                                    try:
                                        tool_name = tool_call_content
                                        if tool_name == 'google_search':
                                            num_search_tool += 1
                                        elif 'read_page' in tool_name:
                                            num_visit_tool += 1
                                        else:
                                            num_other_tool += 1
                                    except Exception:
                                        if "visit" in tool_call_content or "scrape" in tool_call_content:
                                            num_visit_tool += 1
                                        elif "search" in tool_call_content:
                                            num_search_tool += 1
                                        else:
                                            num_other_tool += 1

                                remaining_content = remaining_content[end_idx + 12:]
                            else:
                                break

                think_lengths.append(len(content))

                assistant_tokens = count_tokens_with_tokenizer(content, tokenizer)
                question_assistant_tokens += assistant_tokens
                all_assistant_tokens_per_message.append(assistant_tokens)

        tool_use_cnt.append(num_tool_use)
        visit_tool_cnt.append(num_visit_tool)
        search_tool_cnt.append(num_search_tool)
        other_tool_cnt.append(num_other_tool)

        all_ans_lengths.append(answer_length)
        think_length = sum(think_lengths) / len(think_lengths) if think_lengths else 0
        all_think_lengths.append(think_length)

        all_tool_calls_per_question.append(num_tool_use)
        all_assistant_tokens_per_question.append(question_assistant_tokens)

        termination = get_termination_value(item)
        termination_counts[termination] = termination_counts.get(termination, 0) + 1

        try:
            if len(tokenizer.encode("".join([msg["content"] for msg in messages]))) > 30000:
                num_extra += 1
        except Exception:
            pass

    total_questions = len(contents)
    termination_freq = {k: round(v / total_questions, 3) for k, v in termination_counts.items()}

    return {
        "extra_length": num_extra,
        "num_invalid": num_invalid,
        "avg_action": sum(tool_use_cnt) / len(tool_use_cnt),
        "avg_visit_action": sum(visit_tool_cnt) / len(visit_tool_cnt),
        "avg_search_action": sum(search_tool_cnt) / len(search_tool_cnt),
        "avg_other_action": sum(other_tool_cnt) / len(other_tool_cnt),
        "avg_ans_length": sum(all_ans_lengths) / len(all_ans_lengths),
        "avg_think_length": sum(all_think_lengths) / len(all_think_lengths),
        "avg_tool_calls_per_question": sum(all_tool_calls_per_question) / len(all_tool_calls_per_question) if all_tool_calls_per_question else 0,
        "avg_assistant_tokens_per_question": sum(all_assistant_tokens_per_question) / len(all_assistant_tokens_per_question) if all_assistant_tokens_per_question else 0,
        "avg_assistant_tokens_per_message": sum(all_assistant_tokens_per_message) / len(all_assistant_tokens_per_message) if all_assistant_tokens_per_message else 0,
        "termination_freq": termination_freq
    }


def calculate_enhanced_statistics(round_results, round_items):
    tokenizer = _load_tokenizer()

    correct_tool_calls = []
    correct_assistant_tokens = []

    for round_name in ["round1", "round2", "round3"]:
        results = round_results[round_name]
        items = round_items[round_name]

        for result in results:
            if not is_correct_judgement(result["judgement"]):
                continue
            try:
                matching_item = [item for item in items if item['messages'][1]['content'] == result['question']]
            except Exception:
                items = [item for item in items if len(item['messages']) > 0]
                matching_item = [item for item in items if item['messages'][1]['content'] == result['question']]
            if not matching_item:
                continue
            item = matching_item[0]

            messages = item["messages"]
            num_tool_use = 0
            question_assistant_tokens = 0

            for msg in messages:
                if msg['role'] == 'assistant':
                    reasoning = None
                    if "reasoning" in msg:
                        reasoning = msg["reasoning"]
                    elif "reasoning_content" in msg:
                        reasoning = msg["reasoning_content"]
                    content = msg['content'] if msg['content'] is not None else ""
                    content = "<think>" + reasoning + "<\\think>" + content if reasoning is not None else content

                    think_content = content.split('<think>')[-1].split('</think>')[0]

                    num_tool_use += 1

                    assistant_tokens = count_tokens_with_tokenizer(think_content, tokenizer)
                    question_assistant_tokens += assistant_tokens

            correct_tool_calls.append(num_tool_use)
            correct_assistant_tokens.append(question_assistant_tokens)

    avg_tool_calls_correct = sum(correct_tool_calls) / len(correct_tool_calls) if correct_tool_calls else 0
    avg_assistant_tokens_correct = sum(correct_assistant_tokens) / len(correct_assistant_tokens) if correct_assistant_tokens else 0

    return {
        "avg_tool_calls_per_question_correctly_solved": round(avg_tool_calls_correct, 3),
        "avg_assistant_tokens_per_question_correctly_solved": round(avg_assistant_tokens_correct, 3)
    }
