import json

from constants import APPENDED_TEXT


def process_single_round(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        items = []
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    for item in items:
        question = item["question"]
        while question.endswith(APPENDED_TEXT):
            question = question[:-len(APPENDED_TEXT)]
        item["question"] = question.strip()
    return items


def get_termination_value(item):
    if "termination" in item:
        return item["termination"]

    messages = item.get("messages", [])
    if not messages:
        return "unknown"

    last_message = messages[-1]["content"] if messages else ""

    if "max_turns_reached" in last_message.lower():
        return "max_turns_reached"
    elif "max_tokens_reached" in last_message.lower():
        return "max_tokens_reached"
    elif "<answer>" in last_message and "</answer>" in last_message:
        return "answered"
    else:
        return "unknown"
