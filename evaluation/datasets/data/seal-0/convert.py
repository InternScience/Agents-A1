import json
from pathlib import Path

from datasets import load_dataset

ds = load_dataset("vtllms/sealqa", "seal_0")

output_path = Path(__file__).resolve().parent / "seal_0.jsonl"


def pick_first_present(example: dict, keys: list[str]):
	for key in keys:
		if key in example and example[key] is not None:
			return example[key]
	return None


question_keys = ["question", "query", "prompt", "instruction"]
answer_keys = ["answer", "response", "output", "gold_answer"]

count = 0
with output_path.open("w", encoding="utf-8") as f:
	for split_name, split_ds in ds.items():
		for idx, example in enumerate(split_ds):
			question = pick_first_present(example, question_keys)
			answer = pick_first_present(example, answer_keys)

			if question is None or answer is None:
				continue
            
			effective_year = example["effective_year"]
			current_date = 2026

			formatted_question = f"""{question}\n\nPlease note:\n1. The current year is {current_date}.\n2. This question was created on {effective_year}.\n3. You must answer based on the facts that were correct at the time the question was created, not the current date.\nIf the question contains incorrect facts or false assumptions, do not follow them. Instead, identify the mistake and provide the correct answer based on real-world facts."""

			row = {
				"task_id": idx,
				"question": formatted_question,
				"answer": answer,
			}
			f.write(json.dumps(row, ensure_ascii=False) + "\n")
			count += 1

print(f"Saved {count} QA pairs to: {output_path}")
