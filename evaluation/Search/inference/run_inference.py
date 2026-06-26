import argparse
import json
import os
import math
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from agent import MultiTurnReactAgent, TOOL_MAP, _extract_reasoning

# Must match evaluation/constants.py::APPENDED_TEXT byte-for-byte.
# Appended to each question here; the evaluator strips it before judging.
APPENDED_TEXT = "\n\nPut your reasoning content inside <think></think>. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-turn ReAct agent inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model or model name for remote API")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model or model name for remote API")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file (JSONL or JSON)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--presence-penalty", type=float, default=1.1)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--roll-out-count", type=int, default=3)
    parser.add_argument("--total-splits", type=int, default=1, help="Total number of dataset splits for distributed inference")
    parser.add_argument("--worker-split", type=int, default=1, help="Which split this worker handles (1-indexed)")
    parser.add_argument("--ports", type=str, default="6001", help="Comma-separated list of vLLM server ports (for local mode)")
    args = parser.parse_args()

    model = args.model
    model_path = args.model_path
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    model_name = os.path.basename(model.rstrip('/'))
    model_dir = os.path.join(output_base, model_name)
    dataset_dir = os.path.join(model_dir, os.path.basename(os.path.dirname(args.dataset)))
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Dataset path: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    # Load dataset
    data_filepath = args.dataset
    dataset_base_dir = os.path.dirname(os.path.abspath(data_filepath))
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)
    items = items[start_idx:end_idx]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    # Setup output files
    if total_splits > 1:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}_split{worker_split}of{total_splits}.jsonl") for i in range(1, roll_out_count + 1)}
    else:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl") for i in range(1, roll_out_count + 1)}

    # Check for already-processed queries (supports resume)
    processed_queries_per_rollout = {}
    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass
        processed_queries_per_rollout[rollout_idx] = processed_queries

    # Build task list with port assignment
    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    planning_ports = [int(p) for p in args.ports.split(",")]
    planning_rr_idx = 0
    question_to_ports = {}

    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for item in items:
            question = item.get("question", "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")

            if question and APPENDED_TEXT not in question:
                question = question + APPENDED_TEXT
                item["question"] = question

            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                if question not in question_to_ports:
                    planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
                    question_to_ports[question] = planning_port
                    planning_rr_idx += 1
                planning_port = question_to_ports[question]
                tasks_to_run_all.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                    "planning_port": planning_port,
                    "base_dir": dataset_base_dir,
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        llm_cfg = {
            'model': model,
            'model_path': model_path,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': args.top_k,
                'presence_penalty': args.presence_penalty
            },
        }

        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            # Tools are configured in agent.py (TOOL_CLASS) and prompt.py (TOOLS).
            # Reflect the registered set here so this never drifts or hides a tool.
            function_list=list(TOOL_MAP.keys())
        )

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(test_agent._run, task, model): task
                for task in tasks_to_run_all
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]
                try:
                    result = future.result()
                    validated_msgs = []
                    for msg in result["messages"]:
                        if isinstance(msg, dict):
                            validated_msgs.append(msg)
                        else:
                            _, reasoning_value = _extract_reasoning(msg)
                            content = ("<think>" + reasoning_value.strip() + "</think>" + msg.content.strip()) if reasoning_value else msg.content.strip()
                            validated_msgs.append({
                                "role": msg.role,
                                "content": content,
                                "tool_calls": [{"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls] if hasattr(msg, 'tool_calls') and msg.tool_calls is not None else None,
                            })
                    result["messages"] = validated_msgs

                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    traceback.print_exc()
                    question = task_info["item"].get("question", "")
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "error": str(exc),
                        "messages": [],
                        "prediction": "[Failed]",
                    }
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")
