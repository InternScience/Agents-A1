import argparse
import concurrent.futures
import json
import os
import traceback

from tqdm import tqdm

from aggregation import (
    aggregate_results,
    calculate_avg_pass_at_n,
    calculate_best_pass_at_1,
    calculate_pass_at_k,
)
from data_loading import process_single_round
from judge import call_llm_judge, is_correct_judgement, resolve_judge
from run_stats import (
    aggregate_statistics,
    calculate_enhanced_statistics,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions across multiple rounds")
    parser.add_argument("--input-folder", help="Path to prediction files", required=True)
    parser.add_argument("--restore-result-filename", default='summary.jsonl', help="record result")
    parser.add_argument(
        "--dataset", type=str, default="gaia", choices=[
            "gaia", "browsecomp_zh", "browsecomp_en_full",
            "xbench-deepsearch", "seal-0", "hle"
        ]
    )
    parser.add_argument("--rollout-count", type=int, default=3, help="Number of rollout iterations")
    args = parser.parse_args()

    dataset = args.dataset
    rollout_count = args.rollout_count
    judge_model, judge_prompt = resolve_judge(dataset)

    print(f"Using {dataset} judge prompt ...")
    print(f"Judge prompt:\n {judge_prompt}")
    print(f"Judge model:\n {judge_model}")

    round_names = [f"round{i}" for i in range(1, rollout_count + 1)]
    round_files = [os.path.join(args.input_folder, f"iter{i}.jsonl") for i in range(1, rollout_count + 1)]

    for file in round_files:
        assert os.path.exists(file), f"Prediction {file} not found, {rollout_count} rounds are required"

    round_items = {
        round_names[i]: process_single_round(round_files[i])
        for i in range(rollout_count)
    }

    round_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for round_name, items in round_items.items():
            futures = {
                executor.submit(
                    call_llm_judge, item,
                    judge_prompt=judge_prompt,
                    dataset=dataset,
                    judge_model=judge_model,
                ): item
                for item in items
            }
            round_results[round_name] = []

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Evaluating {round_name}"):
                round_results[round_name].append(future.result())

    for i, round_name in enumerate(round_names):
        input_file = round_files[i]
        scored_file = input_file.replace(".jsonl", "_scored.jsonl")
        original_items = round_items[round_name]

        sorted_results = sorted(
            round_results[round_name],
            key=lambda x: original_items.index(next(item for item in original_items if item["question"] == x["question"]))
        )

        with open(scored_file, 'w', encoding='utf-8') as f:
            for orig_item, scored_result in zip(original_items, sorted_results):
                scored_item = {
                    "is_correct": is_correct_judgement(scored_result["judgement"]),
                    "judgement": scored_result["judgement"]
                }
                if "error" in scored_result:
                    scored_item["error"] = scored_result["error"]

                scored_item.update(orig_item)
                f.write(json.dumps(scored_item, ensure_ascii=False) + '\n')

    aggr_results = aggregate_results(round_results)

    pass_at_n = calculate_pass_at_k(aggr_results, k=rollout_count)
    best_pass_at_1 = calculate_best_pass_at_1(aggr_results)
    avg_pass_at_n = calculate_avg_pass_at_n(aggr_results)

    round_performance = {
        f"Round{i}_Pass@1": round(
            sum(1 for r in round_results[f"round{i}"] if is_correct_judgement(r["judgement"])) / len(round_results[f"round{i}"]) * 100,
            2
        )
        for i in range(1, rollout_count + 1)
    }

    print(f"===========")
    print(f"Avg. Pass@{rollout_count} {avg_pass_at_n}%")
    print(f"Best Pass@1 {best_pass_at_1}%")
    print(f"Pass@{rollout_count} {pass_at_n}%")
    round_perf_str = "  ".join(f"Round {i}: {round_performance[f'Round{i}_Pass@1']}%" for i in range(1, rollout_count + 1))
    print(f"Pass@1 {round_perf_str}\n")

    aggr_statistics = aggregate_statistics(round_files)
    print(f"# Invalid {aggr_statistics['num_invalid']}  # Extra Length {aggr_statistics['extra_length']}")
    print(f"Avg. Action {aggr_statistics['avg_action']:.2f}  Avg. Visit Action {aggr_statistics['avg_visit_action']:.2f}  Avg. Search Action {aggr_statistics['avg_search_action']:.2f}  Avg. Other Action {aggr_statistics['avg_other_action']:.2f}")
    print(f"Avg. Answer Length {aggr_statistics['avg_ans_length']:.2f}  Avg. Thinking Length {aggr_statistics['avg_think_length']:.2f}")
    enhanced_statistics = calculate_enhanced_statistics(round_results, round_items)
    print(f"\n=== ADDITIONAL STATISTICS ===")
    print(f"Avg. Tool Calls per Question: {aggr_statistics['avg_tool_calls_per_question']:.2f}")
    print(f"Avg. Tool Calls per Question (Correctly Solved): {enhanced_statistics['avg_tool_calls_per_question_correctly_solved']:.2f}")
    print(f"Avg. Assistant Tokens per Question: {aggr_statistics['avg_assistant_tokens_per_question']:.2f}")
    print(f"Avg. Assistant Tokens per Question (Correctly Solved): {enhanced_statistics['avg_assistant_tokens_per_question_correctly_solved']:.2f}")
    print(f"Avg. Assistant Tokens per Message: {aggr_statistics['avg_assistant_tokens_per_message']:.2f}")

    print(f"\n=== TERMINATION FREQUENCIES ===")
    for termination_type, frequency in aggr_statistics['termination_freq'].items():
        print(f"{termination_type}: {frequency:.3f}")

    print(f"===========")

    overall_eval_dict = {
        "dataset": dataset,
        "files": {round_names[i]: round_files[i] for i in range(rollout_count)},
        "overall": {
            f"avg_pass_at_{rollout_count}": avg_pass_at_n,
            "best_pass_at_1": best_pass_at_1,
            f"pass_at_{rollout_count}": pass_at_n
        },
        "individual": round_performance,
        "statistics": {**aggr_statistics, **enhanced_statistics}
    }

    with open(f"{args.input_folder}/{args.restore_result_filename}", 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(overall_eval_dict, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"Evaluation Failed: {e}")
        print("Trace Back", error_str)
