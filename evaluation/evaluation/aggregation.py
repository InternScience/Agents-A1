from judge import is_correct_judgement


def aggregate_results(round1_results, round2_results, round3_results):
    query_results = {}

    for results, round_name in zip([round1_results, round2_results, round3_results], ["round1", "round2", "round3"]):
        for result in results:
            query = result["question"]
            if query not in query_results:
                query_results[query] = {
                    "round1": None,
                    "round2": None,
                    "round3": None,
                    "answer": result["answer"]
                }

            if is_correct_judgement(result["judgement"]):
                query_results[query][round_name] = "Correct"
            else:
                query_results[query][round_name] = result["judgement"].capitalize()

    return query_results


def calculate_pass_at_k(query_results, k=10):
    total_correct = 0

    for query, results in query_results.items():
        rounds = [results["round1"], results["round2"], results["round3"]][:k]

        if "Correct" in rounds:
            total_correct += 1

    overall_pass = total_correct / len(query_results)
    return round(overall_pass * 100, 2)


def calculate_best_pass_at_1(query_results):
    round_correct = {round_name: 0 for round_name in ["round1", "round2", "round3"]}

    for query, results in query_results.items():
        for round_name in ["round1", "round2", "round3"]:
            if results[round_name] == "Correct":
                round_correct[round_name] += 1

    overall_best = max(
        round_correct[round_name] / len(query_results)
        for round_name in ["round1", "round2", "round3"]
    )

    return round(overall_best * 100, 2)


def calculate_avg_pass_at_3(query_results):
    round_names = ["round1", "round2", "round3"]
    total_correct = {round_name: 0 for round_name in round_names}

    for query, results in query_results.items():
        for round_name in round_names:
            if results[round_name] == "Correct":
                total_correct[round_name] += 1

    print(total_correct)
    avg_overall = sum(total_correct[r] / len(query_results) for r in round_names) / len(round_names)

    return round(avg_overall * 100, 2)
