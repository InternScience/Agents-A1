from judge import is_correct_judgement


def aggregate_results(rounds):
    """Aggregate results from N rounds.

    Args:
        rounds: dict mapping round_name -> list of result dicts
    """
    round_names = list(rounds.keys())
    query_results = {}

    for round_name, results in rounds.items():
        for result in results:
            query = result["question"]
            if query not in query_results:
                query_results[query] = {rn: None for rn in round_names}
                query_results[query]["answer"] = result["answer"]

            if is_correct_judgement(result["judgement"]):
                query_results[query][round_name] = "Correct"
            else:
                query_results[query][round_name] = result["judgement"].capitalize()

    return query_results


def _get_round_names(query_results):
    sample = next(iter(query_results.values()))
    return [k for k in sample.keys() if k != "answer"]


def calculate_pass_at_k(query_results, k=None):
    round_names = _get_round_names(query_results)
    if k is not None:
        round_names = round_names[:k]

    total_correct = 0
    for query, results in query_results.items():
        rounds = [results[rn] for rn in round_names]
        if "Correct" in rounds:
            total_correct += 1

    overall_pass = total_correct / len(query_results)
    return round(overall_pass * 100, 2)


def calculate_best_pass_at_1(query_results):
    round_names = _get_round_names(query_results)
    round_correct = {rn: 0 for rn in round_names}

    for query, results in query_results.items():
        for rn in round_names:
            if results[rn] == "Correct":
                round_correct[rn] += 1

    overall_best = max(round_correct[rn] / len(query_results) for rn in round_names)
    return round(overall_best * 100, 2)


def calculate_avg_pass_at_n(query_results):
    round_names = _get_round_names(query_results)
    total_correct = {rn: 0 for rn in round_names}

    for query, results in query_results.items():
        for rn in round_names:
            if results[rn] == "Correct":
                total_correct[rn] += 1

    print(total_correct)
    avg_overall = sum(total_correct[rn] / len(query_results) for rn in round_names) / len(round_names)
    return round(avg_overall * 100, 2)
