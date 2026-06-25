import argparse
import os
import json


COLUMNS = ["Overall", "Easy", "Hard", "Short", "Medium", "Long"]
compensated = False


def load_predictions(filename):
    try:
        with open(filename, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        with open(filename, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]


def compute_scores(pred_data):
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for pred in pred_data:
        acc = int(pred['judge'])
        if compensated and pred["pred"] is None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    overall = round(100*(easy_acc+hard_acc)/len(pred_data), 1) if len(pred_data) > 0 else 0
    easy_score = round(100*easy_acc/easy, 1) if easy > 0 else 0
    hard_score = round(100*hard_acc/hard, 1) if hard > 0 else 0
    short_score = round(100*short_acc/short, 1) if short > 0 else 0
    medium_score = round(100*medium_acc/medium, 1) if medium > 0 else 0
    long_score = round(100*long_acc/long, 1) if long > 0 else 0
    return {
        "Overall": overall,
        "Easy": easy_score,
        "Hard": hard_score,
        "Short": short_score,
        "Medium": medium_score,
        "Long": long_score,
    }


def format_row(name, scores):
    return '\t'.join([name] + [str(scores[column]) for column in COLUMNS])


def result_name(model_name, suffix):
    return f"{model_name}_{suffix}" if suffix else model_name


def maybe_append_average(output, scores_by_model, average_model, run_count, result_suffix):
    if not average_model:
        return

    average_model = average_model.split("/")[-1]
    run_names = [
        result_name(f"{average_model}_run{run_id}", result_suffix)
        for run_id in range(1, run_count + 1)
    ]
    missing = [name for name in run_names if name not in scores_by_model]
    if missing:
        print("[result.py] Missing run result(s), skip average: " + ", ".join(missing))
        return

    avg_scores = {}
    for column in COLUMNS:
        avg_scores[column] = round(sum(scores_by_model[name][column] for name in run_names) / run_count, 1)
    output.append(format_row(result_name(f"{average_model}_avg", result_suffix), avg_scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="result.txt")
    parser.add_argument("--average_model", type=str, default=None)
    parser.add_argument("--run_count", type=int, default=3)
    parser.add_argument("--result_suffix", type=str, default="")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.results_dir) if f.endswith('.jsonl') or f.endswith('.json')]
    output = ["Model\t" + "\t".join(COLUMNS)]
    scores_by_model = {}

    for file in files:
        filename = os.path.join(args.results_dir, file)
        pred_data = load_predictions(filename)
        name = '.'.join(file.split('.')[:-1])
        scores = compute_scores(pred_data)
        scores_by_model[name] = scores
        output.append(format_row(name, scores))

    maybe_append_average(output, scores_by_model, args.average_model, args.run_count, args.result_suffix)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))


if __name__ == "__main__":
    main()
