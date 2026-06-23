# Evaluation

Evaluate inference results using LLM-based judging.

## Usage

```bash
# Set judge API credentials, or put them in ../.env
export JUDGE_API_KEY=your_judge_api_key
export JUDGE_API_BASE=https://api.openai.com/v1
# Optional: override the default judge selected for each dataset
export JUDGE_MODEL_NAME=gpt-4o

# Run evaluation
python evaluate.py \
    --input-folder /path/to/results/ \
    --dataset gaia
```

## Requirements

The input folder must contain 3 iteration files: `iter1.jsonl`, `iter2.jsonl`, `iter3.jsonl`.

For single-rollout evaluation, copy iter1 to iter2 and iter3:
```bash
cp iter1.jsonl iter2.jsonl
cp iter1.jsonl iter3.jsonl
```

## Supported Datasets

| Dataset | Judge Model (default) |
|---------|----------------------|
| `gaia` | gpt-4o |
| `seal-0` | gpt-4.1 |
| `browsecomp_en_full` | gpt-4o-2024-08-06 |
| `browsecomp_zh` | gpt-4o-2024-08-06 |
| `xbench-deepsearch` | google/gemini-2.0-flash-001 |

Override the judge model by setting `JUDGE_MODEL_NAME` in `../.env`.

## Output

The script outputs:
- **Avg. Pass@3**: Average accuracy across 3 rollouts
- **Best Pass@1**: Best single-rollout accuracy
- **Pass@3**: Whether any of 3 rollouts got the correct answer
- **Statistics**: Tool usage, token counts, termination patterns
- Scored JSONL files (`iter1_scored.jsonl`, etc.)
