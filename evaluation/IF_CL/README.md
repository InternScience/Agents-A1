# Agents-A1 Evaluation on Instruction Following and Context Learning

This directory provides the evaluation scripts used for the **Instruction
Following** and **Context Learning** part of Agents-A1.

Benchmarks:

- IFBench
- IFEval
- LongBench-v2

Run commands below from this directory:

```bash
cd evaluation/IF_CL
```

The top-level `eval_long.sh` runs the selected benchmarks directly in the
current machine/container. It does not submit `rjob` jobs.

## Data Layout

Benchmark data is stored in the shared Agents-A1 evaluation data directory:

```text
evaluation/datasets/data/
├── ifbench/
│   └── IFBench_test.jsonl
├── ifeval/
│   └── input_data.jsonl
└── longbench_v2/
    └── README.md
```

LongBench-v2 is not committed as an Arrow dataset because the downloaded file is
larger than GitHub's normal file size limit. Download it before running
LongBench:

```bash
cd evaluation/IF_CL/LongBench
python prepare_data.py
```

The script writes to `evaluation/datasets/data/longbench_v2` by default.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare optional NLTK resources for IFBench/IFEval:

```bash
python scripts/download_nltk_data.py
```

Run all three benchmarks:

```bash
MODEL_PATH=/path/to/Agents-A1 \
MODEL_NAME=Agents-A1 \
TOKENIZER_PATH=/path/to/Agents-A1 \
bash eval_long.sh --model_name Agents-A1 --model_path /path/to/Agents-A1
```

Run a subset:

```bash
BENCHMARKS="ifbench ifeval" \
MODEL_PATH=/path/to/Agents-A1 \
MODEL_NAME=Agents-A1 \
bash eval_long.sh --model_name Agents-A1 --model_path /path/to/Agents-A1
```

Allowed benchmark names:

- `ifbench`
- `ifeval`
- `longbench`

## Shared vLLM Parameters

Each benchmark script starts a local OpenAI-compatible vLLM server, waits for
`/v1/models`, runs generation/evaluation, and stops the server on exit.

Default serving parameters:

- `VLLM_PORT=8000`
- `DTYPE=auto`
- `GPU_MEMORY_UTILIZATION=0.8`
- `MAX_MODEL_LEN=150000`
- `MAX_NUM_BATCHED_TOKENS=32768`
- `MAX_NUM_SEQS=16`
- `VLLM_LANGUAGE_MODEL_ONLY=1`
- `VLLM_SLEEP=500`

Tensor parallel defaults:

- `LONGBENCH_TP_SIZE=8`
- `IFBENCH_TP_SIZE=4`
- `IFEVAL_TP_SIZE=4`

Useful overrides:

- `DATA_ROOT=../datasets/data`
- `TOKENIZER_PATH=/path/to/tokenizer`
- `PYTHON_BIN=python`
- `CONDA_ENV=<env-name>`
- `CONDA_SH=/path/to/conda.sh`
- `VLLM_EXTRA_ARGS="..."`
- `VLLM_REASONING_PARSER=qwen3`
- `VLLM_HF_OVERRIDES='{"key": "value"}'`

## IFBench

### Dataset and Evaluation Code

Dataset:

- `../datasets/data/ifbench/IFBench_test.jsonl`

Evaluation code:

- `IFBench/generate_responses.py`
- `IFBench/run_eval.py`
- `IFBench/evaluation_lib.py`
- `IFBench/instructions.py`
- `IFBench/instructions_registry.py`
- `IFBench/instructions_util.py`

Run IFBench only:

```bash
BENCHMARKS=ifbench \
MODEL_PATH=/path/to/Agents-A1 \
MODEL_NAME=Agents-A1 \
bash eval_long.sh --model_name Agents-A1 --model_path /path/to/Agents-A1
```

Outputs:

- Responses: `IFBench/outputs/responses/<MODEL_NAME>.jsonl`
- Evaluation reports: `IFBench/outputs/eval/`

### Inference Parameters

Default generation parameters:

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`
- `max_tokens=4096`
- `seed=42`
- `IFBENCH_WORKERS=64`

Override the input file with `IFBENCH_DATA_FILE=/path/to/IFBench_test.jsonl`.

## IFEval

### Dataset and Evaluation Code

Dataset:

- `../datasets/data/ifeval/input_data.jsonl`

Evaluation code:

- `IFEval/generate_responses.py`
- `IFEval/run_evaluation.py`
- `IFEval/evaluation_lib.py`
- `IFEval/instructions.py`
- `IFEval/instructions_registry.py`
- `IFEval/instructions_util.py`

Run IFEval only:

```bash
BENCHMARKS=ifeval \
MODEL_PATH=/path/to/Agents-A1 \
MODEL_NAME=Agents-A1 \
bash eval_long.sh --model_name Agents-A1 --model_path /path/to/Agents-A1
```

Outputs:

- Responses: `IFEval/outputs/responses/<MODEL_NAME>.jsonl`
- Evaluation reports: `IFEval/outputs/eval/`

### Inference Parameters

Default generation parameters:

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`
- `max_tokens=4096`
- `seed=42`
- `IFEVAL_WORKERS=64`

Override the input file with `IFEVAL_DATA_FILE=/path/to/input_data.jsonl`.

## LongBench-v2

### Dataset and Evaluation Code

Dataset:

- Hugging Face dataset: `THUDM/LongBench-v2`
- Local path: `../datasets/data/longbench_v2`

Prepare data:

```bash
cd LongBench
python prepare_data.py
cd ..
```

Evaluation code:

- `LongBench/pred.py`
- `LongBench/result.py`
- `LongBench/prompts/`
- `LongBench/config/model2maxlen.json`
- `LongBench/config/model2path.json`

Run LongBench only:

```bash
BENCHMARKS=longbench \
MODEL_PATH=/path/to/Agents-A1 \
MODEL_NAME=Agents-A1 \
TOKENIZER_PATH=/path/to/Agents-A1 \
bash eval_long.sh --model_name Agents-A1 --model_path /path/to/Agents-A1
```

Outputs:

- Predictions: `LongBench/results/`
- Summary table: `LongBench/result.txt`

### Inference Parameters

Default LongBench settings:

- `LONGBENCH_RUN_COUNT=3`
- `LONGBENCH_WORKERS=32`
- `LONGBENCH_DATA_DIR=../datasets/data/longbench_v2`
- CoT mode enabled by default
- CoT reasoning call: `max_tokens=1024`
- Final answer call: `max_tokens=128`
- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`

The final reported score averages the configured LongBench runs.

## Runtime Outputs

Runtime artifacts are ignored by `evaluation/IF_CL/.gitignore`:

- vLLM logs
- generated model responses
- benchmark reports
- NLTK resources
- downloaded LongBench-v2 Arrow data
- Python caches
