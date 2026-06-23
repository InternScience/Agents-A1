# Evaluation Framework

Here contains a deep-research agent evaluation framework. It runs ReAct (Reasoning + Acting) agents against benchmark datasets using multi-turn LLM conversations with tool use, then evaluates response quality via LLM-based judging.

## Supported Benchmarks

- **GAIA** — General AI Assistant benchmark
- **BrowseComp** — Web browsing comprehension (English & Chinese)
- **SEAL** — Search-augmented evaluation
- **Xbench** — Cross-lingual deep search
- **HLE** — Humanity's Last Exam

## Quick Start

### 1. Environment Setup

```bash
conda create -n agenteval python=3.10.0
conda activate agenteval
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required API keys:
| Key | Service | Purpose |
|-----|---------|---------|
| `SERPER_KEY_ID` | [Serper](https://serper.dev/) | Web search and Google Scholar |
| `JINA_API_KEYS` | [Jina](https://jina.ai/) | Web page content extraction |
| `SANDBOX_FUSION_ENDPOINT` | [SandboxFusion](https://github.com/bytedance/SandboxFusion) | Sandboxed Python code execution |
| `AGENT_API_KEY` / `AGENT_API_BASE_URL` | Any OpenAI-compatible API | Agent inference (remote mode) |
| `SUMMARY_API_KEY` / `SUMMARY_API_BASE` | Any OpenAI-compatible API | Page content summarization |
| `JUDGE_API_KEY` / `JUDGE_API_BASE` | Any OpenAI-compatible API | LLM-based evaluation judge |

### 3. Prepare Data

The system supports JSONL (recommended) and JSON formats. Each record needs `question` and `answer` fields:

```json
{"question": "What is the capital of France?", "answer": "Paris"}
```

Pre-processed benchmark datasets are available at:
- [Google Drive](https://drive.google.com/file/d/1UJmJPFwh0HYnBpvlrCB5Uzaxm-OqnM7e/view?usp=sharing)
- [Baidu Drive](https://pan.baidu.com/s/1dD1U6TSkHDV2EpUMQRssCg?pwd=jiqn)

## Running Inference

All inference goes through `inference/run_inference.sh`, which loads `.env` and handles both local and remote modes:

```bash
cd inference
bash run_inference.sh <model_name> <model_path> <dataset_path>
```

Or use the full pipeline (inference + evaluation):

```bash
bash run.sh <model_name> <model_path_or_name> [dataset_path]
```

### Local vLLM Mode (default)

When `AGENT_API_BASE_URL` is **not** set in `.env`, the script starts a local vLLM server, waits for readiness, then runs inference.

```bash
bash run_inference.sh  <model_name> /path/to/model /path/to/dataset.jsonl
```

**GPU Configuration**: Edit `inference/run_inference.sh` to adjust:
- `CUDA_VISIBLE_DEVICES` — which GPUs to use
- `--tensor-parallel-size` — number of GPUs per model instance
- `--port` — server port

For example, with 30B-A3B models, 2 GPUs per instance — duplicate the `vllm serve` block with different ports:
```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve $MODEL_PATH --port 6001 --tensor-parallel-size 2 ...
CUDA_VISIBLE_DEVICES=2,3 vllm serve $MODEL_PATH --port 6002 --tensor-parallel-size 2 ...
```

### Remote API Mode

When `AGENT_API_BASE_URL` is set in `.env`, the script skips vLLM startup and sends requests to the remote endpoint directly.

Set these in `.env`:
```
AGENT_API_KEY=your-api-key
AGENT_API_BASE_URL=https://openrouter.ai/api/v1
```

Then run the same command with the model name instead of a local path:
```bash
bash run_inference.sh your-model-name your-model-path /path/to/dataset.jsonl
```

## Running Evaluation

```bash
# Or set these in .env
export JUDGE_API_KEY=your_key
export JUDGE_API_BASE=https://api.openai.com/v1
# Optional: override the default judge selected for each dataset
export JUDGE_MODEL_NAME=gpt-4o

cd evaluation
python evaluate.py \
    --input-folder /path/to/results/ \
    --dataset gaia
```

Evaluation requires 3 iteration files (`iter1.jsonl`, `iter2.jsonl`, `iter3.jsonl`). For single-rollout evaluation:
```bash
cp iter1.jsonl iter2.jsonl && cp iter1.jsonl iter3.jsonl
```

Override the default judge model by setting `JUDGE_MODEL_NAME` in `.env`.

See [evaluation/README.md](./evaluation/README.md) for more details.

## Architecture

### Inference Pipeline

`run_inference.py` is the orchestrator: loads datasets, fans out questions to a `ThreadPoolExecutor`, each thread runs a `MultiTurnReactAgent` instance.

The agent runs a multi-turn loop: call LLM -> parse tool calls -> execute tool -> append result -> repeat. Final answers are extracted from `<answer></answer>` tags. 150-minute timeout per question, max 100 LLM calls (configurable via `MAX_LLM_CALL_PER_RUN`).

### Tools

| Tool | File | External Service |
|------|------|-----------------|
| Web Search | `tool_search.py` | Serper API |
| Web Read | `tool_visit.py` | Jina API + summarization LLM |
| Python Interpreter | `tool_python.py` | SandboxFusion server |

The Python Interpreter (`PythonInterpreter`) lets the agent run Python in a
sandbox for precise computation and data processing. It requires a running
[SandboxFusion](https://github.com/bytedance/SandboxFusion) server; point
`SANDBOX_FUSION_ENDPOINT` at it (a comma-separated list of server URLs is
supported for load balancing). When the endpoint is unreachable or unset, the
tool returns a `[Python Interpreter Error]` string and the run continues with
the other tools.

### Results Format

Inference outputs JSONL with fields: `question`, `answer` (ground truth), `messages` (full conversation), `prediction`, `termination` (answer/exceed_calls/timeout/error).

## License

Uses the open-source [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) framework for tool-calling and message schemas.
