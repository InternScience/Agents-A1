# Agents-A1 Evaluation on Machine Learning Engineering

This document describes how we evaluate the **Agents-A1** model on machine
learning engineering (MLE) — the task of carrying a competition from raw data to
a scored submission through iterative data inspection, preprocessing, feature
engineering, model training, evaluation, and revision.

Unlike domains defined by short tool calls, MLE has an open-ended solution space
with no closed-form optimum: a candidate is just a program, and its quality only
surfaces after a full train-and-grade cycle. Solving a task is therefore a search
in its own right — propose a solution, run it for a graded score, and let that
feedback steer the next attempt.

> **📌 Evaluation code coming soon.** The agentic harness and tool interface used
> for MLE are part of our long-horizon knowledge-action infrastructure and are
> still being prepared for release. This README documents the benchmark, the
> evaluation protocol, and the tool interface.

## Benchmark

We evaluate on **[MLE-bench Lite](https://github.com/openai/mle-bench)**, the
official lightweight split of OpenAI's MLE-bench, comprising **22 Kaggle
competitions** spanning tabular, vision, NLP, audio, and time-series settings.
Each competition ships with the original data, task description, and a local
grader backed by held-out private answers and the real Kaggle leaderboard.

<details>
<summary>22 MLE-bench Lite competitions</summary>

```
aerial-cactus-identification
aptos2019-blindness-detection
denoising-dirty-documents
detecting-insults-in-social-commentary
dog-breed-identification
dogs-vs-cats-redux-kernels-edition
histopathologic-cancer-detection
jigsaw-toxic-comment-classification-challenge
leaf-classification
mlsp-2013-birds
new-york-city-taxi-fare-prediction
nomad2018-predict-transparent-conductors
plant-pathology-2020-fgvc7
random-acts-of-pizza
ranzcr-clip-catheter-line-classification
siim-isic-melanoma-classification
spooky-author-identification
tabular-playground-series-dec-2021
tabular-playground-series-may-2022
text-normalization-challenge-english-language
text-normalization-challenge-russian-language
the-icml-2013-whale-challenge-right-whale-redux
```

</details>

Download the data and graders following the official
[MLE-bench instructions](https://github.com/openai/mle-bench).

## Evaluation Protocol

- **Compute budget**: each run is given a single **NVIDIA H200** GPU and a
  **12-hour** wall-clock budget. The agent is told its remaining time so it can
  pace exploration, training, and final commitment.
- **Seeds**: every competition is run with **3 seeds**, and we report the
  **average** across seeds.
- **Grading**: submissions are scored by the official MLE-bench local graders
  against held-out private answers, then mapped onto the corresponding Kaggle
  leaderboard.

### Metrics

| Metric | Definition |
|--------|------------|
| **Medal rate** | Fraction of tasks where the committed submission earns a Kaggle medal (gold / silver / bronze) under the official thresholds. |

### Inference Parameters

Sampling parameters are aligned with Qwen3.5:

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`

## Agentic Harness

Trajectories are produced by an agentic harness that frames optimization as a
growing **tree of solution nodes**. Each node is a candidate solution: writing a
full script opens a new root (a fresh line of attack), patching a node spawns a
child, and executing a node attaches its observations — stdout, exceptions, the
validation metric, artifacts, and submission validity.

On long runs, an isolated `analyze` sub-agent investigates data or results in its
own context and returns a single structured report, while context compaction
folds earlier steps into a digest so the agent can run the full 12-hour horizon
without losing the thread.

### Tool Interface

The harness exposes a compact action space spanning code authoring, execution,
search-tree navigation, answer management, persistent memory, and delegated
analysis.

| Tool | Function |
|------|----------|
| **Code authoring & execution** | |
| `write_full_code` | Author a complete training script from scratch; opens a new root node (a fresh line of attack). |
| `patch_code` | Apply a localized edit to a node's code; spawns a child node, preserving tree history for incremental refinement. |
| `execute_code` | Run a node, capture stdout and exceptions, extract its validation metric, and check the emitted submission for validity. |
| `execute_bash` | Run a guarded shell command for environment setup and inspection (installs, GPU checks, file operations). |
| **Search-tree navigation & answer management** | |
| `list_nodes` | Survey the solution tree: the selected answer, the recent answer trail, invalidated history, and a metric-ranked listing. |
| `select_node` | Inspect one node in full (code, plan, output, metric, parent chain) before revisiting or branching from it. |
| `invalidate_node` | Exclude a node whose metric is untrustworthy (leakage, overfitting) from ranking and submission. |
| `update_answer` | Commit a node as the current submission candidate, written to the canonical path the grader reads. |
| `get_current_answer` | Report the node currently committed as the answer. |
| **Persistent memory** | |
| `write_notes` / `read_notes` | Append to / re-read a notebook that survives context compaction (decisions, failed strategies and why, hypotheses). |
| **Sub-agent** | |
| `analyze` | Spawn an isolated analysis sub-agent that explores data and results in its own context window and returns a single structured report. |

