# Agents-A1 Evaluation on Scientific Research

This document provides a simple framework for evaluating the **Agents-A1** model on the following benchmarks:

- HLE w/ tools
- HiPhO
- FS-O (FrontierScience-Olympiad)
- FS-R (FrontierScience-Research)

## HLE w/ tools

### Dataset

Use the official HLE benchmark:
https://huggingface.co/datasets/cais/hle

### Evaluation Setting

For baseline models, report official scores whenever available. For Qwen3.6-35B-A3B, which does not yet have an official result for HLE w/ tools, evaluate it with the same tool-augmented pipeline used for Agents-A1, including search, visit, code, and scholar tools.

Use inference parameters aligned with Qwen3.5:

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`

## HiPhO

### Dataset

Use the official HiPhO benchmark:
https://huggingface.co/datasets/SciYu/HiPhO

### Evaluation Setting

Use inference parameters aligned with the official evaluation setting:

- `temperature=0.6`
- `top_p=0.95`

## FS-O (FrontierScience-Olympiad)

### Dataset

Use the official FrontierScience benchmark:
https://huggingface.co/datasets/openai/frontierscience

### Evaluation Setting

Use the unified inference parameters:

- `temperature=0.6`
- `top_p=0.95`

## FS-R (FrontierScience-Research)

### Dataset

Use the official FrontierScience benchmark:
https://huggingface.co/datasets/openai/frontierscience

### Evaluation Setting

Use the unified inference parameters:

- `temperature=0.6`
- `top_p=0.95`


Note: For HiPhO, FS-O, and FS-R, we evaluate all models using a unified set of inference hyperparameters. For Agents-A1, we apply the same tool-augmented reasoning framework across all four scientific research benchmarks, with access to four standardized tool types: search, visit, code, and scholar tools.
