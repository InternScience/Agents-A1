# Agents-A1 Evaluation on Instruction Following and Context Learning

This document provides a simple framework for evaluating the **Agents-A1** model on the following benchmarks:

- IFbench
- IFeval
- Longbench

## IFBench
### Dataset and Evaluation Code
Use the official IFBench benchmark and evaluation scripts:
https://github.com/allenai/IFBench


### Inference parameters
Use inference parameters aligned with Qwen3.5:

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`

## IFEval
### Dataset and Evaluation Code
Use the official IFEval benchmark and evaluation scripts:
https://github.com/google-research/google-research/tree/master/instruction_following_eval


### Inference parameters
Use inference parameters aligned with Qwen3.5:

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`


## Longbench
### Dataset and Evaluation Code
Use the official LongBench-v2 benchmark and evaluation scripts:
https://github.com/THUDM/LongBench

LongBench-v2 dataset:
https://huggingface.co/datasets/THUDM/LongBench-v2

### Inference parameters
Use inference parameters aligned with Qwen3.5:

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`


We use the chain-of-thought prompting setting, truncate inputs to 128K tokens when necessary, and extract the final answer, reporting accuracy over the full set. 