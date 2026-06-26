# Benchmark datasets

1. GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
2. BrowseComp: https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
3. SEAL: https://huggingface.co/datasets/vtllms/sealqa/viewer/seal_0
4. Xbench: https://huggingface.co/datasets/xbench/DeepSearch
5. BrowseComp-ZH: https://github.com/PALIN2018/BrowseComp-ZH

We also provide a example dataset in `./datasets/data/example/standardized_data.jsonl` for demonstration. You will need to convert the official benchmark datasets into the standardized format for evaluation. 

# Dataset Format

The standardized format is a JSONL file with each record containing `question` and `answer` fields:

```json
{"question": "Who are you?", "answer": "I am Agents-A1"}
```

If there are additional files, such as images or videos, you can include them in the same directory and reference them in a `file_name` field:

```json
{"question": "Who are you?", "answer": "I am Agents-A1", "file_name": "images/logo.png"}
```

Note that the Seal-0 dataset contains effective years for each question that might affect the final answer. Consequently, we recommend including the `effective_year` field in the standardized format for Seal-0. For example:

```json
<ORIGINAL QUESTION>

Please note:
1. The current year is <CURRENT YEAR>.
2. This question was created on <EFFECTIVE_YEAR>.
3. You must answer based on the facts that were correct at the time the question was created, not the current date.

If the question contains incorrect facts or false assumptions, do not follow them. Instead, identify the mistake and provide the correct answer based on real-world facts.
```
