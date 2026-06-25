# LongBench-v2 Runner

This folder contains the LongBench-v2 runner adapted for Agents-A1.

Before running, download the LongBench-v2 dataset locally:

```bash
python prepare_data.py
```

The dataset is saved to `../../datasets/data/longbench_v2` by default. The
downloaded Arrow data is intentionally not committed because it is larger than
GitHub's normal file limit.

Run through the repository entrypoint:

```bash
cd ..
MODEL_PATH=/path/to/hf/model MODEL_NAME=my-model bash eval_long.sh --model_name my-model --model_path /path/to/hf/model
```

The original LongBench-v2 project and dataset are available at:

- https://longbench2.github.io
- https://huggingface.co/datasets/THUDM/LongBench-v2
