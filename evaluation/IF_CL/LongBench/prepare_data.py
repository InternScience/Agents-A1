"""Download and save LongBench-v2 dataset to disk for offline use."""
import argparse
from pathlib import Path

from datasets import load_dataset

if __name__ == "__main__":
    default_save_path = Path(__file__).resolve().parents[2] / "datasets/data/longbench_v2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="THUDM/LongBench-v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--save-path", default=str(default_save_path))
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    dataset.save_to_disk(args.save_path)
    print(f"Saved {len(dataset)} examples to {args.save_path}")
