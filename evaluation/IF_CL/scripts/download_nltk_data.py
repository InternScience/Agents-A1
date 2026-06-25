#!/usr/bin/env python3
"""Download NLTK resources used by IFBench and IFEval."""

from pathlib import Path

import nltk


RESOURCES = [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
]


def ensure_resources(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for package, resource_path in RESOURCES:
        try:
            nltk.data.find(resource_path, paths=[str(target)])
        except LookupError:
            nltk.download(package, download_dir=str(target))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for benchmark in ("IFBench", "IFEval"):
        target = repo_root / benchmark / ".nltk_data"
        print(f"Checking NLTK resources in {target}")
        ensure_resources(target)


if __name__ == "__main__":
    main()
