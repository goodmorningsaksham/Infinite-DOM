"""
Upload oracle training data to HuggingFace Hub so the Colab notebook
can download it without needing local environment access.

Usage:
    python training/upload_oracle_data.py YOUR_USERNAME/infinite-dom-data

Requires:
    HF_TOKEN environment variable or `huggingface-cli login`
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DATA_PATH = Path("training/data/oracle_trajectories.jsonl")


def upload(repo_id: str) -> None:
    from huggingface_hub import HfApi

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run generate_oracle_data.py first.")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(DATA_PATH),
        path_in_repo="oracle_trajectories.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )

    size_mb = DATA_PATH.stat().st_size / (1024 * 1024)
    print(f"Uploaded {DATA_PATH} ({size_mb:.1f} MB) to {repo_id}")
    print(f"Download in Colab with:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f'  path = hf_hub_download("{repo_id}", "oracle_trajectories.jsonl", repo_type="dataset")')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python training/upload_oracle_data.py YOUR_USERNAME/infinite-dom-data")
        sys.exit(1)
    upload(sys.argv[1])
