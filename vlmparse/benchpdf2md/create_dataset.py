# %%
"""Create a HuggingFace dataset from the benchmark folder structure."""

import json
from pathlib import Path

import pandas as pd


def create_dataset(
    base_folder: Path,
) -> pd.DataFrame:
    """Load all data from the folder structure.
    One row per test with relative PDF path.

    Args:
        base_folder: Path to the folder containing benchmark data
    """
    data = []

    for subdir in sorted(Path(base_folder).rglob("**/")):
        if not subdir.is_dir() or not len(list(subdir.glob("*.jsonl"))) >= 1:
            continue

        metadata_path = subdir / "metadata.json"
        tests_paths = list(subdir.glob("tests*.jsonl"))
        pdf_path = [p for p in subdir.glob("*.pdf")]
        assert len(pdf_path) == 1, f"Expected 1 PDF file, got {len(pdf_path)}"
        pdf_path = pdf_path[0]

        if not all([metadata_path.exists(), pdf_path.exists(), len(tests_paths) > 0]):
            print(f"Skipping {subdir.name}: missing files")
            continue

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load tests
        tests = []
        for tests_path in subdir.glob("*.jsonl"):
            with open(tests_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    tests.append(json.loads(line.strip()))

        # Create one row per test
        for test in tests:
            row = {
                "pdf_name": metadata["pdf"],
                "page": metadata["page"],
                "doc_type": metadata.get("doc_type"),
                "original_doc_path": metadata.get("original_doc_path"),
                "pdf_path": str(pdf_path),
                **test,  # Unpack all test fields
            }
            data.append(row)

    return pd.DataFrame(data)
