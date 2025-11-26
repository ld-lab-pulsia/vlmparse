# %%
"""Create a HuggingFace dataset from the benchmark folder structure."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset


def load_data_from_folder(
    base_folder: Path,
) -> List[Dict[str, Any]]:
    """Load all data from the folder structure.
    One row per test with relative PDF path.

    Args:
        base_folder: Path to the folder containing benchmark data
    """
    data = []

    for subdir in sorted(base_folder.iterdir()):
        if not subdir.is_dir():
            continue

        metadata_path = subdir / "metadata.json"
        tests_path = subdir / "tests.jsonl"
        pdf_path = [p for p in subdir.glob("*.pdf")]
        assert len(pdf_path) == 1, f"Expected 1 PDF file, got {len(pdf_path)}"
        pdf_path = pdf_path[0]

        if not all([metadata_path.exists(), tests_path.exists(), pdf_path.exists()]):
            print(f"Skipping {subdir.name}: missing files")
            continue

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load tests
        tests = []
        with open(tests_path, "r") as f:
            for line in f:
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

    return data


def create_dataset(base_folder: str, output_path: str = None, push_to_hub: str = None):
    """Create HuggingFace dataset from folder structure.
    One row per test with relative PDF path.

    Args:
        base_folder: Path to folder containing benchmark data
        output_path: Local path to save dataset (optional)
        push_to_hub: HuggingFace Hub repository name to push to (optional)
    """
    base_path = Path(base_folder) / "pdfs"

    print(f"Loading data from {base_path}...")
    data = []
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
        data.extend(load_data_from_folder(subdir))

    print(f"Loaded {len(data)} tests")

    # Create dataset - let it infer features automatically
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    print(f"\nDataset created with {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")

    if output_path:
        print(f"\nSaving dataset to {output_path}...")
        dataset.save_to_disk(output_path)
        print("Dataset saved!")

    if push_to_hub:
        print(f"\nPushing dataset to HuggingFace Hub: {push_to_hub}...")
        dataset.push_to_hub(push_to_hub)
        print("Dataset pushed to Hub!")

    return dataset


# from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument("--folder_path", type=str, default="/home/brigal/data/fr-bench-pdf2md/")
# args = parser.parse_args()

# dataset = create_dataset(args.folder_path)
# dataset.to_parquet("dataset.parquet")
