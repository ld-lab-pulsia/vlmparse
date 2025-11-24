# %%
"""Create a HuggingFace dataset from the benchmark folder structure."""

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset


def load_data_from_folder(base_folder: Path) -> List[Dict[str, Any]]:
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

        # Get relative path to PDF from base_folder
        pdf_relative_path = str(pdf_path.relative_to(base_folder))

        # Create one row per test
        for test in tests:
            row = {
                "pdf_name": metadata["pdf"],
                "page": metadata["page"],
                "doc_type": metadata.get("doc_type"),
                "original_doc_path": metadata.get("original_doc_path"),
                "pdf_path": pdf_relative_path,
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
    base_path = Path(base_folder)

    print(f"Loading data from {base_path}...")
    data = load_data_from_folder(base_path)

    print(f"Loaded {len(data)} tests")

    # Create dataset - let it infer features automatically
    dataset = Dataset.from_list(data)

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


# ds = create_dataset(Path("/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/validated_tests/tiny_test_tests_first_batch/tests/tiny_text_long_text/"))

# # %%
# ds
# # ds.to_parquet("ds.pq")
# from datasets.arrow_dataset import Dataset

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Create HuggingFace dataset from benchmark folder")
#     parser.add_argument(
#         "folder",
#         type=str,
#         help="Path to the folder containing the benchmark data"
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         default=None,
#         help="Output path to save the dataset locally (optional)"
#     )
#     parser.add_argument(
#         "--push-to-hub",
#         type=str,
#         default=None,
#         help="HuggingFace Hub repository name to push to (optional)"
#     )

#     args = parser.parse_args()

#     dataset = create_dataset(args.folder, args.output, args.push_to_hub)

#     # Print some examples
#     print("\n" + "="*80)
#     print("Example from dataset:")
#     print("="*80)
#     example = dataset[0]
#     print(f"PDF: {example['pdf_name']}")
#     print(f"Page: {example['page']}")
#     print(f"Doc type: {example['doc_type']}")
#     print(f"PDF path: {example['pdf_path']}")
#     print(f"Test type: {example['type']}")
#     print(f"\nFull test:")
#     print(json.dumps(example, indent=2, ensure_ascii=False))
