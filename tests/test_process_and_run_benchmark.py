import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from vlmparse.benchpdf2md.run_benchmark import process_and_run_benchmark


@pytest.fixture
def test_data_dir(datadir):
    """Create a temporary test directory with fake benchmark data."""
    tmpdir = Path(tempfile.mkdtemp())

    # Create a subdirectory following the expected structure
    test_subdir = tmpdir / "pdfs" / "random_test" / "Fiche_Graines_A5_page1"
    test_subdir.mkdir(parents=True)

    # Copy the PDF file
    pdf_source = datadir / "Fiche_Graines_A5.pdf"
    pdf_dest = test_subdir / "Fiche_Graines_A5.pdf"

    shutil.copy(pdf_source, pdf_dest)

    # Create metadata.json
    metadata = {
        "pdf": "Fiche_Graines_A5.pdf",
        "page": 1,
        "doc_type": "test_doc",
        "original_doc_path": str(pdf_source),
    }
    with open(test_subdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create tests.jsonl with fake tests
    tests = [
        {
            "pdf": "Fiche_Graines_A5.pdf",
            "id": "test_title_present",
            "type": "present",
            "text": "Fiche",
            "case_sensitive": False,
        },
        {
            "pdf": "Fiche_Graines_A5.pdf",
            "id": "test_graines_present",
            "type": "present",
            "text": "Graines",
            "case_sensitive": False,
        },
        {
            "pdf": "Fiche_Graines_A5.pdf",
            "id": "test_fake_absent",
            "type": "absent",
            "text": "ThisShouldNotBeInTheDocument12345",
            "case_sensitive": False,
        },
    ]

    with open(test_subdir / "tests.jsonl", "w") as f:
        for test in tests:
            f.write(json.dumps(test) + "\n")

    yield tmpdir

    # Cleanup
    shutil.rmtree(tmpdir)


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    # Cleanup
    shutil.rmtree(tmpdir)


def test_process_and_run_benchmark_end2end(test_data_dir, output_dir):
    """End-to-end test for process_and_run_benchmark with real model call."""

    assert os.getenv("GOOGLE_API_KEY") is not None, "GOOGLE_API_KEY is not set"
    # Run the benchmark with the default model
    process_and_run_benchmark(
        model="gemini-2.5-flash-lite",
        uri=None,
        concurrency=1,
        debug=False,
        in_folder=test_data_dir,
        save_folder=output_dir,
    )

    # Verify that outputs were created
    model_dirs = list(output_dir.glob("gemini-2.5-flash-lite/**/results"))
    assert len(model_dirs) >= 1, "Expected at least one timestamp directory"

    latest_run = sorted(model_dirs)[-1]

    # Check that results directory exists
    results_dir = latest_run
    assert results_dir.exists(), "Results directory should exist"

    # Check that zip file was created
    zip_files = list(results_dir.rglob("*.zip"))
    assert len(zip_files) >= 1, "Expected at least one zip file"

    # metrics_results = list(latest_run.parent.glob("test_results/**")).glob("metrics.json"))

    # Check that test_results.parquet was created
    test_results = list(latest_run.parent.rglob("test_results.parquet"))
    assert len(test_results) == 1, "test_results.parquet should exist"
    test_results = test_results[0]

    # Check that by_type.xlsx was created
    by_type_xlsx = list(latest_run.parent.rglob("by_type.xlsx"))
    assert len(by_type_xlsx) == 1, "by_type.xlsx should exist"
    by_type_xlsx = by_type_xlsx[0]
    assert by_type_xlsx.exists(), "by_type.xlsx should exist"

    # Load and verify results
    import pandas as pd

    df = pd.read_parquet(test_results)

    # Should have our 3 tests + 1 baseline test = 4 total
    assert len(df) >= 4, f"Expected at least 4 test results, got {len(df)}"

    # Check that test types are present
    test_types = set(df["type"].unique())
    assert "present" in test_types, "Should have 'present' test type"
    assert "absent" in test_types, "Should have 'absent' test type"
    assert "baseline" in test_types, "Should have 'baseline' test type"

    # Check that test IDs are present
    test_ids = set(df["test_id"].unique())
    assert "test_title_present" in test_ids
    assert "test_graines_present" in test_ids
    assert "test_fake_absent" in test_ids

    # Verify that at least some tests passed
    # (we expect present tests to pass if PDF parsing worked)
    assert df["result"].sum() >= 1, "Expected at least one test to pass"
