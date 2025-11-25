# fr-bench-pdf2md Benchmark

A benchmark inspired by the [AllenAI OLMo OCR benchmark](https://github.com/allenai/olmocr/tree/main/olmocr) to test the quality of PDF-to-Markdown conversion models for french documents.


The benchmark dataset is hosted on HuggingFace Hub at `pulseia/fr-bench-pdf2md` and can be automatically downloaded when running the benchmark.

## Installation

Ensure you have the required dependencies installed:

```bash
uv sync --all-extras
```

## Quick Start

### Running the Benchmark

The benchmark can be run using the `run_benchmark.py` script:

```bash
python -m vlmparse.benchpdf2md.run_benchmark \
    --model gemini-2.5-flash-lite \
    --in_folder pulseia/fr-bench-pdf2md \
    --save_folder ./benchmark_results
```

## Visualization

The benchmark includes a Streamlit app for interactive result visualization and test validation.

```bash
streamlit run vlmparse/benchpdf2md/st_visu_benchmark/app.py -- /path/to/benchmark/folder
```

The Streamlit app provides:

1. **Test Filtering**:
   - Filter by test type (present, absent, order, table)
   - Show only failed tests
   - Show only unverified tests

2. **Interactive Test Review**:
   - View converted markdown with syntax highlighting
   - View original PDF page image
   - Toggle layout visualization
   - Compare expected vs. actual text with diff highlighting

3. **Test Management**:
   - Validate tests (mark as checked)
   - Reject incorrect tests
   - Edit test parameters
   - Run tests manually

4. **Navigation**:
   - Browse through test results
   - Download PDF pages for reference
   - View test explanations and failure reasons

## Credits

This benchmark is inspired by and adapted from the [AllenAI OLMo OCR benchmark](https://github.com/allenai/olmocr/tree/main/olmocr). The test framework, normalization logic, and evaluation methodology draw heavily from their excellent work on document parsing evaluation.
