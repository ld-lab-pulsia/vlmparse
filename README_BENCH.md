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

## Procedure of benchmark creation

### Opinionated choices

- We focused on french documents.
- We did not include mathematical equations in the benchmark as these are language agnostic and not specific to french documents.
- We focused on difficult pages, such that the benchmark is difficult even for state of the art VLMs.
- We reduced strictness of the tests compared to the Olmocr benchmark to ensure that failure indicates a real problem with the transcription.

### Document collection
We collected ~60000 french documents from the CCPDF dataset. Then we selected the most difficult pages to create the benchmark by doing the transcription with two VLMs and comparing the results (the largest edit distances were considered as the most difficult pages).

This led us to select these categories of pages:
- Pages with tiny text (the OCR is harder at low resolution)
- Pages with multiple columns (the flow from one column to the next is not always respected)
- Pages with long tables (long tables are still difficult even for state of the art VLMs)
- Pages with manuscript text:
  - Some pages were downloaded from Gallica
  - Some pages were generated from French forms (such as CERFA): a filled form was generated from en empty one using the OpenAI image generation API.

### Test generation
Different catagories of tests were generated with prompts specifically adapted to each category (using the scripts in the `scripts/generation_scripts` folder).

The tests were then manually reviewed and edited by a human annotator using the Streamlit app in (`vlmparse/benchpdf2md/st_visu_benchmark/app.py`).


## Credits

This benchmark is inspired by and adapted from the [AllenAI OLMo OCR benchmark](https://github.com/allenai/olmocr/tree/main/olmocr). The test framework, normalization logic, and evaluation methodology draw heavily from their excellent work on document parsing evaluation.
