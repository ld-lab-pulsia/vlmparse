import asyncio
import json
import os
import random
from pathlib import Path

import pypdfium2 as pdfium
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import (
    TableTest,
    TextPresenceTest,
    save_tests,
)
from vlmparse.build_doc import convert_pdfium_to_images
from vlmparse.utils import to_base64

DATA_PROJECT = Path("/data_project")


class HandwritingTableTest(BaseModel):
    """Test for text in a table cell with optional relationships."""

    cell: str = Field(description="The cell content to find")
    up: str = Field(default="", description="Expected cell content above")
    down: str = Field(default="", description="Expected cell content below")
    left: str = Field(default="", description="Expected cell content to the left")
    right: str = Field(default="", description="Expected cell content to the right")
    top_heading: str = Field(default="", description="Expected column header")
    left_heading: str = Field(default="", description="Expected row header")


class HandwritingPresenceTest(BaseModel):
    """Test for presence of handwritten text in the document."""

    text: str = Field(description="Handwritten text to find", min_length=1)


class HandwritingTestsResponse(BaseModel):
    """Response containing handwriting tests (either table or presence tests)."""

    handwriting_tests: list[HandwritingPresenceTest] = Field(
        description="List of 1 to 10 handwriting tests.",
        min_length=0,
        max_length=10,
    )
    table_tests: list[HandwritingTableTest] = Field(
        description="List of table tests for the page focusing on handwritten text in tables.",
        min_length=0,
        max_length=5,
    )


PROMPT = """Analyze this form page and identify handwritten text that should be tested for OCR accuracy.

Focus on:
- Text that contains handwritten text (never only printed text)
- Minimum 1 word per test
- Handwritten check marks count as handwritten text.

Generate tests in one of two formats:

1. **HandwritingTableTest**: Use when handwritten text appears in a table/form with clear structure
   - Specify the handwritten cell content
   - Optionally specify relationships (up, down, left, right cells)
   - Optionally specify headers (top_heading, left_heading)

    Note:
    - When cells are spanning multiple rows or columns, any cell adjacent to a multiple row or column cell should be counted as correct.
    - When a cell is adjacent to a cell that is spanning multiple rows or columns, it should be counted as correct.
    - The row header is defined as the first column in the table independent of the appearance of the table.


2. **HandwritingPresenceTest**: Use for handwritten text outside tables or when structure is unclear
   - Just specify the exact handwritten text to find.
   - The text can also be a combination of handwritten and printed text provided it contains handwritten text.

Additional instructions:
- Return between 0 and 10 table tests and between 1 and 10 handwriting tests depending on how much handwritten content is present.
- When you see any handwritten text, you should always create at least one test.
- Extract text exactly as written (including any grammatical mistakes)."""


async def generate_tests_for_page(
    client,
    image,
    pdf_name: str,
    page_num: int,
    model: str = "gemini-3-pro-preview",
):
    """Generate handwriting tests for a single page."""
    img_b64 = to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ],
        }
    ]

    response = await client.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=HandwritingTestsResponse,
    )

    tests_response = response.choices[0].message.parsed
    if tests_response is None:
        return None

    # Convert to actual test objects
    handwriting_tests = []
    for idx, test_data in enumerate(tests_response.handwriting_tests):
        test = TextPresenceTest(
            category="handwriting",
            pdf=pdf_name,
            page=page_num,
            id=f"{pdf_name}_p{page_num}_text{idx}",
            type="present",
            text=test_data.text,
            max_diffs=2,
            unidecode=True,
        )
        handwriting_tests.append(test)
    table_tests = []
    for idx, test_data in enumerate(tests_response.table_tests):
        # Create TableTest
        test = TableTest(
            category="handwriting",
            pdf=pdf_name,
            page=page_num,
            id=f"{pdf_name}_p{page_num}_table{idx}",
            type="table",
            cell=test_data.cell,
            up=test_data.up,
            down=test_data.down,
            left=test_data.left,
            right=test_data.right,
            top_heading=test_data.top_heading,
            left_heading=test_data.left_heading,
            max_diffs=2,
            unidecode=True,
        )
        table_tests.append(test)
    return handwriting_tests, table_tests


def save_pdf_page(pdf_path: Path, page_num: int, output_path: Path):
    """Save a single page from a PDF to a new PDF file."""
    pdf = pdfium.PdfDocument(pdf_path)
    new_pdf = pdfium.PdfDocument.new()
    new_pdf.import_pages(pdf, [page_num])
    new_pdf.save(output_path)
    new_pdf.close()
    pdf.close()


async def main(
    gt_folder: Path,
    save_folder: Path,
    start_idx: int = 0,
    end_idx: int = 100,
    model: str = "gemini-2.5-pro",
    add_page_num: bool = False,
):
    """Generate handwriting tests for PDF forms."""
    async_client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=500,
    )
    files = list(gt_folder.rglob("*.pdf"))
    random.seed(42)
    random.shuffle(files)

    for file in tqdm(files[start_idx:end_idx]):
        images = convert_pdfium_to_images(file, dpi=250)
        for page_num, image in enumerate(images):
            if add_page_num:
                page_folder = save_folder / f"{file.stem}_page{page_num}"
            else:
                page_folder = save_folder / f"{file.stem}"

            pdf_page_path = page_folder / (page_folder.name + ".pdf")
            test_name = "tests_handwriting.jsonl"
            output_file = page_folder / test_name

            test_name_table = "tests_table.jsonl"
            output_file_table = page_folder / test_name_table
            if output_file_table.exists() or output_file.exists():
                continue

            print(f"Generating tests for {file.name} page {page_num}")
            handwriting_tests, table_tests = await generate_tests_for_page(
                async_client,
                image,
                file.name,
                page_num,
                model=model,
            )

            if (
                handwriting_tests is None
                or len(handwriting_tests) == 0
                or table_tests is None
                or len(table_tests) == 0
            ):
                print(
                    f"No handwriting or table tests found for {file.name} page {page_num}"
                )
                continue

            page_folder.mkdir(parents=True, exist_ok=True)
            save_tests(handwriting_tests, str(output_file))
            save_tests(table_tests, str(output_file_table))

            if not pdf_page_path.exists():
                save_pdf_page(file, page_num, pdf_page_path)

            metadata_path = page_folder / "metadata.json"
            if not metadata_path.exists():
                metadata = {
                    "original_doc_path": str(file.absolute()),
                    "pdf": file.name,
                    "page": page_num,
                    "doc_type": "handwritten_form",
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            print(f"Saved {len(handwriting_tests)} handwriting tests to {output_file}")
            print(f"Saved {len(table_tests)} table tests to {output_file_table}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_folder",
        type=str,
        default=os.path.expanduser("~/data/bench_data/handwritten_forms/"),
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=os.path.expanduser("~/data/bench_data/tests/"),
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview")

    args = parser.parse_args()

    asyncio.run(
        main(
            Path(args.gt_folder),
            Path(args.save_folder),
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            model=args.model,
            add_page_num=True,
        )
    )
