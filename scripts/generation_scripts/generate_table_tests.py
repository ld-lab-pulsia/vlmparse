# %%
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import TableTest, save_tests
from vlmparse.build_doc import convert_pdfium_to_images
from vlmparse.utils import to_base64

DATA_PROJECT = Path("/data_project")


class TableTestSpec(BaseModel):
    cell: str = Field(description="The target cell value that must exist in the table")
    up: Optional[str] = Field(
        default="", description="Cell immediately above the target cell"
    )
    down: Optional[str] = Field(
        default="", description="Cell immediately below the target cell"
    )
    left: Optional[str] = Field(
        default="", description="Cell immediately to the left of the target cell"
    )
    right: Optional[str] = Field(
        default="", description="Cell immediately to the right of the target cell"
    )
    top_heading: Optional[str] = Field(
        default="", description="Column header (cell all the way up)"
    )
    left_heading: Optional[str] = Field(
        default="", description="Row header (cell all the way left)"
    )


class TableTestsResponse(BaseModel):
    html_table: str = Field(description="The HTML table as a string")
    tests: list[TableTestSpec] = Field(
        description="List of exactly 3 table tests for this page",
        min_length=2,
        max_length=6,
    )


PROMPT = """Analyze the table(s) in this image, select one table and parse it to html then generate between 2 and 6 test cases to verify correct table parsing of the selected table.

For each test, identify:
- A target cell value
- Its relationships to adjacent cells (up/down/left/right), adjacent cells can be absent, then return empty string
- Its column header (top_heading) and row header (left_heading), these can be absent, then return empty string

Note:
- When cells are spanning multiple rows or columns, any cell adjacent to a multiple row or column cell should be counted as correct.
- When a cell is adjacent to a cell that is spanning multiple rows or columns, it should be counted as correct.
- The row header is defined as the first column in the table independent of the appearance of the table.

Focus on:
1. Important data cells with clear relationships
2. Cells that test different table structures (headers, data cells)
3. Cells that verify table boundaries and spanning

Return between 2 and 6 distinct tests."""


async def generate_tests_for_page(client, image, pdf_name: str, page_num: int):
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
        model="gemini-2.5-pro",
        messages=messages,
        response_format=TableTestsResponse,
    )

    tests_response = response.choices[0].message.parsed
    if tests_response is None:
        return []
    tests = []
    for idx, test_spec in enumerate(tests_response.tests):
        test = TableTest(
            pdf=pdf_name,
            page=page_num,
            id=f"{pdf_name}_p{page_num}_t{idx}",
            type="table",
            cell=test_spec.cell,
            up=test_spec.up or "",
            down=test_spec.down or "",
            left=test_spec.left or "",
            right=test_spec.right or "",
            top_heading=test_spec.top_heading or "",
            left_heading=test_spec.left_heading or "",
            max_diffs=0,
            category="long_table",
        )
        tests.append(test)
    return tests


def save_pdf_page(pdf_path: Path, page_num: int, output_path: Path):
    pdf = pdfium.PdfDocument(pdf_path)
    new_pdf = pdfium.PdfDocument.new()
    new_pdf.import_pages(pdf, [page_num])
    new_pdf.save(output_path)
    new_pdf.close()
    pdf.close()


async def main(
    gt_folder: Path, save_folder: Path, start_idx: int = 300, end_idx: int = 400
):
    async_client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=500,
    )

    files = [f for f in gt_folder.iterdir() if f.is_file() and f.name.endswith(".pdf")]
    import random

    random.seed(42)
    random.shuffle(files)
    for file in files[start_idx:end_idx]:
        images = convert_pdfium_to_images(file, dpi=250)
        for page_num, image in enumerate(images):
            print(f"Generating tests for {file.name} page {page_num}")
            tests = await generate_tests_for_page(
                async_client, image, file.name, page_num
            )

            page_folder = save_folder / f"{file.stem}_page{page_num}"
            if page_folder.exists():
                continue
            page_folder.mkdir(parents=True, exist_ok=True)

            output_file = page_folder / "tests.jsonl"
            save_tests(tests, str(output_file))

            pdf_page_path = page_folder / (page_folder.name + ".pdf")
            save_pdf_page(file, page_num, pdf_page_path)

            metadata = {
                "original_doc_path": str(file.absolute()),
                "pdf": file.name,
                "page": page_num,
            }
            metadata_path = page_folder / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved {len(tests)} tests to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_folder",
        type=str,
        default="/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/long_table/",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/tests/long_table2",
    )
    parser.add_argument("--start_idx", type=int, default=300)
    parser.add_argument("--end_idx", type=int, default=500)

    args = parser.parse_args()

    asyncio.run(
        main(
            Path(args.gt_folder),
            Path(args.save_folder),
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )
    )
