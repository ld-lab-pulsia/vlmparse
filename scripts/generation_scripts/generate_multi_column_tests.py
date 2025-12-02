import asyncio
import json
import os
from pathlib import Path

import pypdfium2 as pdfium
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import TextPresenceTest, save_tests
from vlmparse.build_doc import convert_pdfium_to_images
from vlmparse.utils import to_base64


class MultiColumnTestSpecs(BaseModel):
    markdown_translation: str = Field(
        description="Full markdown translation of the entire page"
    )
    texts_flowing: list[str] = Field(
        description="List of 1-4 texts that flow from one column to the next"
    )


PROMPT = """First, translate the entire page to markdown format, preserving the text flow and structure.

Then, analyze this multi-column document page and identify between 1 and 4 instances where text flows from one column to the next.

For each instance, select a text paragraph that flows from one column to the next:
- Be actual consecutive text that flows across columns
- Not include column headers or footers"""


async def generate_test_for_page(client, image, pdf_name: str, page_num: int):
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
        response_format=MultiColumnTestSpecs,
    )

    specs = response.choices[0].message.parsed
    if specs is None:
        return []
    tests = []
    for idx, spec in enumerate(specs.texts_flowing):
        concatenated_text = spec
        test = TextPresenceTest(
            pdf=pdf_name,
            page=page_num,
            id=f"{pdf_name}_p{page_num}_multicolumn_{idx}",
            type="present",
            text=concatenated_text,
            max_diffs=0,
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
    gt_folder: Path, save_folder: Path, start_idx: int = 0, end_idx: int = None
):
    async_client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=500,
    )
    files = [f for f in gt_folder.iterdir() if f.is_file() and f.name.endswith(".pdf")]

    print(f"Number of files: {len(files)}")
    import random

    random.seed(42)
    random.shuffle(files)
    keep_files = []
    already_seen_files = set()
    for file in files:
        stem = "_".join(str(file.name).split("_")[1:-1])
        if stem in already_seen_files:
            continue
        else:
            keep_files.append(file)
        already_seen_files.add(stem)

    files = keep_files
    print(f"Number of files after filtering: {len(files)}")
    # files = files[start_idx:end_idx]

    for file in tqdm(files):
        page_folder = save_folder / f"{file.stem}"

        if page_folder.exists():
            continue
        images = convert_pdfium_to_images(file, dpi=200)
        page_num = 0
        image = images[page_num]

        print(f"Generating tests for {file.name}")
        tests = await generate_test_for_page(async_client, image, file.name, page_num)
        if len(tests) == 0:
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

        print(f"Saved {len(tests)} test(s) to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_folder",
        type=str,
        default=os.path.expanduser("~/data/bench_data/difficult_pdfs/multi_column/"),
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=os.path.expanduser("~/data/bench_data/multi_column3"),
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=300)

    args = parser.parse_args()

    asyncio.run(
        main(
            Path(args.gt_folder),
            Path(args.save_folder),
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )
    )
