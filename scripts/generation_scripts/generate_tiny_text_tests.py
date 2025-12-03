import asyncio
import json
import os
import random
from pathlib import Path

import pypdfium2 as pdfium
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import TextPresenceTest, save_tests
from vlmparse.build_doc import convert_pdfium_to_images
from vlmparse.utils import to_base64

DATA_PROJECT = Path("/data_project")


class TinyTextTestsResponse(BaseModel):
    texts: list[str] = Field(
        description="List of exactly 3 tiny texts: minimum of 3 words, can be several paragraphs that is visually small or hard to read",
        min_length=3,
        max_length=3,
    )


PROMPTS = {
    "tiny_text": """Analyze this page and identify exactly 3 of the TINIEST, most difficult to read text extracts.

Focus on:
- Very small font size text (fine print, subscripts, small annotations)
- Text that appears visually small or compressed
- Minimum of 3 words each, can be several paragraphs
- Text that might be challenging for OCR to detect
- Do not include footnotes, headers, footers, page numbers, watermarks, etc
- Do not include text that appear in images, plots, charts, etc...

For each fragment, extract the exact text as it appears.

Return exactly 3 distinct tiny text extracts.""",
    "handwritten_text": """Analyze this page and identify from 1 to 10 handwritten text extracts. Return the extracts as a list of strings that should be the exact unmodified OCR of the handwritten text.

Focus exclusively on Handwritten text:
- Minimum of 3 words each, can be several paragraphs
- Do not include footnotes, headers, footers, page numbers, watermarks, etc...
- Do not include text that appear in images, plots, charts, etc...""",
    "headers_footers": """Analyze this page and identify page headers and footer text: texts that are not part of the linear text flow of the document and that bring metadata or other information.

Focus on:
- Page Headers: WARNING: do not include section headers, document titles or page titles: they are part of the main content flow.
- Footers: text at the bottom of the page (page numbers, copyright notices): WARNING: do not include footnotes.
- Extract the exact text as it appears
- Do not include text that is part of the main body content
- Do not include text that appears in images, plots, charts, etc.""",
    "graphics": """Analyze this page and identify graphics, ie plots, curve, charts, diagrams, schemas, histograms, etc... Return a list of strings that should be the exact unmodified OCR of single text elements in the graphics (a number, a label, a title, a legend, a paragraph, etc...). The goal is to test the presence of this text single element in a pdf to markdown OCR conversion.

Focus on:
- Graphics: plots, curve, charts, diagrams, schemas, histograms, maps.
- Single text elements: a number, a label, a title, a legend, a paragraph.
- Extract the exact text as it appears
- If there is none of the graphics mentioned above, return an empty list.
- Do not include text that is in footers or headers.
""",
}


async def generate_tests_for_page(
    client,
    image,
    pdf_name: str,
    page_num: int,
    test_type: str = "tiny_text",
    model: str = "gemini-2.5-pro",
):
    img_b64 = to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPTS[test_type]},
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
        response_format=TinyTextTestsResponse,
    )
    import pdb

    pdb.set_trace()

    tests_response = response.choices[0].message.parsed
    if tests_response is None:
        return None
    tests = []
    for idx, text in enumerate(tests_response.texts):
        test = TextPresenceTest(
            category=test_type,
            pdf=pdf_name,
            page=page_num,
            id=f"{pdf_name}_p{page_num}_tiny{idx}",
            type="absent" if test_type == "headers_footers" else "present",
            text=text,
            max_diffs=0,
            unidecode=True,
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
    gt_folder: Path,
    save_folder: Path,
    start_idx: int = 300,
    end_idx: int = 400,
    no_inverse_aspect_ratio: bool = False,
    test_type: str = "tiny_text",
    model: str = "gemini-2.5-pro",
    add_page_num: bool = True,
):
    # save_folder = save_folder / test_type
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
            if no_inverse_aspect_ratio and image.width > image.height:
                continue
            if add_page_num:
                page_folder = save_folder / f"{file.stem}_page{page_num}"
            else:
                page_folder = save_folder / f"{file.stem}"

            pdf_page_path = page_folder / (page_folder.name + ".pdf")
            # old_tests = os.listdir("/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/tests/tiny_text_long_text2")

            test_name = f"tests_{test_type}.jsonl"

            output_file = page_folder / test_name
            if output_file.exists():  # or f"{file.stem}_page{page_num}" in old_tests:
                # import pdb;pdb.set_trace()
                continue

            print(f"Generating tests for {file.name} page {page_num}")
            tests = await generate_tests_for_page(
                async_client,
                image,
                file.name,
                page_num,
                test_type=test_type,
                model=model,
            )
            if tests is None or len(tests) == 0:
                continue

            page_folder.mkdir(parents=True, exist_ok=True)

            save_tests(tests, str(output_file))

            if not pdf_page_path.exists():
                save_pdf_page(file, page_num, pdf_page_path)
            metadata_path = page_folder / "metadata.json"

            if not metadata_path.exists():
                metadata = {
                    "original_doc_path": str(file.absolute()),
                    "pdf": file.name,
                    "page": page_num,
                    "doc_type": "long_text",
                }

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            print(f"Saved {len(tests)} tests to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_folder",
        type=str,
        default=os.path.expanduser("~/data/bench_data/difficult_pdfs/handwritten3/"),
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=os.path.expanduser("~/data/bench_data/tests/"),
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10000)
    parser.add_argument("--test_type", type=str, default="handwritten_text")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    args = parser.parse_args()
    asyncio.run(
        main(
            Path(args.gt_folder),
            Path(args.save_folder),
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            test_type=args.test_type,
            model=args.model,
        )
    )
