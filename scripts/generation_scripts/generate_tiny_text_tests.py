import asyncio
import json
import os
from pathlib import Path
import pypdfium2 as pdfium
from pydantic import BaseModel, Field
from vlmparse.utils import to_base64
from vlmparse.build_doc import convert_pdfium_to_images
from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import save_tests, TextPresenceTest
import random
from openai import AsyncOpenAI

DATA_PROJECT = Path("/data_project")


class TinyTextTestsResponse(BaseModel):
    tests: list[str] = Field(description="List of exactly 3 tiny texts: minimum of 3 words, can be several paragraphs that is visually small or hard to read", min_length=3, max_length=3)

PROMPT = """Analyze this page and identify exactly 3 of the TINIEST, most difficult to read text extracts.

Focus on:
- Very small font size text (fine print, subscripts, small annotations)
- Text that appears visually small or compressed
- Minimum of 3 words each, can be several paragraphs
- Text that might be challenging for OCR to detect
- Do not include footnotes, headers, footers, page numbers, watermarks, etc
- Do not include text that appear in images, plots, charts, etc...

For each fragment, extract the exact text as it appears.

Return exactly 3 distinct tiny text extracts."""

async def generate_tests_for_page(client, image, pdf_name: str, page_num: int):
    img_b64 = to_base64(image)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }]
    
    response = await client.chat.completions.parse(
        model="gemini-2.5-pro",
        messages=messages,
        response_format=TinyTextTestsResponse,
    )
    
    tests_response = response.choices[0].message.parsed
    if tests_response is None:
        return None
    tests = []
    for idx, text in enumerate(tests_response.tests):
        test = TextPresenceTest(
            category="tiny_text",
            pdf=pdf_name,
            page=page_num,
            id=f"{pdf_name}_p{page_num}_tiny{idx}",
            type="present",
            text=text,
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

async def main(gt_folder: Path, save_folder: Path, start_idx: int = 300, end_idx: int = 400, no_inverse_aspect_ratio: bool = True):
    async_client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=500,
    )
    files = [f for f in gt_folder.iterdir() if f.is_file() and f.name.endswith(".pdf")]
    random.seed(42)
    random.shuffle(files)
    
    for file in files[start_idx:end_idx]:
        images = convert_pdfium_to_images(file, dpi=250)
        for page_num, image in enumerate(images):
            if no_inverse_aspect_ratio and image.width > image.height:
                continue

            print(f"Generating tests for {file.name} page {page_num}")
            tests = await generate_tests_for_page(async_client, image, file.name, page_num)
            if tests is None:
                continue
            page_folder = save_folder / f"{file.stem}_page{page_num}"
            page_folder.mkdir(parents=True, exist_ok=True)
            
            output_file = page_folder / "tests.jsonl"
            save_tests(tests, str(output_file))
            
            pdf_page_path = page_folder / "page.pdf"
            save_pdf_page(file, page_num, pdf_page_path)
            
            metadata = {
                "original_doc_path": str(file.absolute()),
                "pdf": file.name,
                "page": page_num,
                "doc_type": "long_text",
            }
            metadata_path = page_folder / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved {len(tests)} tests to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_folder", type=str, default="/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/long_text/")
    parser.add_argument("--save_folder", type=str, default="/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/tests/tiny_text_long_text2")
    parser.add_argument("--start_idx", type=int, default=500)
    parser.add_argument("--end_idx", type=int, default=10000)
    args = parser.parse_args()
    asyncio.run(main(Path(args.gt_folder), Path(args.save_folder), start_idx=args.start_idx, end_idx=args.end_idx))

