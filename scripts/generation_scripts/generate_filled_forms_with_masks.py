import asyncio
import base64
import io
import os
from io import BytesIO
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from tqdm import tqdm

from vlmparse.build_doc import convert_pdfium_to_images


class BoundingBox(BaseModel):
    y_min: float = Field(description="Minimum y coordinate (0-1000 normalized)")
    x_min: float = Field(description="Minimum x coordinate (0-1000 normalized)")
    y_max: float = Field(description="Maximum y coordinate (0-1000 normalized)")
    x_max: float = Field(description="Maximum x coordinate (0-1000 normalized)")


class FormFieldsResponse(BaseModel):
    boxes: list[BoundingBox] = Field(
        description="List of bounding boxes for all form fields that should be filled by hand, in format [y_min, x_min, y_max, x_max] with coordinates normalized 0-1000"
    )


DETECTION_PROMPT = """Analyze this form document and identify all fields that should be filled by hand.

Identify:
- Text input fields (name, address, date, etc.)
- Checkboxes that need to be checked
- Signature fields
- Date fields
- Any other fields where handwritten input is expected

For each field, provide the bounding box in format [y_min, x_min, y_max, x_max] with coordinates normalized between 0 and 1000."""


FILL_PROMPT = """Instructions strictes :
- Remplis tous les champs du formulaire avec des données fictives cohérentes et réalistes (n'utilise pas Dupont, utilise des données réalistes).
- Les données doivent être crédibles et cohérentes entre elles (dates, noms, adresses, etc.)
- Conserve exactement la même structure, les mêmes champs et la même disposition
- Ne modifie aucun pixel du formulaire original en dehors des champs à remplir.
- Préserve exactement la mise en page, les lignes, les bordures, les polices imprimées et tout le texte existant.
- N'altère ni la netteté ni la structure du formulaire.
- Le seul changement autorisé est l'ajout d'une écriture manuscrite réaliste dans les zones prévues pour la saisie.
- Utilise une écriture manuscrite crédible et difficile à lire (lisibilité minimale) de couleur rouge, bleue ou noire"""


async def detect_form_fields(
    gemini_client, image: Image.Image, pdf_name: str, page_num: int
):
    """Detect form fields"""

    # Save original image size before any resizing
    original_width, original_height = image.size

    # Resize image if needed
    max_dimension = 2048
    if max(image.size) > max_dimension:
        image = image.copy()
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    # Convert to base64
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DETECTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ],
        }
    ]

    response = await gemini_client.chat.completions.parse(
        model="gemini-2.5-pro",
        messages=messages,
        response_format=FormFieldsResponse,
    )

    fields_response = response.choices[0].message.parsed

    if fields_response is None:
        return None

    # Convert normalized coordinates (0-1000) to pixel coordinates using ORIGINAL image size
    pixel_boxes = []
    for box in fields_response.boxes:
        left = int(box.x_min * original_width / 1000)
        top = int(box.y_min * original_height / 1000)
        right = int(box.x_max * original_width / 1000)
        bottom = int(box.y_max * original_height / 1000)
        pixel_boxes.append((left, top, right, bottom))

    return pixel_boxes


def generate_mask(image: Image.Image, boxes):
    """Generate a mask image: original image with transparent boxes for form fields."""
    # Start with original image as RGBA
    mask = image.convert("RGBA")
    draw = ImageDraw.Draw(mask)

    for left, top, right, bottom in boxes:
        # Make field area transparent
        draw.rectangle(
            (left, top, right, bottom),
            fill=(0, 0, 0, 0),
        )

    return mask


def generate_filled_form(openai_client, image: Image.Image, mask: Image.Image):
    """Generate a filled form image using OpenAI's image edit API."""

    # Resize image if needed
    max_dimension = 1024
    if max(image.size) > max_dimension:
        # Calculate new size maintaining aspect ratio
        ratio = max_dimension / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        mask = mask.resize(new_size, Image.Resampling.LANCZOS)

    # Convert image to RGBA
    image = image.convert("RGBA")

    # Prepare image bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    img_bytes.name = "image.png"

    # Prepare mask bytes
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")
    mask_bytes.seek(0)
    mask_bytes.name = "mask.png"

    # Call OpenAI API
    result = openai_client.images.edit(
        model="gpt-image-1",
        image=img_bytes,
        mask=mask_bytes,
        prompt=FILL_PROMPT,
        size="1024x1536",
        quality="high",
    )

    # Decode result
    image_data = result.data[0].b64_json
    image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes))


async def process_page(
    gemini_client,
    openai_client,
    image: Image.Image,
    pdf_name: str,
    page_num: int,
    save_folder: Path,
):
    """Process a single page: detect fields, generate mask, generate filled form."""

    base_filename = f"{Path(pdf_name).stem}_page{page_num}"
    mask_path = save_folder / f"{base_filename}_mask.png"
    filled_path = save_folder / f"{base_filename}_filled_minimal_readability.png"

    # Check if filled form already exists
    if filled_path.exists():
        print(f"  Skipping: filled form already exists at {filled_path}")
        return

    # Save original image
    original_path = save_folder / f"{base_filename}_original.png"
    if not original_path.exists():
        image.save(original_path, format="PNG")
        print(f"  Saved original to {original_path}")

    # Check if mask already exists
    if mask_path.exists():
        print(f"  Loading existing mask from {mask_path}")
        mask = Image.open(mask_path)
    else:
        # Detect form fields
        print("  Detecting form fields...")
        boxes = await detect_form_fields(gemini_client, image, pdf_name, page_num)

        if boxes is None or len(boxes) == 0:
            print(f"  No fields detected for {pdf_name} page {page_num}")
            return

        print(f"  Detected {len(boxes)} fields")

        # Generate mask
        mask = generate_mask(image, boxes)
        mask.save(mask_path, format="PNG")
        print(f"  Saved mask to {mask_path}")

    # Generate filled form
    print("  Generating filled form...")
    filled_image = generate_filled_form(openai_client, image, mask)

    if filled_image is None:
        print("  Failed to generate filled form")
        return

    filled_image.save(filled_path, format="PNG")
    print(f"  Saved filled form to {filled_path}")


async def main(
    pdf_folder: Path, save_folder: Path, start_idx: int = 0, end_idx: int = None
):
    """Process PDFs and generate filled forms with masks."""

    # Initialize clients
    gemini_client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=500,
    )

    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=1000,
    )

    files = [f for f in pdf_folder.iterdir() if f.is_file() and f.name.endswith(".pdf")]
    files.sort()

    if end_idx is None:
        end_idx = len(files)

    save_folder.mkdir(parents=True, exist_ok=True)

    for file_idx, file in tqdm(
        enumerate(files[start_idx:end_idx]), total=len(files[start_idx:end_idx])
    ):
        print(
            f"\nProcessing {file.name} ({file_idx + 1}/{len(files[start_idx:end_idx])})"
        )

        try:
            images = convert_pdfium_to_images(file, dpi=200)

            for page_num, image in enumerate(images):
                print(f"  Processing page {page_num + 1}/{len(images)}")

                await process_page(
                    gemini_client,
                    openai_client,
                    image,
                    file.name,
                    page_num,
                    save_folder,
                )

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            import traceback

            traceback.print_exc()
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate filled form images with masks using Gemini for detection and OpenAI for generation"
    )
    parser.add_argument(
        "--pdf_folder",
        type=str,
        default="/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/forms2/",
        help="Folder containing PDF forms to process",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/forms2_complete/",
        help="Folder to save all generated images (original, mask, filled)",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Starting index for processing files"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Ending index for processing files (None for all remaining)",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            Path(args.pdf_folder),
            Path(args.save_folder),
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )
    )
