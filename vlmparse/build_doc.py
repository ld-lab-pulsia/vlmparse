import re
from pathlib import Path

import numpy as np
import PIL
import pypdfium2 as pdfium
from loguru import logger

from .constants import PDF_EXTENSION


def convert_pdfium(file_path, dpi):
    pdf = pdfium.PdfDocument(file_path)
    pil_images = []
    for page in pdf:
        pil_images.append(page.render(scale=dpi / 72).to_pil())

    pdf.close()
    return pil_images


def custom_ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def convert_pdfium_to_images(file_path, dpi=175):
    try:
        images = convert_pdfium(file_path, dpi=dpi)
        images = [
            img.convert("L").convert("RGB") if img.mode != "RGB" else img
            for img in images
        ]

    except PIL.Image.DecompressionBombError as e:
        logger.exception(f"Got problem size document with {file_path}")
        cur_size, limit_size = map(int, re.findall(r"\d+", str(e)))
        factor = custom_ceil(cur_size / limit_size, precision=1)
        logger.warning(
            f"Try again by reducing DPI for doc {file_path} from {dpi} to {dpi//factor}"
        )
        dpi = dpi // factor
        images = convert_pdfium(file_path, dpi=dpi)

    return images


def convert_specific_page_to_image(file_path, page_number, dpi=175):
    pdf = pdfium.PdfDocument(file_path)
    page = pdf.get_page(page_number)
    image = page.render(scale=dpi / 72).to_pil()
    image = image.convert("L").convert("RGB") if image.mode != "RGB" else image
    pdf.close()
    return image


def resize_image(image, max_image_size):
    if max_image_size is not None:
        ratio = max_image_size / max(image.size)
        if ratio < 1:
            new_size = (
                int(image.size[0] * ratio),
                int(image.size[1] * ratio),
            )
            image = image.resize(new_size)
            logger.info(f"Resized image to {new_size}")
    return image


def get_page_count(file_path):
    if Path(file_path).suffix.lower() == PDF_EXTENSION:
        pdf = pdfium.PdfDocument(file_path)
        count = len(pdf)
        pdf.close()
        return count
    else:
        return 1
