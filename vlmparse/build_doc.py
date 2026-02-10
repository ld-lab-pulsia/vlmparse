import re
from pathlib import Path

import numpy as np
import PIL
import pypdfium2 as pdfium
from loguru import logger

from .constants import PDF_EXTENSION


def convert_pdfium(file_path, dpi):
    pil_images = []
    with pdfium.PdfDocument(file_path) as pdf:
        for page in pdf:
            pil_images.append(page.render(scale=dpi / 72).to_pil())
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

    except PIL.Image.DecompressionBombError as e:  # type: ignore[no-untyped-exception]
        logger.opt(exception=True).warning(
            "Decompression bomb detected for {file_path}, reducing DPI",
            file_path=str(file_path),
        )
        cur_size, limit_size = map(int, re.findall(r"\d+", str(e)))
        factor = custom_ceil(cur_size / limit_size, precision=1)
        new_dpi = dpi // factor
        logger.info(
            "Retrying {file_path} with reduced DPI: {old_dpi} -> {new_dpi}",
            file_path=str(file_path),
            old_dpi=dpi,
            new_dpi=new_dpi,
        )
        images = convert_pdfium(file_path, dpi=new_dpi)

    return images


def convert_specific_page_to_image(file_path, page_number, dpi=175):
    with pdfium.PdfDocument(file_path) as pdf:
        page = pdf.get_page(page_number)
        image = page.render(scale=dpi / 72).to_pil()
        image = image.convert("L").convert("RGB") if image.mode != "RGB" else image
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
        with pdfium.PdfDocument(file_path) as pdf:
            return len(pdf)
    else:
        return 1
