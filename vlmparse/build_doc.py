import re
import threading
from pathlib import Path

import numpy as np
import PIL
import PIL.Image
import pypdfium2 as pdfium
from loguru import logger

from .constants import PDF_EXTENSION

# Add a lock to ensure PDFium is accessed by only one thread/task at a time
PDFIUM_LOCK = threading.Lock()


def convert_pdfium(file_path, dpi):
    pil_images = []
    with PDFIUM_LOCK:
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

    except PIL.Image.DecompressionBombError as e:
        logger.opt(exception=True).warning(
            "Decompression bomb detected for {file_path}, reducing DPI",
            file_path=str(file_path),
        )
        cur_size, limit_size = map(int, re.findall(r"\d+", str(e)))
        factor = custom_ceil(cur_size / limit_size, precision=1)
        new_dpi = dpi // factor
        logger.debug(
            "Retrying {file_path} with reduced DPI: {old_dpi} -> {new_dpi}",
            file_path=str(file_path),
            old_dpi=dpi,
            new_dpi=new_dpi,
        )
        images = convert_pdfium(file_path, dpi=new_dpi)

    return images


def convert_specific_page_to_image(file_path, page_number, dpi=175):
    with PDFIUM_LOCK:
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
            logger.debug(f"Resized image to {new_size}")
    return image


def get_page_count(file_path):
    if Path(file_path).suffix.lower() == PDF_EXTENSION:
        with PDFIUM_LOCK:
            with pdfium.PdfDocument(file_path) as pdf:
                return len(pdf)
    else:
        return 1


def parse_page_selection(spec: str, num_pages: int | None = None) -> list[int]:
    """Parse a 1-based page selection string into sorted, unique 0-based indices.

    Supports single pages and ranges, e.g. ``"1-5,8,10"`` -> ``[0, 1, 2, 3, 4, 7, 9]``.

    Args:
        spec: Selection string. Comma-separated list of single pages ("3") and/or
            inclusive ranges ("1-5"). Page numbers are 1-based.
        num_pages: If provided, indices outside ``[0, num_pages)`` are dropped and a
            warning is logged.

    Returns:
        Sorted list of unique 0-based page indices.

    Raises:
        ValueError: If the selection string is malformed or contains page numbers < 1.
    """
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start, end = int(start_str), int(end_str)
            except ValueError as e:
                raise ValueError(f"Invalid page range '{part}'") from e
            if start < 1 or end < 1:
                raise ValueError(f"Page numbers must be >= 1: '{part}'")
            if end < start:
                raise ValueError(f"Invalid page range '{part}': end < start")
            indices.update(range(start - 1, end))
        else:
            try:
                page = int(part)
            except ValueError as e:
                raise ValueError(f"Invalid page number '{part}'") from e
            if page < 1:
                raise ValueError(f"Page numbers must be >= 1: '{part}'")
            indices.add(page - 1)

    result = sorted(indices)
    if num_pages is not None:
        dropped = [i + 1 for i in result if i >= num_pages]
        if dropped:
            logger.warning(
                "Ignoring out-of-range page(s) {dropped}; document has {num_pages} page(s)",
                dropped=dropped,
                num_pages=num_pages,
            )
        result = [i for i in result if i < num_pages]
    return result
