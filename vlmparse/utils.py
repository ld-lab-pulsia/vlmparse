import base64
import os
from glob import glob
from io import BytesIO

from loguru import logger
from PIL import Image


def to_base64(image: Image, extension="PNG"):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=extension)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")


def from_base64(base64_str: str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def get_file_paths(inputs: str | list[str]):
    # Expand file paths from glob patterns
    file_paths = []
    if isinstance(inputs, str):
        inputs = [inputs]
    for pattern in inputs:
        if "*" in pattern or "?" in pattern:
            file_paths.extend(glob(pattern, recursive=True))
        elif os.path.isdir(pattern):
            file_paths.extend(glob(os.path.join(pattern, "*.*"), recursive=True))
        elif os.path.isfile(pattern):
            file_paths.append(pattern)
        else:
            logger.error(f"Invalid input: {pattern}")
    file_paths = [f for f in file_paths if os.path.exists(f) and os.path.isfile(f)]

    if not file_paths:
        logger.error("No PDF files found matching the inputs patterns")

    return file_paths
