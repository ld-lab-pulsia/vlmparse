import base64
from io import BytesIO

from PIL import Image


def to_base64(image: Image, extension="PNG"):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=extension)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")


def from_base64(base64_str: str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))
