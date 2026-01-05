import os
import traceback
import zipfile
from pathlib import Path
from typing import Optional

import orjson
from PIL import Image
from PIL import Image as PILImage
from pydantic import Field

from vlmparse.base_model import VLMParseBaseModel
from vlmparse.utils import from_base64, to_base64

from .box import BoundingBox


class ProcessingError(VLMParseBaseModel):
    module_class: str
    traceback: str

    @classmethod
    def from_class(cls, klass):
        return cls(
            module_class=type(klass).__name__,
            traceback=traceback.format_exc(),
        )


class Item(VLMParseBaseModel):
    category: str
    box: BoundingBox
    text: str


class Page(VLMParseBaseModel):
    text: str | None = None
    raw_response: str | None = None
    items: list[Item] | None = None
    error: ProcessingError | None = None
    buffer_image: Optional[Image.Image | str | dict] = None
    latency: Optional[float] = None
    """Time taken to process the page in seconds."""

    @property
    def image(self):
        if isinstance(self.buffer_image, dict):
            from vlmparse.build_doc import convert_specific_page_to_image, resize_image

            image = convert_specific_page_to_image(
                self.buffer_image["file_path"],
                self.buffer_image["page_idx"],
                self.buffer_image["dpi"],
            )
            image = resize_image(image, self.buffer_image["max_image_size"])
            self.buffer_image = image

        if isinstance(self.buffer_image, str):
            self.buffer_image = from_base64(self.buffer_image)
        return self.buffer_image

    def get_image_with_boxes(self, layout=False):
        from PIL import ImageDraw

        from .box import draw_text_of_box

        image = self.image

        if layout:
            if self.items is None:
                return image
            items = self.items
            for item in items:
                box = item.box

                draw = ImageDraw.Draw(image)
                draw.rectangle(
                    (box.l, box.t, box.r, box.b),
                    outline=(255, 0, 0),
                    width=5,
                )

                image = draw_text_of_box(
                    image, box.l, box.t, item.category, font_size=40
                )
        return image


class Document(VLMParseBaseModel):
    file_path: str
    pages: list[Page] = []
    error: ProcessingError | None = None
    metadata: dict = Field(default_factory=dict)
    latency: Optional[float] = None
    """Time taken to process the document in seconds."""

    @property
    def text(self):
        return "\n\n".join([page.text for page in self.pages])

    @property
    def is_error(self):
        return self.error is not None or any(
            page.error is not None for page in self.pages
        )

    def to_zip(
        self,
        file_path,
        overwrite_file: bool = True,
        image_extension: str = "webp",
    ):
        file_path = Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)
        archive_path = str(file_path).removesuffix(".zip") + ".zip"

        if not overwrite_file:
            assert not os.path.isfile(archive_path)

        def _custom_encoder(x):
            if isinstance(x, PILImage.Image):
                return to_base64(x, image_extension)
            if isinstance(x, str):
                return x
            raise TypeError(
                f"Object of type {type(x).__name__} is not JSON serializable"
            )

        json_bytes = orjson.dumps(
            self.model_dump(),
            default=_custom_encoder,
            option=orjson.OPT_INDENT_2,
        )

        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zipf:
            zipf.writestr("data.json", json_bytes)

    @classmethod
    def from_zip(cls, file_path):
        with zipfile.ZipFile(file_path, "r") as zipf:
            if "data.json" not in zipf.namelist():
                raise FileNotFoundError("data.json not found in the archive")

            json_bytes = zipf.read("data.json")
            data = orjson.loads(json_bytes)
        return cls.model_validate(data)
