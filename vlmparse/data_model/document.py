import os
import traceback
import zipfile
from pathlib import Path
from typing import Any, Optional, cast

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
    buffer_image: Optional[Image.Image | str | dict[str, Any]] = None
    latency: Optional[float] = None
    """Time taken to process the page in seconds."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    """Include reasoning tokens"""
    reasoning_tokens: Optional[int] = None

    @property
    def image(self) -> Optional[PILImage.Image]:
        buffer = self.buffer_image
        if isinstance(buffer, dict):
            from vlmparse.build_doc import convert_specific_page_to_image, resize_image

            # Casting solves the 'Top[dict]' or 'key: Never' issue with Pydantic fields
            d = cast(dict[str, Any], buffer)
            image = convert_specific_page_to_image(
                d["file_path"],
                d["page_idx"],
                d["dpi"],
            )
            image = resize_image(image, d.get("max_image_size"))
            self.buffer_image = image
            return image

        if isinstance(buffer, str):
            image = from_base64(buffer)
            self.buffer_image = image
            return image

        return cast(Optional[PILImage.Image], buffer)

    def get_image_with_boxes(self, layout=False):
        from PIL import ImageDraw

        from .box import draw_text_of_box

        image = self.image

        if layout and image is not None:
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

    def to_markdown(self, **kwargs):
        return self.text if self.text is not None else ""


class Document(VLMParseBaseModel):
    file_path: str
    pages: list[Page] = []
    error: ProcessingError | None = None
    metadata: dict = Field(default_factory=dict)
    latency: Optional[float] = None
    """Time taken to process the document in seconds."""

    @property
    def text(self):
        return "\n\n".join([page.text for page in self.pages if page.text is not None])

    @property
    def is_error(self):
        return self.error is not None or any(
            page.error is not None for page in self.pages
        )

    def to_markdown(self, **kwargs):
        return "\n\n".join([page.to_markdown(**kwargs) for page in self.pages])

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
