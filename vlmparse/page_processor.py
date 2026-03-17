from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal, Union

from pydantic import Field

from .base_model import VLMParseBaseModel
from .data_model.document import Page


class PageProcessor(ABC):
    """Base class for page pre/post processors.

    Subclasses must implement ``__call__`` which receives a :class:`Page`
    and optional keyword context (e.g. ``file_path``, ``page_idx``) and
    returns the (possibly mutated) page.
    """

    @abstractmethod
    def __call__(self, page: Page, **context: Any) -> Page: ...


class AsyncPageProcessor(ABC):
    """Base class for async page pre/post processors.

    Subclasses must implement ``__call__`` as an async method.
    When ``to_thread`` is set on the config it is ignored for async
    processors — they are always awaited directly.
    """

    @abstractmethod
    async def __call__(self, page: Page, **context: Any) -> Page: ...


class PageProcessorConfig(VLMParseBaseModel):
    """Base configuration for a page processor.

    Subclasses must define a ``class_name`` :class:`Literal` field so
    that the discriminated union can deserialise configs from plain dicts.

    Parameters
    ----------
    to_thread:
        When *True* the processor is executed via ``asyncio.to_thread``
        so that blocking work does not stall the event loop.
    """

    class_name: str
    to_thread: bool = False

    def get_processor(self) -> PageProcessor | AsyncPageProcessor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in preprocessor: native text-cell extraction
# ---------------------------------------------------------------------------


class ExtractTextCellsProcessor(PageProcessor):
    """Extract native PDF text cells and scale their boxes to image space."""

    def __call__(self, page: Page, **context: Any) -> Page:
        file_path = context.get("file_path")
        page_idx = context.get("page_idx")
        if file_path is None or page_idx is None:
            return page

        from .data_model.box import BoundingBox
        from .docling_extractor import extract_page_text_cells

        cells, pdf_w, pdf_h = extract_page_text_cells(file_path, page_idx)
        if cells is not None and pdf_w and pdf_h:
            img = page.image
            if img is not None:
                scale_x = img.width / pdf_w
                scale_y = img.height / pdf_h
                for cell in cells:
                    b = cell.box
                    cell.box = BoundingBox(
                        l=b.l * scale_x,
                        t=b.t * scale_y,
                        r=b.r * scale_x,
                        b=b.b * scale_y,
                    )
            page.text_cells = cells
        return page


class ExtractTextCellsConfig(PageProcessorConfig):
    class_name: Literal["ExtractTextCellsConfig"] = "ExtractTextCellsConfig"
    to_thread: bool = True

    def get_processor(self) -> PageProcessor:
        return ExtractTextCellsProcessor()


# ---------------------------------------------------------------------------
# Built-in postprocessor: URI annotation
# ---------------------------------------------------------------------------


class UriAnnotatorProcessor(PageProcessor):
    """Annotate page items and text with hyperlink URIs from text cells."""

    def __call__(self, page: Page, **context: Any) -> Page:
        from .uri_annotator import annotate_page_items_with_uris

        annotate_page_items_with_uris(page)
        return page


class UriAnnotatorConfig(PageProcessorConfig):
    class_name: Literal["UriAnnotatorConfig"] = "UriAnnotatorConfig"
    to_thread: bool = False

    def get_processor(self) -> PageProcessor:
        return UriAnnotatorProcessor()


# ---------------------------------------------------------------------------
# Discriminated union of all built-in processor configs
# ---------------------------------------------------------------------------

PageProcessorConfigs = Annotated[
    Union[
        ExtractTextCellsConfig,
        UriAnnotatorConfig,
    ],
    Field(discriminator="class_name"),
]
