from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field

from .page_processor_base import AsyncPageProcessor, PageProcessor, PageProcessorConfig

# Re-export so existing imports like `from vlmparse.page_processor import AsyncPageProcessor` keep working
__all__ = [
    "AsyncPageProcessor",
    "PageProcessor",
    "PageProcessorConfig",
]

from .data_model.document import Page  # noqa: F401 (used by processors below)

# ---------------------------------------------------------------------------
# Built-in preprocessor: native text-cell extraction
# ---------------------------------------------------------------------------


class ExtractTextCellsProcessor(PageProcessor):
    """Extract native PDF text cells and scale their boxes to image space."""

    def __call__(self, page: Page, file_path: str | Path, page_idx: int) -> Page:
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

    def __call__(self, page: Page, file_path: str | Path, page_idx: int) -> Page:
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

# Imported here (after all base classes are defined) to avoid circular imports.
from vlmparse.clients.item_description_processors import (  # noqa: E402
    DeepSeekOCR2ItemDescriptionConfig,
    DeepSeekOCRItemDescriptionConfig,
    VLMItemDescriptionConfig,
)

PageProcessorConfigs = Annotated[
    Union[
        ExtractTextCellsConfig,
        UriAnnotatorConfig,
        DeepSeekOCRItemDescriptionConfig,
        DeepSeekOCR2ItemDescriptionConfig,
        VLMItemDescriptionConfig,
    ],
    Field(discriminator="class_name"),
]
