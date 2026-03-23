"""Abstract base classes for page processors.

Kept in a standalone module so that ``item_description_processors`` can
import from here without creating a circular dependency with
``page_processor`` (which imports the concrete description configs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .base_model import VLMParseBaseModel
from .data_model.document import Page


class PageProcessor(ABC):
    @abstractmethod
    def __call__(self, page: Page, file_path: str | Path, page_idx: int) -> Page: ...


class AsyncPageProcessor(ABC):
    @abstractmethod
    async def __call__(
        self, page: Page, file_path: str | Path, page_idx: int
    ) -> Page: ...


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
