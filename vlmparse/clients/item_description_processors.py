"""Async page post-processors that fill item.text with image descriptions.

Two flavours are provided:
- ``DeepSeekOCRItemDescriptionConfig`` / ``DeepSeekOCR2ItemDescriptionConfig``:
  use the native DeepSeek-OCR prompting + NGram logits-processor parameters.
- ``VLMItemDescriptionConfig``: generic OpenAI-compatible VLM; *model_name*
  and *completion_kwargs* are required (no defaults).

Both flavours share ``BaseItemDescriptionProcessor`` — only the config
defaults differ.  Items whose ``category`` is not in ``config.categories``
are skipped; all matching items on a page are described concurrently via
``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from loguru import logger
from PIL import Image
from pydantic import Field

from vlmparse.clients.deepseekocr_constants import (  # noqa: E402
    DEEPSEEK_IMAGE_DESCRIPTION_PROMPT as _DEEPSEEK_IMAGE_DESCRIPTION_PROMPT,
)
from vlmparse.clients.deepseekocr_constants import (
    deepseek_completion_kwargs as _deepseek_completion_kwargs,
)
from vlmparse.data_model.document import Page
from vlmparse.model_endpoint_config import ModelEndpointConfig as ModelEndpointConfig
from vlmparse.page_processor_base import AsyncPageProcessor, PageProcessorConfig
from vlmparse.servers.docker_server import DEFAULT_MODEL_NAME as _DEFAULT_MODEL_NAME
from vlmparse.utils import to_base64

DEFAULT_IMAGE_DESCRIPTION_PROMPT: str = "Briefly describe the image. If it is a graph or a diagram, translate it into a table, a Mermaid diagram, or any other appropriate format."
DEFAULT_CATEGORIES: list[str] = ["picture", "image", "figure", "chart"]

# ---------------------------------------------------------------------------
# Processor (shared by all flavours)
# ---------------------------------------------------------------------------


class BaseItemDescriptionProcessor(AsyncPageProcessor):
    """Crop items whose category is in *config.categories* and describe them.

    All matching items on a page are processed concurrently via
    ``asyncio.gather``.  Individual item failures log a warning and leave
    ``item.text`` unchanged rather than interrupting the whole page.
    """

    def __init__(self, config: BaseItemDescriptionConfig) -> None:
        self.config = config
        self._client = None
        self._client_loop = None

    async def _get_async_client(self):
        from openai import AsyncOpenAI

        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception:
                    pass
            conn = self.config.connection
            self._client = AsyncOpenAI(
                base_url=conn.base_url,
                api_key=conn.api_key or "no-key",
                timeout=conn.timeout,
                max_retries=conn.max_retries,
            )
            self._client_loop = loop
        return self._client

    async def _describe_crop(self, crop: Image.Image) -> str:
        client = await self._get_async_client()
        b64 = await asyncio.to_thread(to_base64, crop)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": self.config.prompt},
                ],
            }
        ]
        response = await client.chat.completions.create(
            model=self.config.connection.model_name,
            messages=messages,
            **self.config.completion_kwargs,
        )
        return response.choices[0].message.content or ""

    async def __call__(self, page: Page, file_path: str | Path, page_idx: int) -> Page:
        if page.items is None or page.image is None:
            return page

        target_items = [
            item for item in page.items if item.category in self.config.categories
        ]
        if not target_items:
            return page

        image = page.image

        async def describe_item(item) -> None:
            try:
                crop = image.crop((item.box.l, item.box.t, item.box.r, item.box.b))
                item.text = await self._describe_crop(crop)
                logger.debug(
                    "Described item category={} ({} chars)",
                    item.category,
                    len(item.text),
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to describe item category={} box={}",
                    item.category,
                    item.box,
                )

        await asyncio.gather(*[describe_item(item) for item in target_items])
        return page


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------


class BaseItemDescriptionConfig(PageProcessorConfig):
    """Shared configuration for item-level image description post-processors.

    Connection parameters live inside the nested ``connection`` field
    (:class:`ModelEndpointConfig`).  Task-specific knobs (prompt,
    completion_kwargs, categories) remain at the top level.
    """

    connection: ModelEndpointConfig
    prompt: str = DEFAULT_IMAGE_DESCRIPTION_PROMPT
    completion_kwargs: dict
    categories: list[str] = Field(default_factory=lambda: list(DEFAULT_CATEGORIES))

    def get_processor(self) -> BaseItemDescriptionProcessor:
        # Cache the processor on the config instance so the underlying
        # AsyncOpenAI client is reused across pages.
        try:
            return self._cached_processor  # ty: ignore[unresolved-attribute]
        except AttributeError:
            processor = BaseItemDescriptionProcessor(config=self)
            object.__setattr__(self, "_cached_processor", processor)
            return processor


# ---------------------------------------------------------------------------
# DeepSeek-OCR flavours (native prompting + NGram logits processor)
# ---------------------------------------------------------------------------


class DeepSeekOCRItemDescriptionConfig(BaseItemDescriptionConfig):
    """DeepSeek-OCR v1 — NGram logits processor (ngram_size=30, window_size=90)."""

    class_name: Literal["DeepSeekOCRItemDescriptionConfig"] = (
        "DeepSeekOCRItemDescriptionConfig"
    )
    connection: ModelEndpointConfig = Field(
        default_factory=lambda: ModelEndpointConfig(model_name=_DEFAULT_MODEL_NAME)
    )
    prompt: str = _DEEPSEEK_IMAGE_DESCRIPTION_PROMPT
    completion_kwargs: dict = Field(
        default_factory=lambda: _deepseek_completion_kwargs(
            ngram_size=30, window_size=90, max_tokens=512
        )
    )


class DeepSeekOCR2ItemDescriptionConfig(BaseItemDescriptionConfig):
    """DeepSeek-OCR-2 — NGram logits processor (ngram_size=20, window_size=50)."""

    class_name: Literal["DeepSeekOCR2ItemDescriptionConfig"] = (
        "DeepSeekOCR2ItemDescriptionConfig"
    )
    connection: ModelEndpointConfig = Field(
        default_factory=lambda: ModelEndpointConfig(model_name=_DEFAULT_MODEL_NAME)
    )
    prompt: str = _DEEPSEEK_IMAGE_DESCRIPTION_PROMPT
    completion_kwargs: dict = Field(
        default_factory=lambda: _deepseek_completion_kwargs(
            ngram_size=20, window_size=50, max_tokens=512
        )
    )


# ---------------------------------------------------------------------------
# Generic / generalist VLM (no defaults for model_name or completion_kwargs)
# ---------------------------------------------------------------------------


class VLMItemDescriptionConfig(BaseItemDescriptionConfig):
    """Generic OpenAI-compatible VLM image description.

    ``connection`` and ``completion_kwargs`` are required — no defaults are
    provided so callers must be explicit about the endpoint and generation
    parameters (temperature, max_tokens, etc.).
    """

    class_name: Literal["VLMItemDescriptionConfig"] = "VLMItemDescriptionConfig"
    connection: ModelEndpointConfig
    completion_kwargs: dict
