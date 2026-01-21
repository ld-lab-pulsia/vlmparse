import os

import httpx
import orjson
from loguru import logger
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import Page
from vlmparse.utils import to_base64


class MistralOCRConverterConfig(ConverterConfig):
    """Configuration for Mistral OCR converter."""

    base_url: str = "https://api.mistral.ai/v1"
    model_name: str = "mistral-ocr-latest"
    api_key: str | None = None
    timeout: int = 300
    aliases: list[str] = Field(
        default_factory=lambda: ["mistral-ocr-latest", "mistral-ocr"]
    )

    def get_client(self, **kwargs) -> "MistralOCRConverter":
        return MistralOCRConverter(config=self, **kwargs)


class MistralOCRConverter(BaseConverter):
    """Client for Mistral OCR API."""

    config: MistralOCRConverterConfig

    def __init__(self, config: MistralOCRConverterConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        if not self.config.api_key:
            self.config.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.config.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self._base_url = self.config.base_url.rstrip("/")

    async def _async_ocr(self, image) -> httpx.Response:
        payload = {
            "model": self.config.model_name,
            "document": {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{to_base64(image)}",
            },
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self._base_url}/ocr",
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        return response

    async def async_call_inside_page(self, page: Page) -> Page:
        response = await self._async_ocr(page.image)
        page.raw_response = response.text

        try:
            data = response.json()
        except ValueError:
            logger.warning("Mistral OCR returned non-JSON response")
            page.text = clean_response(response.text)
            return page

        pages = data.get("pages") or []
        if pages:
            page_data = pages[0]
            text = page_data.get("markdown") or page_data.get("text") or ""
        else:
            text = (
                data.get("markdown") or data.get("text") or orjson.dumps(data).decode()
            )

        text = clean_response(text)
        text = html_to_md_keep_tables(text)
        page.text = text
        return page
