import asyncio
import os
from loguru import logger
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import Page
from vlmparse.utils import to_base64

from .prompts import PDF2MD_PROMPT


class AnthropicConverterConfig(ConverterConfig):
    api_key: str = ""
    timeout: int | None = 500
    max_retries: int = 1
    preprompt: str | None = None
    postprompt: str = PDF2MD_PROMPT
    max_tokens: int = 4096
    completion_kwargs: dict = Field(default_factory=dict)

    def get_client(self, **kwargs) -> "AnthropicConverterClient":
        return AnthropicConverterClient(config=self, **kwargs)


class AnthropicConverterClient(BaseConverter):
    """Client for Anthropic Claude API (vision)."""

    config: AnthropicConverterConfig

    def __init__(self, config: AnthropicConverterConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self._client = None
        self._client_loop = None

    async def _get_async_client(self):
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            await self._close_client()
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY", ""),
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            self._client_loop = loop
        return self._client

    async def _close_client(self):
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            finally:
                self._client = None
                self._client_loop = None

    async def aclose(self):
        await self._close_client()

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image
        assert image is not None, "Page image is required for conversion"

        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": to_base64(image),
                },
            },
            {
                "type": "text",
                "text": self.config.postprompt,
            },
        ]

        messages = [{"role": "user", "content": content}]

        create_kwargs = dict(self.config.completion_kwargs)
        if self.config.preprompt:
            create_kwargs["system"] = self.config.preprompt

        client = await self._get_async_client()
        response = await client.messages.create(
            model=self.config.default_model_name,
            max_tokens=self.config.max_tokens,
            messages=messages,
            **create_kwargs,
        )

        text_content = response.content[0].text if response.content else ""
        logger.debug("Response: " + str(text_content))
        page.raw_response = text_content
        text = clean_response(text_content)
        text = html_to_md_keep_tables(text)
        page.text = text

        if response.usage:
            page.prompt_tokens = response.usage.input_tokens
            page.completion_tokens = response.usage.output_tokens

        return page
