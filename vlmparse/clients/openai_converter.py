import asyncio
import os
from typing import Literal, Optional

from loguru import logger
from openai.types.completion import CompletionUsage
from openai.types.responses import ResponseUsage
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import Page
from vlmparse.utils import to_base64

from .prompts import PDF2MD_PROMPT

GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class OpenAIConverterConfig(ConverterConfig):
    api_key: str | None = None
    is_azure: bool = False
    timeout: int | None = 500
    max_retries: int = 1
    preprompt: str | None = None
    postprompt: str | dict[str, str] | None = PDF2MD_PROMPT
    prompts: dict[str, str] = Field(default_factory=dict)
    prompt_mode_map: dict[str, str] = Field(default_factory=dict)
    completion_kwargs: dict = Field(default_factory=dict)
    stream: bool = False
    use_response_api: bool = False

    def get_client(self, **kwargs) -> "OpenAIConverterClient":
        return OpenAIConverterClient(config=self, **kwargs)


class OpenAIConverterClient(BaseConverter):
    """Client for OpenAI-compatible API servers."""

    config: OpenAIConverterConfig

    def get_prompt_key(self) -> str | None:
        """Resolve a prompt key from conversion_mode using class mappings."""
        mode = getattr(self.config, "conversion_mode", None) or "ocr"
        prompts = self._get_prompts()
        if mode in prompts:
            return mode
        mapped = self._get_prompt_mode_map().get(mode)
        if mapped in prompts:
            return mapped
        return None

    def get_prompt_for_mode(self) -> str | None:
        key = self.get_prompt_key()
        if key is None:
            return None
        return self._get_prompts().get(key)

    def _get_prompts(self) -> dict[str, str]:
        if self.config.prompts:
            return self.config.prompts
        if isinstance(self.config.postprompt, dict):
            return self.config.postprompt
        return {}

    def _get_prompt_mode_map(self) -> dict[str, str]:
        return self.config.prompt_mode_map or {}

    def __init__(
        self,
        config: OpenAIConverterConfig,
        num_concurrent_files: int = 10,
        num_concurrent_pages: int = 10,
        save_folder: str | None = None,
        save_mode: Literal["document", "md", "md_page"] = "document",
        debug: bool = False,
        return_documents_in_batch_mode: bool = False,
    ):
        super().__init__(
            config=config,
            num_concurrent_files=num_concurrent_files,
            num_concurrent_pages=num_concurrent_pages,
            save_folder=save_folder,
            save_mode=save_mode,
            debug=debug,
            return_documents_in_batch_mode=return_documents_in_batch_mode,
        )
        self._model = None
        self._model_loop = None

    async def _get_async_model(self):
        loop = asyncio.get_running_loop()
        if self._model is None or self._model_loop is not loop:
            await self._close_model()
            from openai import AsyncAzureOpenAI, AsyncOpenAI

            if self.config.is_azure:
                assert (
                    self.config.base_url is not None
                ), "Azure OpenAI requires a base URL"

                self._model = AsyncAzureOpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                    api_version=os.getenv(
                        "AZURE_OPENAI_API_VERSION", "2025-04-01-preview"
                    ),
                )
            else:
                self._model = AsyncOpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
            self._model_loop = loop
        return self._model

    async def _close_model(self):
        """Close the async OpenAI client if it exists."""
        if self._model is not None:
            try:
                await self._model.close()
            except RuntimeError:
                # Event loop may already be closed
                pass
            finally:
                self._model = None
                self._model_loop = None

    async def aclose(self):
        """Close the converter and release resources."""
        await self._close_model()

    def close(self):
        """Synchronously close the converter if possible."""
        if self._model is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._close_model())
            except RuntimeError:
                # No running loop, try to close synchronously
                try:
                    asyncio.run(self._close_model())
                except RuntimeError:
                    # Event loop already closed, force cleanup
                    self._model = None
                    self._model_loop = None

    async def _get_chat_completion(
        self, messages: list[dict], completion_kwargs: dict | None = None
    ) -> tuple[str, Optional[CompletionUsage | ResponseUsage]]:  # noqa: F821
        """Helper to handle chat completion with optional streaming."""
        if completion_kwargs is None:
            completion_kwargs = self.config.completion_kwargs

        model = await self._get_async_model()
        if self.config.use_response_api:
            response_obj = await model.responses.create(
                model=self.config.default_model_name,
                input=messages,
                **completion_kwargs,
            )

            out = [o for o in response_obj.output if o.type == "message"][0]

            if out.content[0].text is None:
                raise ValueError("Response is None, finish reason: ")

            return out.content[0].text, response_obj.usage
        else:
            if self.config.stream:
                response_stream = await model.responses.create(
                    model=self.config.default_model_name,
                    input=messages,
                    stream=True,
                    **completion_kwargs,
                )
                response_parts = []
                async for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        response_parts.append(chunk.choices[0].delta.content)

                return "".join(response_parts), None
            else:
                response_obj = await model.chat.completions.create(
                    model=self.config.default_model_name,
                    messages=messages,
                    **completion_kwargs,
                )

                if response_obj.choices[0].message.content is None:
                    raise ValueError(
                        "Response is None, finish reason: "
                        + response_obj.choices[0].finish_reason
                    )

                return response_obj.choices[0].message.content, response_obj.usage

    async def async_call_inside_page(self, page: Page) -> Page:
        """Process a single page using OpenAI-compatible API."""
        image = page.image
        assert image is not None, "Page image is required for conversion"
        if self.config.preprompt:
            preprompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.config.preprompt}],
                }
            ]
        else:
            preprompt = []

        selected_prompt = self.get_prompt_for_mode()
        if self.config.use_response_api:
            text_key = "input_text"
            image_payload = {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{to_base64(image)}",
            }
        else:
            text_key = "text"
            image_payload = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{to_base64(image)}"},
            }
        if selected_prompt is not None:
            postprompt = [{"type": text_key, "text": selected_prompt}]
        else:
            postprompt = (
                [{"type": text_key, "text": self.config.postprompt}]
                if isinstance(self.config.postprompt, str) and self.config.postprompt
                else []
            )

        messages = [
            *preprompt,
            {
                "role": "user",
                "content": [
                    image_payload,
                    *postprompt,
                ],
            },
        ]

        response, usage = await self._get_chat_completion(messages)
        logger.debug("Response: " + str(response))
        page.raw_response = response
        text = clean_response(response)

        text = html_to_md_keep_tables(text)
        page.text = text
        if usage is not None:
            if isinstance(usage, ResponseUsage):
                page.prompt_tokens = usage.input_tokens
                page.completion_tokens = usage.output_tokens
            else:
                page.prompt_tokens = usage.prompt_tokens
                page.completion_tokens = usage.completion_tokens

        return page
