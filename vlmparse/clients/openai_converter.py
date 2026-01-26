from typing import Literal, Optional

from loguru import logger
from pydantic import Field

from vlmparse.base_model import VLMParseBaseModel
from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import Page
from vlmparse.servers.docker_server import DEFAULT_MODEL_NAME
from vlmparse.utils import to_base64

from .prompts import PDF2MD_PROMPT

GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMParams(VLMParseBaseModel):
    api_key: str = ""
    base_url: str | None = None
    model_name: str = DEFAULT_MODEL_NAME
    timeout: int | None = 500
    max_retries: int = 1


class OpenAIConverterConfig(ConverterConfig):
    llm_params: LLMParams | None = (
        None  # Deprecated, use base_url, model_name, api_key instead
    )
    api_key: str = ""
    timeout: int | None = 500
    max_retries: int = 1
    preprompt: str | None = None
    postprompt: str | None = PDF2MD_PROMPT
    completion_kwargs: dict = Field(default_factory=dict)
    stream: bool = False

    def get_client(self, **kwargs) -> "OpenAIConverterClient":
        return OpenAIConverterClient(config=self, **kwargs)


class OpenAIConverterClient(BaseConverter):
    """Client for OpenAI-compatible API servers."""

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
        from openai import AsyncOpenAI

        self.model = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    async def _get_chat_completion(
        self, messages: list[dict], completion_kwargs: dict | None = None
    ) -> tuple[str, Optional["CompletionUsage"]]:  # noqa: F821
        """Helper to handle chat completion with optional streaming."""
        if completion_kwargs is None:
            completion_kwargs = self.config.completion_kwargs

        if self.config.stream:
            response_stream = await self.model.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                stream=True,
                **completion_kwargs,
            )
            response_parts = []
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_parts.append(chunk.choices[0].delta.content)

            return "".join(response_parts), None
        else:
            response_obj = await self.model.chat.completions.create(
                model=self.config.model_name,
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
        if self.config.preprompt:
            preprompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.config.preprompt}],
                }
            ]
        else:
            preprompt = []

        postprompt = (
            [{"type": "text", "text": self.config.postprompt}]
            if self.config.postprompt
            else []
        )

        messages = [
            *preprompt,
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{to_base64(image)}"
                        },
                    },
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
            page.prompt_tokens = usage.prompt_tokens
            page.completion_tokens = usage.completion_tokens
            if hasattr(usage, "reasoning_tokens"):
                page.reasoning_tokens = usage.reasoning_tokens

        return page
