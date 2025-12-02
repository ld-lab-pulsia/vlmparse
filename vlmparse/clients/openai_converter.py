import os
from typing import Literal

from loguru import logger
from pydantic import Field

from vlmparse.base_model import VLMParseBaseModel
from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import Page
from vlmparse.utils import to_base64

from .prompts import PDF2MD_PROMPT

GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMParams(VLMParseBaseModel):
    api_key: str = os.getenv("GOOGLE_API_KEY")
    base_url: str | None = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model_name: str = (
        "gemini-2.5-flash-lite"  # "gemini-2.5-flash-lite" #"gemini-2.5-flash"
    )
    timeout: int | None = 500
    max_retries: int = 1


def get_llm_params(model_name: str, uri: str | None = None):
    if uri is not None:
        return LLMParams(base_url=uri, model_name="vllm-model", api_key="")
    if model_name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ]:
        base_url = None
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        if model_name in [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]:
            base_url = GOOGLE_API_BASE_URL
            api_key = os.getenv("GOOGLE_API_KEY")
        else:
            return None
    return LLMParams(base_url=base_url, model_name=model_name, api_key=api_key)


class OpenAIConverterConfig(ConverterConfig):
    llm_params: LLMParams
    preprompt: str | None = None
    postprompt: str | None = PDF2MD_PROMPT
    completion_kwargs: dict = Field(default_factory=dict)

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
    ):
        super().__init__(
            config=config,
            num_concurrent_files=num_concurrent_files,
            num_concurrent_pages=num_concurrent_pages,
            save_folder=save_folder,
            save_mode=save_mode,
            debug=debug,
        )
        from openai import AsyncOpenAI

        self.model = AsyncOpenAI(
            base_url=self.config.llm_params.base_url,
            api_key=self.config.llm_params.api_key,
            timeout=self.config.llm_params.timeout,
        )

    async def async_call_inside_page(self, page: Page) -> Page:
        """Process a single page using OpenAI-compatible API."""
        image = page.image
        preprompt = (
            [{"type": "text", "text": self.config.preprompt}]
            if self.config.preprompt
            else []
        )
        postprompt = (
            [{"type": "text", "text": self.config.postprompt}]
            if self.config.postprompt
            else []
        )

        messages = [
            {
                "role": "user",
                "content": [
                    *preprompt,
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{to_base64(image)}"
                        },
                    },
                    *postprompt,
                ],
            }
        ]

        response_obj = await self.model.chat.completions.create(
            model=self.config.llm_params.model_name,
            messages=messages,
            **self.config.completion_kwargs,
        )

        response = response_obj.choices[0].message.content
        logger.info("Response: " + str(response))
        page.raw_response = response
        text = clean_response(response)

        text = html_to_md_keep_tables(text)
        page.text = text

        return page
