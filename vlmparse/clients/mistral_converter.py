import asyncio
import inspect
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

MISTRAL_API_BASE_URL = "https://api.mistral.ai"
_MISTRAL_MODEL_NAMES_CACHE: set[str] | None = None
_MISTRAL_OCR_MODELS = {"ocr3", "ocr-3", "mistral-ocr-3"}


def _get_mistral_client(api_key: str, base_url: str | None = None):
    try:
        from mistralai import Mistral
    except ImportError as exc:
        raise ImportError("Please install mistralai to use the client") from exc

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        signature = inspect.signature(Mistral)
        if "server_url" in signature.parameters:
            client_kwargs["server_url"] = base_url
        elif "base_url" in signature.parameters:
            client_kwargs["base_url"] = base_url
    return Mistral(**client_kwargs)


def get_mistral_model_names() -> set[str]:
    """Return available Mistral model IDs, cached for the process lifetime."""
    global _MISTRAL_MODEL_NAMES_CACHE
    if _MISTRAL_MODEL_NAMES_CACHE is not None:
        return _MISTRAL_MODEL_NAMES_CACHE

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        _MISTRAL_MODEL_NAMES_CACHE = set()
        return _MISTRAL_MODEL_NAMES_CACHE

    try:
        client = _get_mistral_client(api_key, base_url=MISTRAL_API_BASE_URL)
        models = client.models.list()
        model_items = getattr(models, "data", models)

        def get_model_id(model: object) -> str | None:
            if hasattr(model, "id"):
                return model.id
            if isinstance(model, dict):
                return model.get("id")
            return None

        _MISTRAL_MODEL_NAMES_CACHE = {
            model_id for model in model_items if (model_id := get_model_id(model))
        }
    except Exception as exc:
        logger.warning("Failed to fetch Mistral models: {}", exc)
        _MISTRAL_MODEL_NAMES_CACHE = set()
    return _MISTRAL_MODEL_NAMES_CACHE


def is_mistral_model(model_name: str) -> bool:
    if model_name in _MISTRAL_OCR_MODELS:
        return True
    return model_name in get_mistral_model_names()


class LLMParams(VLMParseBaseModel):
    api_key: str = os.getenv("MISTRAL_API_KEY")
    base_url: str | None = None
    model_name: str = "ocr3"
    timeout: int | None = 500
    max_retries: int = 1


def get_llm_params(model_name: str):
    if not is_mistral_model(model_name):
        return None
    return LLMParams(model_name=model_name)


class MistralConverterConfig(ConverterConfig):
    llm_params: LLMParams
    preprompt: str | None = None
    postprompt: str | None = PDF2MD_PROMPT
    completion_kwargs: dict = Field(default_factory=dict)
    stream: bool = False

    def get_client(self, **kwargs) -> "MistralConverterClient":
        return MistralConverterClient(config=self, **kwargs)


class MistralConverterClient(BaseConverter):
    """Client for Mistral's OCR API."""

    def __init__(
        self,
        config: MistralConverterConfig,
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
        self.model = _get_mistral_client(
            self.config.llm_params.api_key,
            base_url=self.config.llm_params.base_url,
        )

    async def _run_ocr(self, document: dict) -> object:
        if inspect.iscoroutinefunction(self.model.ocr.process):
            return await self.model.ocr.process(
                model=self.config.llm_params.model_name,
                document=document,
            )
        return await asyncio.to_thread(
            self.model.ocr.process,
            model=self.config.llm_params.model_name,
            document=document,
        )

    def _extract_ocr_text(self, response: object) -> str:
        if hasattr(response, "pages"):
            pages = response.pages
        elif isinstance(response, dict) and "pages" in response:
            pages = response["pages"]
        else:
            pages = None

        if pages:
            page = pages[0]
            for attr in ("markdown", "text", "content"):
                if hasattr(page, attr):
                    value = getattr(page, attr)
                elif isinstance(page, dict):
                    value = page.get(attr)
                else:
                    value = None
                if value:
                    return value

        for attr in ("markdown", "text", "content"):
            if hasattr(response, attr):
                value = getattr(response, attr)
            elif isinstance(response, dict):
                value = response.get(attr)
            else:
                value = None
            if value:
                return value

        return str(response)

    async def async_call_inside_page(self, page: Page) -> Page:
        """Process a single page using the Mistral API."""
        image = page.image
        document = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{to_base64(image)}",
        }

        response = await self._run_ocr(document)
        logger.info("Response: " + str(response))
        page.raw_response = str(response)
        text = clean_response(self._extract_ocr_text(response))

        text = html_to_md_keep_tables(text)
        page.text = text

        return page
