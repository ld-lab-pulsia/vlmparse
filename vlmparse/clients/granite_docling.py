from pydantic import Field

from vlmparse.clients.openai_converter import (
    OpenAIConverterClient,
    OpenAIConverterConfig,
)
from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.data_model.document import Page
from vlmparse.servers.docker_server import VLLMDockerServerConfig
from vlmparse.utils import to_base64


class GraniteDoclingDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for Granite Docling model."""

    model_name: str = "ibm-granite/granite-docling-258M"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--revision",
            "untied",
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--trust-remote-code",
        ]
    )
    aliases: list[str] = Field(default_factory=lambda: ["granite-docling"])

    @property
    def client_config(self):
        return GraniteDoclingConverterConfig(llm_params=self.llm_params)


class GraniteDoclingConverterConfig(OpenAIConverterConfig):
    """Granite Docling converter configuration."""

    preprompt: str | None = None
    postprompt: str | None = "Convert this page to docling."
    completion_kwargs: dict | None = {
        "temperature": 0.0,
        "max_tokens": 8000,
        "extra_body": {
            "skip_special_tokens": False,
        },
    }
    aliases: list[str] = Field(default_factory=lambda: ["granite-docling"])

    def get_client(self, **kwargs) -> "GraniteDoclingConverter":
        return GraniteDoclingConverter(config=self, **kwargs)


class GraniteDoclingConverter(OpenAIConverterClient):
    """Client for Granite Docling model."""

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{to_base64(image)}"
                        },
                    },
                    {"type": "text", "text": self.config.postprompt},
                ],
            }
        ]

        doctags = await self._get_chat_completion_adaptive(
            messages, completion_kwargs=self.config.completion_kwargs
        )
        doctags = clean_response(doctags)

        page.raw_response = doctags
        page.text = _doctags_to_markdown(doctags, image)
        return page

    async def _get_chat_completion_adaptive(
        self, messages: list[dict], completion_kwargs: dict | None
    ) -> str:
        """
        vLLM enforces input+output <= model context length. If `max_tokens` is too
        high (especially for multimodal prompts), retry with progressively smaller
        `max_tokens`.
        """
        kwargs = (completion_kwargs or {}).copy()
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")

        for _ in range(6):
            try:
                return await self._get_chat_completion(
                    messages, completion_kwargs=kwargs
                )
            except Exception as e:
                msg = str(e)
                too_large = (
                    "max_tokens" in msg
                    and "maximum context length" in msg
                    and "is too large" in msg
                )
                if not too_large or not isinstance(max_tokens, int):
                    raise

                max_tokens = max(256, int(max_tokens * 0.75))
                if "max_tokens" in kwargs:
                    kwargs["max_tokens"] = max_tokens
                if "max_completion_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = max_tokens

        return await self._get_chat_completion(messages, completion_kwargs=kwargs)


def _doctags_to_markdown(doctags: str, image):
    try:
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing optional dependency 'docling-core'. "
            "Install it with: pip install 'vlmparse[docling_core]'"
        ) from e

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")

    html = doc.export_to_html()
    html = clean_response(html)
    md = html_to_md_keep_tables(html, remove_head=True)
    return md
