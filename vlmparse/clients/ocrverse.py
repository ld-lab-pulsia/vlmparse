from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig


class OCRVerseDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for OCRVerse model."""

    model_name: str = "DocTron/OCRVerse"
    aliases: list[str] = Field(default_factory=lambda: ["ocrverse"])

    @property
    def client_config(self):
        return OCRVerseConverterConfig(
            **self._create_client_kwargs(
                f"http://localhost:{self.docker_port}{self.get_base_url_suffix()}"
            )
        )


class OCRVerseConverterConfig(OpenAIConverterConfig):
    """Configuration for OCRVerse model.

    OCRVerse is a holistic OCR model supporting both text-centric (documents,
    tables, formulas) and vision-centric (charts, web pages, scientific plots)
    tasks. The model is based on Qwen3VL and was trained with the
    ``qwen3_vl_nothink`` template, so thinking mode is disabled during inference.

    Reference: https://github.com/DocTron-hub/OCRVerse
    """

    model_name: str = "DocTron/OCRVerse"
    preprompt: str | None = None
    postprompt: str | None = (
        "Extract the main content from the document in the image, keeping the original structure. Convert all formulas to LaTeX and all tables to HTML."
    )
    completion_kwargs: dict | None = {
        "temperature": 0.0,
        "max_completion_tokens": 8192,
        # OCRVerse is based on Qwen3VL trained with the nothink template;
        # disable the built-in thinking chain so the output is plain text.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["ocrverse"])
