from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig


class QianfanOCRDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for Qianfan-OCR model (baidu/Qianfan-OCR).

    A 4B-parameter end-to-end document intelligence model that performs
    direct image-to-Markdown conversion. Requires --trust-remote-code for vLLM.

    Reference: https://huggingface.co/baidu/Qianfan-OCR
    """

    model_name: str = "baidu/Qianfan-OCR"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--trust-remote-code",
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--max-model-len",
            "32768",
        ]
    )
    aliases: list[str] = Field(default_factory=lambda: ["qianfanocr"])

    @property
    def client_config(self):
        return QianfanOCRConverterConfig(
            **self._create_client_kwargs(
                f"http://localhost:{self.docker_port}{self.get_base_url_suffix()}"
            )
        )


class QianfanOCRConverterConfig(OpenAIConverterConfig):
    """Converter config for Qianfan-OCR.

    Standard mode: prompts the model to parse the document to Markdown.
    """

    model_name: str = "baidu/Qianfan-OCR"
    preprompt: str | None = None
    postprompt: str | None = "Parse this document to Markdown."
    completion_kwargs: dict | None = {
        "temperature": 0.0,
        "max_tokens": 16384,
    }
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["qianfanocr"])


class QianfanOCRThinkingDockerServerConfig(QianfanOCRDockerServerConfig):
    """Qianfan-OCR with Layout-as-Thought (thinking mode) enabled.

    The model generates structured layout analysis via <think> tokens before
    producing the final Markdown output. Useful for complex, heterogeneous pages.
    """

    aliases: list[str] = Field(default_factory=lambda: ["qianfanocr-thinking"])

    @property
    def client_config(self):
        return QianfanOCRThinkingConverterConfig(
            **self._create_client_kwargs(
                f"http://localhost:{self.docker_port}{self.get_base_url_suffix()}"
            )
        )


class QianfanOCRThinkingConverterConfig(QianfanOCRConverterConfig):
    """Qianfan-OCR with Layout-as-Thought (thinking mode) enabled.

    Appends <think> to the prompt to trigger the layout-analysis thinking phase.
    """

    postprompt: str | None = "Parse this document to Markdown.<think>"
    aliases: list[str] = Field(default_factory=lambda: ["qianfanocr-thinking"])
