from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig


class HunyuanOCRDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for HunyuanOCR model."""

    model_name: str = "tencent/HunyuanOCR"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--async-scheduling",
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb",
            "0",
            # Default argument in the hunyuan model, not sure why it is set this low.
            "--gpu-memory-utilization",
            "0.2",
        ]
    )
    aliases: list[str] = Field(default_factory=lambda: ["hunyuanocr"])

    @property
    def client_config(self):
        return HunyuanOCRConverterConfig(
            **self._create_client_kwargs(
                f"http://localhost:{self.docker_port}{self.get_base_url_suffix()}"
            )
        )


class HunyuanOCRConverterConfig(OpenAIConverterConfig):
    """HunyuanOCR converter"""

    model_name: str = "tencent/HunyuanOCR"
    preprompt: str | None = ""
    postprompt: str | None = (
        "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."
    )
    completion_kwargs: dict | None = {
        "temperature": 0.0,
        "extra_body": {"top_k": 1, "repetition_penalty": 1.0},
        "max_completion_tokens": 16384,  # max token len used in training according to the technical report is 32000, but in practice the model breaks earlier
    }
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["hunyuanocr"])
    stream: bool = True
