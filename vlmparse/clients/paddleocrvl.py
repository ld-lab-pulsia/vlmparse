from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig


class PaddleOCRVLDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for PaddleOCRVL model."""

    model_name: str = "PaddlePaddle/PaddleOCR-VL"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--async-scheduling",
            "--trust-remote-code",
            "--mm-processor-cache-gb",
            "0",
        ]
    )
    aliases: list[str] = Field(default_factory=lambda: ["paddleocrvl"])

    @property
    def client_config(self):
        return PaddleOCRVLConverterConfig(llm_params=self.llm_params)


# Task-specific base prompts
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


class PaddleOCRVLConverterConfig(OpenAIConverterConfig):
    """PaddleOCRVL converter"""

    model_name: str = "PaddlePaddle/PaddleOCR-VL"
    preprompt: str | None = None
    postprompt: str | None = TASKS["ocr"]
    completion_kwargs: dict | None = {
        "temperature": 0.0,
    }
    max_image_size: int | None = 1540
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["paddleocrvl"])
