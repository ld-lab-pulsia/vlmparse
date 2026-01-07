from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig


class OlmOCRDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for OlmOCR model."""

    model_name: str = "allenai/olmOCR-2-7B-1025-FP8"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--limit-mm-per-prompt",
            '{"image": 1, "video": 0}',
            "--disable-log-requests",
            "--uvicorn-log-level",
            "warning",
            "--max-model-len",
            "16384",
        ]
    )
    aliases: list[str] = Field(default_factory=lambda: ["olmocr-2-fp8"])

    @property
    def client_config(self):
        return OlmOCRConverterConfig(llm_params=self.llm_params)


class OlmOCRConverterConfig(OpenAIConverterConfig):
    """OlmOCR converter"""

    model_name: str = "allenai/olmOCR-2-7B-1025-FP8"
    preprompt: str | None = (
        "Attached is one page of a document that you must process. "
        "Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.\n"
        "If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
        "Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
    )
    postprompt: str | None = None
    completion_kwargs: dict | None = {
        "temperature": 0.1,
        "max_tokens": 8000,
    }
    max_image_size: int | None = 1288
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["olmocr-2-fp8"])
