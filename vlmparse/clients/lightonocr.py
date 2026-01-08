from pathlib import Path

from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig

DOCKERFILE_DIR = Path(__file__).parent.parent.parent / "docker_pipelines"


class LightOnOCRDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for LightOnOCR model."""

    model_name: str = "lightonai/LightOnOCR-1B-1025"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--mm-processor-cache-gb",
            "0",
            "--no-enable-prefix-caching",
        ]
    )
    aliases: list[str] = Field(default_factory=lambda: ["lightonocr"])

    @property
    def client_config(self):
        return LightOnOCRConverterConfig(llm_params=self.llm_params)


class LightOnOCRConverterConfig(OpenAIConverterConfig):
    """LightOnOCR converter - backward compatibility alias."""

    model_name: str = "lightonai/LightOnOCR-1B-1025"
    preprompt: str | None = None
    postprompt: str | None = None
    completion_kwargs: dict | None = {
        "temperature": 0.2,
        "max_tokens": 4096,
        "top_p": 0.9,
    }
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["lightonocr"])
