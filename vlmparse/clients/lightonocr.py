from pathlib import Path

from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig

DOCKERFILE_DIR = Path(__file__).parent.parent.parent / "docker_pipelines"


class LightOnOCRDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for LightOnOCR model."""

    model_name: str = "lightonai/LightOnOCR-1B-1025"
    docker_image: str = "lightonocr:latest"
    dockerfile_dir: str = str(DOCKERFILE_DIR / "lightonocr")
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--async-scheduling",
        ]
    )

    @property
    def client_config(self):
        return LightOnOCRConverterConfig(llm_params=self.llm_params)


class LightOnOCRConverterConfig(OpenAIConverterConfig):
    """LightOnOCR converter - backward compatibility alias."""

    preprompt: str | None = ""
    postprompt: str | None = None
    completion_kwargs: dict | None = {"temperature": 0.2}
    max_image_size: int | None = 1540
    dpi: int = 200
