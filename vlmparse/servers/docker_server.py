import os

from pydantic import Field

from .base_server import BaseServer, BaseServerConfig
from .docker_run_deployment import docker_server


class DockerServerConfig(BaseServerConfig):
    """Configuration for deploying a Docker server.

    Inherits from BaseServerConfig which provides common server configuration.
    """

    docker_image: str
    dockerfile_dir: str | None = None
    command_args: list[str] = Field(default_factory=list)
    volumes: dict[str, dict] | None = None
    entrypoint: str | None = None

    @property
    def client_config(self):
        """Override in subclasses to return appropriate client config."""
        raise NotImplementedError

    def get_server(self, auto_stop: bool = True):
        return ConverterServer(config=self, auto_stop=auto_stop)

    def get_command(self) -> list[str] | None:
        """Build command for container. Override in subclasses for specific logic."""
        return self.command_args if self.command_args else None

    def update_command_args(
        self,
        vllm_args: dict | None = None,
        forget_predefined_vllm_args: bool = False,
    ) -> list[str]:
        if vllm_args is not None:
            if forget_predefined_vllm_args:
                self.command_args = vllm_args
            else:
                self.command_args.extend(vllm_args)

        return self.command_args

    def get_volumes(self) -> dict | None:
        """Setup volumes for container. Override in subclasses for specific logic."""
        return self.volumes


DEFAULT_MODEL_NAME = "vllm-model"


class VLLMDockerServerConfig(DockerServerConfig):
    """Configuration for deploying a VLLM Docker server."""

    docker_image: str = "vllm/vllm-openai:latest"
    default_model_name: str = DEFAULT_MODEL_NAME
    hf_home_folder: str | None = os.getenv("HF_HOME", None)
    add_model_key_to_server: bool = False
    container_port: int = 8000

    @property
    def client_config(self):
        from vlmparse.clients.openai_converter import OpenAIConverterConfig

        return OpenAIConverterConfig(
            **self._create_client_kwargs(
                f"http://localhost:{self.docker_port}{self.get_base_url_suffix()}"
            )
        )

    def get_command(self) -> list[str]:
        """Build VLLM-specific command."""
        model_key = ["--model"] if self.add_model_key_to_server else []
        command = (
            model_key
            + [
                self.model_name,
                "--port",
                str(self.container_port),
            ]
            + self.command_args
            + ["--served-model-name", self.default_model_name]
        )
        return command

    def get_volumes(self) -> dict | None:
        """Setup volumes for HuggingFace model caching."""
        if self.hf_home_folder is not None:
            from pathlib import Path

            return {
                str(Path(self.hf_home_folder).absolute()): {
                    "bind": "/root/.cache/huggingface",
                    "mode": "rw",
                }
            }
        return None

    def get_environment(self) -> dict | None:
        """Setup environment variables for VLLM."""
        if self.hf_home_folder is not None:
            return {
                "HF_HOME": self.hf_home_folder,
                "TRITON_CACHE_DIR": self.hf_home_folder,
            }
        return None

    def get_base_url_suffix(self) -> str:
        """VLLM uses OpenAI-compatible API with /v1 suffix."""
        return "/v1"


class ConverterServer(BaseServer):
    """Manages Docker server lifecycle with start/stop methods."""

    def _create_server_context(self):
        """Create the Docker server context."""
        return docker_server(config=self.config, cleanup=self.auto_stop)
