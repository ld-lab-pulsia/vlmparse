import os
from typing import Callable

from loguru import logger
from pydantic import BaseModel, Field

from .utils import docker_server


class DockerServerConfig(BaseModel):
    """Base configuration for deploying a Docker server."""

    model_name: str
    docker_image: str
    dockerfile_dir: str | None = None
    command_args: list[str] = Field(default_factory=list)
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: [
            "Application startup complete",
            "Uvicorn running",
            "Starting vLLM API server",
        ]
    )
    docker_port: int = 8056
    gpu_device_ids: list[str] | None = None
    container_port: int = 8000
    environment: dict[str, str] = Field(default_factory=dict)
    volumes: dict[str, dict] | None = None
    entrypoint: str | None = None
    aliases: list[str] = Field(default_factory=list)

    class Config:
        extra = "allow"

    @property
    def client_config(self):
        """Override in subclasses to return appropriate client config."""
        raise NotImplementedError

    def get_client(self, **kwargs):
        return self.client_config.get_client(**kwargs)

    def get_server(self, auto_stop: bool = True):
        return ConverterServer(config=self, auto_stop=auto_stop)

    def get_command(self) -> list[str] | None:
        """Build command for container. Override in subclasses for specific logic."""
        return self.command_args if self.command_args else None

    def get_volumes(self) -> dict | None:
        """Setup volumes for container. Override in subclasses for specific logic."""
        return self.volumes

    def get_environment(self) -> dict | None:
        """Setup environment variables. Override in subclasses for specific logic."""
        return self.environment if self.environment else None

    def get_base_url_suffix(self) -> str:
        """Return URL suffix (e.g., '/v1' for OpenAI-compatible APIs). Override in subclasses."""
        return ""


DEFAULT_MODEL_NAME = "vllm-model"


class VLLMDockerServerConfig(DockerServerConfig):
    """Configuration for deploying a VLLM Docker server."""

    docker_image: str = "vllm/vllm-openai:latest"
    default_model_name: str = DEFAULT_MODEL_NAME
    hf_home_folder: str | None = os.getenv("HF_HOME", None)
    add_model_key_to_server: bool = False
    container_port: int = 8000
    aliases: list[str] = Field(default_factory=list)

    @property
    def llm_params(self):
        from vlmparse.clients.openai_converter import LLMParams

        return LLMParams(
            base_url=f"http://localhost:{self.docker_port}/v1",
            model_name=self.default_model_name,
        )

    @property
    def client_config(self):
        from vlmparse.clients.openai_converter import OpenAIConverterConfig

        return OpenAIConverterConfig(llm_params=self.llm_params)

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


class ConverterServer:
    """Manages Docker server lifecycle with start/stop methods."""

    def __init__(self, config: DockerServerConfig, auto_stop: bool = True):
        self.config = config
        self.auto_stop = auto_stop
        self._server_context = None
        self._container = None
        self.base_url = None

    def start(self):
        """Start the Docker server."""
        if self._server_context is not None:
            logger.warning("Server already started")
            return self.base_url

        # Use the generic docker_server for all server types
        self._server_context = docker_server(config=self.config, cleanup=self.auto_stop)

        self.base_url, self._container = self._server_context.__enter__()
        logger.info(f"Server started at {self.base_url}")
        logger.info(f"Container ID: {self._container.id}")
        logger.info(f"Container name: {self._container.name}")
        return self.base_url, self._container

    def stop(self):
        """Stop the Docker server."""
        if self._server_context is not None:
            self._server_context.__exit__(None, None, None)
            self._server_context = None
            self._container = None
            self.base_url = None
            logger.info("Server stopped")

    def __del__(self):
        """Automatically stop server when object is destroyed if auto_stop is True."""
        if self.auto_stop and self._server_context is not None:
            self.stop()


class DockerConfigRegistry:
    """Registry for mapping model names to their Docker configurations."""

    def __init__(self):
        self._registry = dict()

    def register(
        self, model_name: str, config_factory: Callable[[], DockerServerConfig | None]
    ):
        """Register a config factory for a model name."""
        self._registry[model_name] = config_factory

    def get(self, model_name: str, default=False) -> DockerServerConfig | None:
        """Get config for a model name. Returns default if not registered."""
        if model_name not in self._registry:
            if default:
                return VLLMDockerServerConfig(model_name=model_name)
            return None
        return self._registry[model_name]()

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._registry.keys())


# Global registry instance
docker_config_registry = DockerConfigRegistry()
