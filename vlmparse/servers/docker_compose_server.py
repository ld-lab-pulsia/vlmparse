from loguru import logger
from pydantic import Field

from .docker_compose_deployment import docker_compose_server
from .model_identity import ModelIdentityMixin


class DockerComposeServerConfig(ModelIdentityMixin):
    """Base configuration for deploying a Docker Compose server.

    Inherits from ModelIdentityMixin which provides:
    - model_name: str
    - default_model_name: str | None
    - aliases: list[str]
    - _create_client_kwargs(base_url): Helper for creating client configs
    - get_all_names(): All names this model can be referenced by
    """

    compose_file: str
    server_service: str
    compose_services: list[str] = Field(default_factory=list)
    compose_project_name: str | None = None
    compose_env: dict[str, str] = Field(default_factory=dict)
    docker_port: int = 8056
    container_port: int = 8000
    gpu_device_ids: list[str] | None = None
    gpu_service_names: list[str] | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    environment_services: list[str] | None = None
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: [
            "Application startup complete",
            "Uvicorn running",
            "Starting vLLM API server",
        ]
    )

    class Config:
        extra = "allow"

    @property
    def client_config(self):
        """Override in subclasses to return appropriate client config."""
        raise NotImplementedError

    def get_client(self, **kwargs):
        return self.client_config.get_client(**kwargs)

    def get_server(self, auto_stop: bool = True):
        return ComposeServer(config=self, auto_stop=auto_stop)

    def get_environment(self) -> dict | None:
        """Setup environment variables. Override in subclasses for specific logic."""
        return self.environment if self.environment else None

    def get_compose_env(self) -> dict | None:
        """Environment variables for docker compose command substitution."""
        return self.compose_env if self.compose_env else None

    def update_command_args(
        self,
        vllm_args: dict | None = None,
        forget_predefined_vllm_args: bool = False,
    ) -> list[str]:
        """Compatibility no-op for compose configs."""
        _ = vllm_args, forget_predefined_vllm_args
        return []

    def get_base_url_suffix(self) -> str:
        """Return URL suffix (e.g., '/v1' for OpenAI-compatible APIs). Override in subclasses."""
        return ""


class ComposeServer:
    """Manages Docker Compose server lifecycle with start/stop methods."""

    def __init__(self, config: DockerComposeServerConfig, auto_stop: bool = True):
        self.config = config
        self.auto_stop = auto_stop
        self._server_context = None
        self._container = None
        self.base_url = None

    def start(self):
        """Start the Docker Compose server."""
        if self._server_context is not None:
            logger.warning("Server already started")
            return self.base_url, self._container

        self._server_context = docker_compose_server(
            config=self.config, cleanup=self.auto_stop
        )

        self.base_url, self._container = self._server_context.__enter__()
        logger.info(f"Server started at {self.base_url}")
        if self._container is not None:
            logger.info(f"Container ID: {self._container.id}")
            logger.info(f"Container name: {self._container.name}")
        return self.base_url, self._container

    def stop(self):
        """Stop the Docker Compose server."""
        if self._server_context is not None:
            try:
                self._server_context.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during server cleanup: {e}")
            finally:
                self._server_context = None
                self._container = None
                self.base_url = None
            logger.info("Server stopped")

    def __del__(self):
        """Automatically stop server when object is destroyed if auto_stop is True.

        Note: This is a fallback mechanism. Prefer using the context manager
        or explicitly calling stop() for reliable cleanup.
        """
        try:
            if self.auto_stop and self._server_context is not None:
                self.stop()
        except Exception:
            pass  # Suppress errors during garbage collection
