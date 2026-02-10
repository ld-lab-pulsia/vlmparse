"""Base classes for server configurations and server lifecycle management."""

from abc import ABC, abstractmethod

from loguru import logger
from pydantic import Field

from .model_identity import ModelIdentityMixin


class BaseServerConfig(ModelIdentityMixin, ABC):
    """Base configuration for deploying a server.

    Inherits from ModelIdentityMixin which provides:
    - model_name: str
    - default_model_name: str | None
    - aliases: list[str]
    - _create_client_kwargs(base_url): Helper for creating client configs
    - get_all_names(): All names this model can be referenced by

    All server configs should inherit from this base class.
    """

    docker_port: int = 8056
    container_port: int = 8000
    gpu_device_ids: list[str] | None = None
    environment: dict[str, str] = Field(default_factory=dict)
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
    @abstractmethod
    def client_config(self):
        """Override in subclasses to return appropriate client config."""
        raise NotImplementedError

    def get_client(self, **kwargs):
        """Get a client instance configured for this server."""
        return self.client_config.get_client(**kwargs)

    @abstractmethod
    def get_server(self, auto_stop: bool = True):
        """Get a server instance for this configuration."""
        raise NotImplementedError

    def get_environment(self) -> dict | None:
        """Setup environment variables. Override in subclasses for specific logic."""
        return self.environment if self.environment else None

    def get_base_url_suffix(self) -> str:
        """Return URL suffix (e.g., '/v1' for OpenAI-compatible APIs). Override in subclasses."""
        return ""

    def update_command_args(
        self,
        vllm_args: list[str] | None = None,
        forget_predefined_vllm_args: bool = False,
    ) -> list[str]:
        """Update command arguments. Override in subclasses that support this."""
        _ = vllm_args, forget_predefined_vllm_args
        return []


class BaseServer(ABC):
    """Base class for managing server lifecycle with start/stop methods.

    All server implementations should inherit from this class.
    """

    def __init__(self, config: BaseServerConfig, auto_stop: bool = True):
        self.config = config
        self.auto_stop = auto_stop
        self._server_context = None
        self._container = None
        self.base_url = None

    @abstractmethod
    def _create_server_context(self):
        """Create the appropriate server context. Override in subclasses."""
        raise NotImplementedError

    def start(self):
        """Start the server."""
        if self._server_context is not None:
            logger.warning("Server already started")
            return self.base_url, self._container

        self._server_context = self._create_server_context()
        self.base_url, self._container = self._server_context.__enter__()
        logger.info(f"Server started at {self.base_url}")
        if self._container is not None:
            logger.info(f"Container ID: {self._container.id}")
            logger.info(f"Container name: {self._container.name}")
        return self.base_url, self._container

    def stop(self):
        """Stop the server."""
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
