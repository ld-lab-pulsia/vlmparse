from pydantic import Field

from .base_server import BaseServer, BaseServerConfig
from .docker_compose_deployment import docker_compose_server


class DockerComposeServerConfig(BaseServerConfig):
    """Configuration for deploying a Docker Compose server.

    Inherits from BaseServerConfig which provides common server configuration.
    """

    compose_file: str
    server_service: str
    compose_services: list[str] = Field(default_factory=list)
    compose_project_name: str | None = None
    compose_env: dict[str, str] = Field(default_factory=dict)
    gpu_service_names: list[str] | None = None
    environment_services: list[str] | None = None

    @property
    def client_config(self):
        """Override in subclasses to return appropriate client config."""
        raise NotImplementedError

    def get_server(self, auto_stop: bool = True):
        return ComposeServer(config=self, auto_stop=auto_stop)

    def get_compose_env(self) -> dict | None:
        """Environment variables for docker compose command substitution."""
        return self.compose_env if self.compose_env else None


class ComposeServer(BaseServer):
    """Manages Docker Compose server lifecycle with start/stop methods."""

    def _create_server_context(self):
        """Create the Docker Compose server context."""
        return docker_compose_server(config=self.config, cleanup=self.auto_stop)
