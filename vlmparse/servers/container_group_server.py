"""ContainerGroupServer: pure-SDK multi-container server (no docker compose subprocess)."""

import getpass
from pathlib import Path

from pydantic import BaseModel, Field

from .base_server import BaseServer, BaseServerConfig
from .container_group_deployment import container_group_server


class ServiceDefinition(BaseModel):
    """Definition of a single service within a ContainerGroup deployment.

    Attributes:
        image: Docker image to run.
        internal_port: Port the service listens on **inside** the container.
            The host-side (external) port is managed by the group and never
            set here – see :meth:`ContainerGroupServerConfig.get_host_port`.
        command: Optional command override.
        environment: Per-service environment variables (merged with the shared
            environment defined on the server config).
        volumes: Volume mounts in docker SDK format.
        entrypoint: Entrypoint override.
        dockerfile_dir: If set, build *image* from this Dockerfile directory
            when the image is not already present locally.
    """

    image: str
    internal_port: int
    command: list[str] | str | None = None
    environment: dict[str, str] | None = None
    volumes: dict[str, dict] | None = None
    entrypoint: list[str] | str | None = None
    dockerfile_dir: Path | None = None
    shm_size: str | None = None
    ready_indicators: list[str] = Field(default_factory=list)


class ContainerGroupServerConfig(BaseServerConfig):
    """Configuration for a multi-container server deployed via pure Docker SDK.

    Unlike :class:`~vlmparse.servers.docker_compose_server.DockerComposeServerConfig`,
    this uses only the Docker Python SDK – no ``docker compose`` subprocess calls.

    **Port design** – only ONE host-side port needs to be configured:

    * ``docker_port`` is the **external** (host) port for the primary service.
    * All other services receive host ports ``docker_port+1``, ``docker_port+2``,
      … in their definition order.  Use :meth:`get_host_port` to look them up.
    * ``ServiceDefinition.internal_port`` is always the port **inside** the
      container; it is independent of the host port.

    The service identified by *server_service* is the primary container:

    * It is bound to ``docker_port``.
    * Its logs are polled for *server_ready_indicators*.
    * It receives the ``vlmparse_*`` Docker labels.

    Attributes:
        services: Ordered mapping of service name → :class:`ServiceDefinition`.
        server_service: Key in *services* that is the primary server.
        group_name: Docker network / container name prefix.  Auto-generated
            from ``model_name`` and the current user when not set.
        gpu_services: Services that receive GPU device requests.  Defaults to
            ``[server_service]`` when *None*.
    """

    services: dict[str, ServiceDefinition]
    server_service: str
    group_name: str | None = None
    gpu_services: list[str] | None = None

    @property
    def resolved_group_name(self) -> str:
        """Docker-safe group name derived from model_name and current user."""
        return (
            self.group_name
            or f"vlmparse-{self.model_name.replace('/', '-')}-{getpass.getuser()}"
        )

    def get_host_port(self, service_name: str) -> int:
        """Return the host-side port for *service_name*.

        The server service maps to ``docker_port``.  Every other service is
        assigned ``docker_port + 1``, ``docker_port + 2``, … in definition order.
        """
        if service_name == self.server_service:
            return self.docker_port
        aux_services = [n for n in self.services if n != self.server_service]
        try:
            idx = aux_services.index(service_name)
        except ValueError:
            raise KeyError(f"Service '{service_name}' not found in services") from None
        return self.docker_port + 1 + idx

    @property
    def client_config(self):
        """Override in subclasses to return an appropriate client config."""
        raise NotImplementedError

    def _client_config_from_service_urls(self, service_urls: dict[str, str]):
        """Build a client config from a fully-resolved {service→url} mapping.

        Override in subclasses that expose multiple services (e.g. layout +
        vLLM).  The base implementation just uses the primary server URL.
        """
        return self.client_config.model_copy(
            update={"base_url": service_urls[self.server_service]}
        )

    def client_config_for_uri(self, uri: str):
        """Build a client config targeting *uri* as the primary service.

        Extracts host, scheme, and port from *uri* so that every auxiliary
        service URL is constructed with the same host/scheme but a port
        derived via :meth:`get_host_port`.  This works whether the containers
        are on localhost or a remote host.
        """
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        host = parsed.hostname or "localhost"
        scheme = parsed.scheme or "http"
        port = parsed.port or self.docker_port

        original_port = self.docker_port
        self.docker_port = port
        try:
            service_urls = {
                name: f"{scheme}://{host}:{self.get_host_port(name)}"
                for name in self.services
            }
        finally:
            self.docker_port = original_port

        return self._client_config_from_service_urls(service_urls)

    def get_server(self, auto_stop: bool = True) -> "ContainerGroupServer":
        return ContainerGroupServer(config=self, auto_stop=auto_stop)


class ContainerGroupServer(BaseServer):
    """Manages a :class:`ContainerGroup`-based server lifecycle."""

    def _create_server_context(self):
        assert isinstance(self.config, ContainerGroupServerConfig)
        return container_group_server(config=self.config, cleanup=self.auto_stop)
