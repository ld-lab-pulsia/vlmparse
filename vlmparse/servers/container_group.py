"""ContainerGroup: Python-native Docker multi-container manager.

Provides a shared bridge network and sequential host-port allocation for a set
of containers that belong to the same logical deployment group.
"""

import docker
import docker.models.containers
import docker.models.networks
from loguru import logger


class ContainerGroup:
    """Manages a set of Docker containers sharing an isolated bridge network.

    Ports for containers without an explicit host binding are allocated
    sequentially starting from *base_port*.

    Can be used as a context manager – the network is created on ``__enter__``
    and the whole group (containers + network) is torn down on ``__exit__``.

    Example::

        with ContainerGroup("my-stack", base_port=9000) as group:
            container, port = group.run("redis:7", container_port=6379)
            ...
    """

    def __init__(self, name: str, base_port: int = 9000) -> None:
        self.name = name
        self.base_port = base_port
        self._port_offset = 0
        self._client = docker.from_env()
        self._network: docker.models.networks.Network | None = None
        self._containers: dict[str, docker.models.containers.Container] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ContainerGroup":
        self._network = self._client.networks.create(self.name, driver="bridge")
        logger.info(f"Created Docker network '{self.name}'")
        return self

    def __exit__(self, *exc_info) -> None:
        self.teardown()

    # ------------------------------------------------------------------
    # Port allocation
    # ------------------------------------------------------------------

    def next_port(self) -> int:
        """Return the next available sequential host port."""
        port = self.base_port + self._port_offset
        self._port_offset += 1
        return port

    # ------------------------------------------------------------------
    # Container management
    # ------------------------------------------------------------------

    def run(
        self,
        image: str,
        internal_port: int,
        *,
        host_port: int | None = None,
        service_name: str | None = None,
        command: list[str] | str | None = None,
        environment: dict[str, str] | None = None,
        volumes: dict | None = None,
        entrypoint: list[str] | str | None = None,
        device_requests: list | None = None,
        labels: dict[str, str] | None = None,
        shm_size: str | None = None,
        remove: bool = False,
    ) -> tuple:
        """Start a container inside this group.

        Args:
            image: Docker image to run.
            internal_port: Port the service listens on **inside** the container.
            host_port: Host-side port to publish. Auto-allocated when *None*.
            service_name: Logical name used to build the container name.
            command: Command override.
            environment: Environment variables passed to the container.
            volumes: Volume mounts (docker SDK format).
            entrypoint: Entrypoint override.
            device_requests: GPU / device requests.
            labels: Docker labels attached to the container.
            remove: Auto-remove the container when it exits.

        Returns:
            ``(container, host_port)`` tuple.
        """
        if self._network is None:
            raise RuntimeError(
                "ContainerGroup network not initialised – call __enter__ first"
            )

        if host_port is None:
            host_port = self.next_port()

        slug = (service_name or image).replace("/", "-").replace(":", "-")
        container_name = f"{self.name}-{slug}"

        run_kwargs: dict = {
            "image": image,
            "detach": True,
            "network": self._network.name,
            "ports": {f"{internal_port}/tcp": host_port},
            "name": container_name,
            "remove": remove,
        }
        if command:
            run_kwargs["command"] = command
        if environment:
            run_kwargs["environment"] = environment
        if volumes:
            run_kwargs["volumes"] = volumes
        if entrypoint:
            run_kwargs["entrypoint"] = entrypoint
        if device_requests:
            run_kwargs["device_requests"] = device_requests
        if labels:
            run_kwargs["labels"] = labels
        if shm_size:
            run_kwargs["shm_size"] = shm_size

        container = self._client.containers.run(**run_kwargs)
        self._containers[container_name] = container
        logger.info(
            f"Started container '{container_name}' ({container.short_id}) "
            f"on host port {host_port}"
        )
        return container, host_port

    def teardown(self) -> None:
        """Stop all containers in the group and remove the shared network."""
        for name, container in list(self._containers.items()):
            try:
                container.stop(timeout=10)
                logger.info(f"Stopped container '{name}'")
            except Exception as e:
                logger.warning(f"Error stopping container '{name}': {e}")
        self._containers.clear()

        if self._network is not None:
            try:
                self._network.remove()
                logger.info(f"Removed Docker network '{self.name}'")
            except Exception as e:
                logger.warning(f"Error removing Docker network '{self.name}': {e}")
            self._network = None
