"""Context manager for deploying a multi-container server group via pure Docker SDK."""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import docker
import docker.errors
import docker.models.containers
import docker.types
from loguru import logger

from .container_group import ContainerGroup
from .docker_run_deployment import _ensure_image_exists

if TYPE_CHECKING:
    from .container_group_server import ContainerGroupServerConfig


def _build_device_requests(gpu_device_ids: list[str] | None) -> list | None:
    """Build Docker device requests for GPU access."""
    if gpu_device_ids is None:
        # All GPUs
        return [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
    if gpu_device_ids and gpu_device_ids[0] != "":
        # Specific GPUs
        return [
            docker.types.DeviceRequest(
                device_ids=gpu_device_ids, capabilities=[["gpu"]]
            )
        ]
    # Empty list → CPU only
    return None


def _gpu_label(gpu_device_ids: list[str] | None) -> str:
    if gpu_device_ids is None:
        return "0"
    if not gpu_device_ids or gpu_device_ids[0] == "":
        return "cpu"
    return ",".join(gpu_device_ids)


@contextmanager
def container_group_server(
    config: "ContainerGroupServerConfig",
    timeout: int = 1000,
    cleanup: bool = True,
):
    """Deploy a multi-container group using pure Docker SDK (no subprocess).

    All services share a bridge network created by :class:`ContainerGroup`.
    The service named by ``config.server_service`` is bound to
    ``config.docker_port``; every other service receives an auto-allocated port
    from ``config.aux_base_port``.

    Args:
        config: Describes all services and which one is the server.
        timeout: Seconds to wait for the server service to become ready.
        cleanup: Stop and remove containers / network on exit when True.

    Yields:
        ``(base_url, server_container)`` tuple.
    """
    group = ContainerGroup(
        name=config.resolved_group_name, base_port=config.docker_port + 1
    )
    server_container = None
    service_containers: dict[str, docker.models.containers.Container] = {}
    client = docker.from_env()

    group.__enter__()
    try:
        uri = f"http://localhost:{config.docker_port}{config.get_base_url_suffix()}"

        # Services that should receive GPU access (default: server service only)
        gpu_services = (
            config.gpu_services
            if config.gpu_services is not None
            else [config.server_service]
        )

        for service_name, service_def in config.services.items():
            is_server = service_name == config.server_service

            # Build image from Dockerfile if requested
            if service_def.dockerfile_dir is not None:
                _ensure_image_exists(
                    client,
                    service_def.image,
                    Path(service_def.dockerfile_dir),
                )

            host_port = config.get_host_port(service_name)

            device_requests = (
                _build_device_requests(config.gpu_device_ids)
                if service_name in gpu_services
                else None
            )

            # Attach identifying labels to the server container
            labels = None
            if is_server:
                labels = {
                    "vlmparse_model_name": config.model_name,
                    "vlmparse_uri": uri,
                    "vlmparse_gpus": _gpu_label(config.gpu_device_ids),
                }

            # Merge shared environment with per-service overrides
            merged_env = {
                **(config.get_environment() or {}),
                **(service_def.environment or {}),
            }

            container, _ = group.run(
                image=service_def.image,
                internal_port=service_def.internal_port,
                host_port=host_port,
                service_name=service_name,
                command=service_def.command,
                environment=merged_env or None,
                volumes=service_def.volumes,
                entrypoint=service_def.entrypoint,
                device_requests=device_requests,
                labels=labels,
                shm_size=service_def.shm_size,
            )

            if is_server:
                server_container = container

            service_containers[service_name] = container

        if server_container is None:
            raise ValueError(
                f"Server service '{config.server_service}' not found in services"
            )

        # Build a map of service_name → required ready indicators.
        # The server service uses config.server_ready_indicators; other
        # services use their own ServiceDefinition.ready_indicators (if any).
        pending: dict[str, list[str]] = {}
        for service_name, service_def in config.services.items():
            if service_name == config.server_service:
                indicators = config.server_ready_indicators
            else:
                indicators = service_def.ready_indicators
            if indicators:
                pending[service_name] = list(indicators)

        pending_services_str = ", ".join(f"'{s}'" for s in pending)
        logger.info(
            f"All services started, waiting for {pending_services_str} to be ready..."
        )

        start_time = time.time()
        log_positions: dict[str, int] = {name: 0 for name in pending}

        while pending and time.time() - start_time < timeout:
            newly_ready = []
            for service_name, indicators in pending.items():
                container = service_containers[service_name]
                try:
                    container.reload()
                except docker.errors.NotFound as e:
                    logger.error(
                        f"Container '{service_name}' stopped unexpectedly during startup"
                    )
                    raise RuntimeError(
                        f"Container '{service_name}' crashed during initialization. "
                        "Check Docker logs for details."
                    ) from e

                if container.status != "running":
                    continue

                all_logs = container.logs().decode("utf-8")
                last_pos = log_positions[service_name]
                if len(all_logs) > last_pos:
                    new_logs = all_logs[last_pos:]
                    for line in new_logs.splitlines():
                        if line.strip():
                            logger.info(f"[{service_name}] {line}")
                    log_positions[service_name] = len(all_logs)

                for indicator in indicators:
                    if indicator in all_logs:
                        logger.info(
                            f"Service '{service_name}' ready (indicator: '{indicator}')"
                        )
                        newly_ready.append(service_name)
                        break

            for name in newly_ready:
                pending.pop(name)

            if pending:
                time.sleep(2)

        if pending:
            raise TimeoutError(
                f"Services {list(pending)} did not become ready within {timeout} seconds"
            )

        base_url = (
            f"http://localhost:{config.docker_port}{config.get_base_url_suffix()}"
        )
        logger.info(f"{config.model_name} server ready at {base_url}")

        yield base_url, server_container

    finally:
        if cleanup:
            group.__exit__(None, None, None)
