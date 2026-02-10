import getpass
import os
import re
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import docker
from loguru import logger

if TYPE_CHECKING:
    from .docker_compose_server import DockerComposeServerConfig


def _sanitize_compose_project_name(name: str, fallback: str = "vlmparse") -> str:
    """Return a Docker Compose-compatible project name.

    Compose requires only lowercase letters, numbers, hyphens, and underscores,
    and the name must start with a letter or number.
    """

    if not name:
        return fallback

    sanitized = re.sub(r"[^a-z0-9_-]+", "-", name.lower())
    sanitized = re.sub(r"^[^a-z0-9]+", "", sanitized)
    sanitized = sanitized.strip("-_")

    return sanitized or fallback


def _build_compose_override_yaml(config: "DockerComposeServerConfig") -> str | None:
    services_overrides: dict[str, dict] = {}

    service_names = config.compose_services or [config.server_service]

    # Labels for model/uri inference (match docker_server behavior)
    uri = f"http://localhost:{config.docker_port}{config.get_base_url_suffix()}"
    if config.gpu_device_ids is None:
        gpu_label = "0"
    elif len(config.gpu_device_ids) == 0 or (
        len(config.gpu_device_ids) == 1 and config.gpu_device_ids[0] == ""
    ):
        gpu_label = "cpu"
    else:
        gpu_label = ",".join(config.gpu_device_ids)
    if config.server_service:
        services_overrides.setdefault(config.server_service, {}).setdefault(
            "labels", {}
        ).update(
            {
                "vlmparse_model_name": config.model_name,
                "vlmparse_uri": uri,
                "vlmparse_gpus": gpu_label,
                "vlmparse_deployment": "docker_compose",
                "vlmparse_compose_project": str(config.compose_project_name or ""),
                "vlmparse_compose_file": str(config.compose_file or ""),
                "vlmparse_compose_server_service": str(config.server_service or ""),
            }
        )

    # Port override for the server service
    if config.server_service:
        services_overrides.setdefault(config.server_service, {})["ports"] = [
            f"{config.docker_port}:{config.container_port}"
        ]

    # Environment overrides
    environment = config.get_environment()
    if environment:
        env_targets = (
            config.environment_services
            if config.environment_services is not None
            else [config.server_service]
        )
        for service in env_targets:
            services_overrides.setdefault(service, {})["environment"] = environment

    # GPU overrides
    if config.gpu_device_ids is not None:
        gpu_targets = (
            config.gpu_service_names
            if config.gpu_service_names is not None
            else service_names
        )

        if len(config.gpu_device_ids) == 0 or (
            len(config.gpu_device_ids) == 1 and config.gpu_device_ids[0] == ""
        ):
            devices_value = []
        else:
            devices_value = [
                {
                    "driver": "nvidia",
                    "device_ids": config.gpu_device_ids,
                    "capabilities": ["gpu"],
                }
            ]

        for service in gpu_targets:
            services_overrides.setdefault(service, {}).setdefault("deploy", {})
            services_overrides[service]["deploy"].setdefault("resources", {})
            services_overrides[service]["deploy"]["resources"].setdefault(
                "reservations", {}
            )
            services_overrides[service]["deploy"]["resources"]["reservations"][
                "devices"
            ] = devices_value

    if not services_overrides:
        return None

    # Manual YAML rendering for the limited structure we need.
    lines = ["services:"]
    for service, overrides in services_overrides.items():
        lines.append(f"  {service}:")

        if "ports" in overrides:
            # Compose override files normally merge/append list fields (like ports).
            # Use !override so we replace the base compose list instead of adding
            # extra published ports.
            lines.append("    ports: !override")
            for port in overrides["ports"]:
                lines.append(f'      - "{port}"')

        if "environment" in overrides:
            lines.append("    environment:")
            for key, value in overrides["environment"].items():
                lines.append(f'      {key}: "{value}"')

        if "labels" in overrides:
            lines.append("    labels:")
            for key, value in overrides["labels"].items():
                lines.append(f'      {key}: "{value}"')

        if "deploy" in overrides:
            lines.append("    deploy:")
            lines.append("      resources:")
            lines.append("        reservations:")
            # Same story as ports: ensure we replace the base list.
            lines.append("          devices: !override")
            devices = overrides["deploy"]["resources"]["reservations"].get("devices")
            if not devices:
                lines.append("            []")
            else:
                for device in devices:
                    lines.append("            - driver: nvidia")
                    lines.append("              device_ids:")
                    for device_id in device.get("device_ids", []):
                        lines.append(f'                - "{device_id}"')
                    lines.append("              capabilities:")
                    for cap in device.get("capabilities", []):
                        lines.append(f"                - {cap}")

    return "\n".join(lines) + "\n"


def _get_compose_container(
    client: docker.DockerClient, project_name: str, service: str
):
    containers = client.containers.list(
        all=True,
        filters={
            "label": [
                f"com.docker.compose.project={project_name}",
                f"com.docker.compose.service={service}",
            ]
        },
    )
    if not containers:
        return None
    return containers[0]


def _start_compose_logs_stream(
    compose_cmd: list[str],
    *,
    env: dict | None,
    model_name: str,
    services: list[str] | None = None,
    tail: int = 200,
) -> tuple[subprocess.Popen[str], threading.Event, threading.Thread] | None:
    """Stream `docker compose logs -f` output into loguru.

    This is useful for compose stacks where the actual failure happens in a
    dependency service (db, redis, etc). Output is forwarded line-by-line so it
    shows up similarly to the single-container `docker_server` startup logs.

    Returns a (process, stop_event, thread) triple, or None if the stream can't
    be started.
    """

    cmd = list(compose_cmd) + [
        "logs",
        "--no-color",
        "--follow",
        "--tail",
        str(tail),
    ]
    if services:
        cmd.extend([s for s in services if s])

    try:
        proc: subprocess.Popen[str] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
    except Exception as e:
        logger.debug(f"Failed to start docker compose log stream: {e}")
        return None

    stop_event = threading.Event()

    def _pump_logs() -> None:
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                if stop_event.is_set():
                    break
                line = line.rstrip("\n")
                if line.strip():
                    # Compose already prefixes with service name; keep a stable
                    # model prefix so users can correlate when multiplexed.
                    logger.info(f"[{model_name}] {line}")
        except Exception as e:
            logger.debug(f"Compose log stream stopped: {e}")
        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass

    thread = threading.Thread(target=_pump_logs, name="compose-logs", daemon=True)
    thread.start()
    return proc, stop_event, thread


def _stop_compose_logs_stream(
    stream: tuple[subprocess.Popen[str], threading.Event, threading.Thread] | None,
    *,
    timeout: float = 2.0,
) -> None:
    if stream is None:
        return

    proc, stop_event, thread = stream
    stop_event.set()
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    thread.join(timeout=timeout)


@contextmanager
def docker_compose_server(
    config: "DockerComposeServerConfig",
    timeout: int = 1000,
    cleanup: bool = True,
):
    """Generic context manager for Docker Compose server deployment.

    Args:
        config: DockerComposeServerConfig
        timeout: Timeout in seconds to wait for server to be ready
        cleanup: If True, stop and remove containers on exit. If False, leave running

    Yields:
        tuple: (base_url, container) - The base URL of the server and the container object
    """

    compose_file = Path(config.compose_file).expanduser().resolve()
    if not compose_file.exists():
        raise FileNotFoundError(f"Compose file not found at {compose_file}")

    project_name = _sanitize_compose_project_name(
        config.compose_project_name
        or f"vlmparse-{config.model_name.replace('/', '-')}-{getpass.getuser()}"
    )

    # Persist resolved name so it can be attached as a label via the override YAML.
    # This allows `vlmparse stop` to bring down the full compose stack.
    config.compose_project_name = project_name

    base_cmd = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "--project-name",
        project_name,
    ]

    client = docker.from_env()
    container = None

    override_content = _build_compose_override_yaml(config)
    compose_env = config.get_compose_env()
    env = None
    if compose_env:
        env = os.environ.copy()
        env.update(compose_env)

    with tempfile.TemporaryDirectory() as temp_dir:
        override_file = None
        if override_content:
            override_file = Path(temp_dir) / "compose.override.yaml"
            override_file.write_text(override_content)

        compose_cmd = list(base_cmd)
        if override_file is not None:
            compose_cmd.extend(["-f", str(override_file)])

        logs_stream = None

        try:
            logger.info(
                f"Starting Docker Compose for {config.model_name} on port {config.docker_port}"
            )

            try:
                subprocess.run(
                    compose_cmd + ["up", "-d"],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except subprocess.CalledProcessError as e:
                if e.stdout:
                    logger.error(
                        f"Docker Compose stdout for {config.model_name}:\n{e.stdout}"
                    )
                if e.stderr:
                    logger.error(
                        f"Docker Compose stderr for {config.model_name}:\n{e.stderr}"
                    )

                # Best-effort diagnostics: container status and last logs.
                # This is especially helpful when dependencies crash quickly (e.g. exit 137).
                try:
                    ps_res = subprocess.run(
                        compose_cmd + ["ps"],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    if ps_res.stdout:
                        logger.error(
                            f"Docker Compose ps for {config.model_name}:\n{ps_res.stdout}"
                        )
                    if ps_res.stderr:
                        logger.error(
                            f"Docker Compose ps stderr for {config.model_name}:\n{ps_res.stderr}"
                        )

                    logs_res = subprocess.run(
                        compose_cmd + ["logs", "--no-color", "--tail", "200"],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    if logs_res.stdout:
                        logger.error(
                            f"Docker Compose logs (tail=200) for {config.model_name}:\n{logs_res.stdout}"
                        )
                    if logs_res.stderr:
                        logger.error(
                            f"Docker Compose logs stderr for {config.model_name}:\n{logs_res.stderr}"
                        )
                except Exception as diag_error:
                    logger.warning(
                        f"Failed to collect docker compose diagnostics: {diag_error}"
                    )
                raise

            logger.info("Compose stack started, waiting for server to be ready...")

            # Stream logs for the whole compose stack (or requested services).
            # This mirrors the visibility you get with `docker_server` while
            # being much more useful for multi-service compose deployments.
            service_names = (
                config.compose_services
                if config.compose_services
                else ([config.server_service] if config.server_service else None)
            )
            logs_stream = _start_compose_logs_stream(
                compose_cmd,
                env=env,
                model_name=config.model_name,
                services=service_names,
                tail=200,
            )

            start_time = time.time()
            server_ready = False
            last_log_position = 0

            while time.time() - start_time < timeout:
                container = _get_compose_container(
                    client, project_name, config.server_service
                )

                if container is None:
                    time.sleep(2)
                    continue

                try:
                    container.reload()
                except docker.errors.NotFound as e:  # type: ignore[name-defined]
                    logger.error("Container stopped unexpectedly during startup")
                    raise RuntimeError(
                        "Container crashed during initialization. Check Docker logs for details."
                    ) from e

                if container.status == "running":
                    all_logs = container.logs().decode("utf-8")
                    # Keep the old log-position tracker for readiness checks.
                    # Actual log visibility is handled by the compose log stream.
                    if len(all_logs) > last_log_position:
                        last_log_position = len(all_logs)

                    for indicator in config.server_ready_indicators:
                        if indicator in all_logs:
                            server_ready = True
                            break

                    if not server_ready:
                        health = container.attrs.get("State", {}).get("Health", {})
                        if health.get("Status") == "healthy":
                            server_ready = True

                    if server_ready:
                        logger.info(
                            f"Server ready indicator found for service '{config.server_service}'"
                        )
                        break

                time.sleep(2)

            if not server_ready:
                raise TimeoutError(
                    f"Server did not become ready within {timeout} seconds"
                )

            base_url = (
                f"http://localhost:{config.docker_port}{config.get_base_url_suffix()}"
            )
            logger.info(f"{config.model_name} server ready at {base_url}")

            # Deployment phase is over: stop log streaming so we don't
            # continuously forward runtime logs (caller can still fetch logs
            # explicitly via Docker / CLI if needed).
            _stop_compose_logs_stream(logs_stream)
            logs_stream = None

            yield base_url, container

        finally:
            _stop_compose_logs_stream(logs_stream)
            if cleanup:
                logger.info("Stopping Docker Compose stack")
                subprocess.run(
                    compose_cmd + ["down"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                logger.info("Compose stack stopped")
