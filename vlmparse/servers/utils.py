import getpass
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

import docker


def _ensure_image_exists(
    client: docker.DockerClient,
    image: str,
    dockerfile_path: Path,
):
    """Check if image exists, build it if not."""
    try:
        client.images.get(image)
        logger.info(f"Docker image {image} found")
        return
    except docker.errors.ImageNotFound:
        logger.info(f"Docker image {image} not found, building...")

        if not dockerfile_path.exists():
            raise FileNotFoundError(
                f"Dockerfile directory not found at {dockerfile_path}"
            ) from None

        logger.info(f"Building image from {dockerfile_path}")

        # Use low-level API for real-time streaming
        api_client = docker.APIClient(base_url="unix://var/run/docker.sock")

        # Build the image with streaming
        build_stream = api_client.build(
            path=str(dockerfile_path),
            tag=image,
            rm=True,
            decode=True,  # Automatically decode JSON responses to dict
        )

        # Stream build logs in real-time
        for chunk in build_stream:
            if "stream" in chunk:
                for line in chunk["stream"].splitlines():
                    logger.info(line)
            elif "error" in chunk:
                logger.error(chunk["error"])
                raise docker.errors.BuildError(chunk["error"], build_stream) from None
            elif "status" in chunk:
                # Handle status updates (e.g., downloading layers)
                logger.debug(chunk["status"])

        logger.info(f"Successfully built image {image}")


@contextmanager
def docker_server(
    config: "DockerServerConfig",  # noqa: F821
    timeout: int = 1000,
    cleanup: bool = True,
):
    """Generic context manager for Docker server deployment.

    Args:
        config: DockerServerConfig (can be VLLMDockerServerConfig or GenericDockerServerConfig)
        timeout: Timeout in seconds to wait for server to be ready
        cleanup: If True, stop and remove container on exit. If False, leave container running

    Yields:
        tuple: (base_url, container) - The base URL of the server and the Docker container object
    """

    client = docker.from_env()
    container = None

    try:
        # Ensure image exists
        logger.info(f"Checking for Docker image {config.docker_image}...")

        if config.dockerfile_dir is not None:
            _ensure_image_exists(
                client, config.docker_image, Path(config.dockerfile_dir)
            )
        else:
            # Pull pre-built image
            try:
                client.images.get(config.docker_image)
                logger.info(f"Docker image {config.docker_image} found locally")
            except docker.errors.ImageNotFound:
                logger.info(
                    f"Docker image {config.docker_image} not found locally, pulling..."
                )
                client.images.pull(config.docker_image)
                logger.info(f"Successfully pulled {config.docker_image}")

        logger.info(
            f"Starting Docker container for {config.model_name} on port {config.docker_port}"
        )

        # Configure GPU access
        device_requests = None

        if config.gpu_device_ids is None:
            # Default: Try to use all GPUs if available
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        elif len(config.gpu_device_ids) > 0 and config.gpu_device_ids[0] != "":
            # Use specific GPU devices
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=config.gpu_device_ids, capabilities=[["gpu"]]
                )
            ]
        else:
            # Empty list means CPU-only, no GPU
            device_requests = None

        # Use generic methods from config
        command = config.get_command()
        volumes = config.get_volumes()
        environment = config.get_environment()
        container_port = config.container_port
        log_prefix = config.model_name

        # Construct URI for label
        uri = f"http://localhost:{config.docker_port}{config.get_base_url_suffix()}"

        # Determine GPU label
        if config.gpu_device_ids is None:
            gpu_label = "all"
        elif len(config.gpu_device_ids) == 0 or (
            len(config.gpu_device_ids) == 1 and config.gpu_device_ids[0] == ""
        ):
            gpu_label = "cpu"
        else:
            gpu_label = ",".join(config.gpu_device_ids)

        # Start container
        container_kwargs = {
            "image": config.docker_image,
            "ports": {f"{container_port}/tcp": config.docker_port},
            "detach": True,
            "remove": True,
            "name": f"vlmparse-{config.model_name.replace('/', '-')}-{getpass.getuser()}",
            "labels": {
                "vlmparse_model_name": config.model_name,
                "vlmparse_uri": uri,
                "vlmparse_gpus": gpu_label,
            },
        }

        if device_requests is not None:
            container_kwargs["device_requests"] = device_requests
        if command:
            container_kwargs["command"] = command
        if environment:
            container_kwargs["environment"] = environment
        if volumes:
            container_kwargs["volumes"] = volumes
        if config.entrypoint:
            container_kwargs["entrypoint"] = config.entrypoint

        container = client.containers.run(**container_kwargs)

        logger.info(
            f"Container {container.short_id} started, waiting for server to be ready..."
        )

        # Wait for server to be ready
        start_time = time.time()
        server_ready = False
        last_log_position = 0

        while time.time() - start_time < timeout:
            try:
                container.reload()
            except docker.errors.NotFound as e:
                logger.error("Container stopped unexpectedly during startup")
                raise RuntimeError(
                    "Container crashed during initialization. Check Docker logs for details."
                ) from e

            if container.status == "running":
                # Get all logs and display new ones
                all_logs = container.logs().decode("utf-8")

                # Display new log lines
                if len(all_logs) > last_log_position:
                    new_logs = all_logs[last_log_position:]
                    for line in new_logs.splitlines():
                        if line.strip():  # Only print non-empty lines
                            logger.info(f"[{log_prefix}] {line}")
                    last_log_position = len(all_logs)

                # Check if server is ready
                for indicator in config.server_ready_indicators:
                    if indicator in all_logs:
                        server_ready = True
                if server_ready:
                    logger.info(f"Server ready indicator '{indicator}' found in logs")
                    break

            time.sleep(2)

        if not server_ready:
            raise TimeoutError(f"Server did not become ready within {timeout} seconds")

        # Build base URL using config's suffix method
        base_url = (
            f"http://localhost:{config.docker_port}{config.get_base_url_suffix()}"
        )

        logger.info(f"{log_prefix} server ready at {base_url}")

        yield base_url, container

    finally:
        if cleanup and container:
            logger.info(f"Stopping container {container.short_id}")
            container.stop(timeout=10)
            logger.info("Container stopped")


def get_model_from_uri(uri: str) -> str:
    model = None
    client = docker.from_env()
    containers = client.containers.list()
    for container in containers:
        c_uri = container.labels.get("vlmparse_uri")
        c_model = container.labels.get("vlmparse_model_name")

        # Check if user URI matches container URI (ignoring /v1 suffix if missing)
        if c_uri and (
            c_uri == uri or c_uri.startswith(uri.rstrip("/")) or uri.startswith(c_uri)
        ):
            # Update URI to the correct one from container (likely has /v1)
            if len(c_uri) > len(uri.rstrip("/")):
                logger.info(f"Updating URI from {uri} to {c_uri}")
                uri = c_uri

            # Infer model if not provided
            if model is None and c_model:
                logger.info(f"Inferred model {c_model} from container")
                model = c_model
            break
    if model is None:
        raise ValueError(f"No model found for URI {uri}")
    return model
