import time
from contextlib import contextmanager
from pathlib import Path


import docker
from loguru import logger


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
            raise FileNotFoundError(f"Dockerfile directory not found at {dockerfile_path}")
        
        logger.info(f"Building image from {dockerfile_path}")
        
        # Use low-level API for real-time streaming
        api_client = docker.APIClient(base_url='unix://var/run/docker.sock')
        
        # Build the image with streaming
        build_stream = api_client.build(
            path=str(dockerfile_path),
            tag=image,
            rm=True,
            decode=True  # Automatically decode JSON responses to dict
        )
        
        # Stream build logs in real-time
        for chunk in build_stream:
            if 'stream' in chunk:
                for line in chunk['stream'].splitlines():
                    logger.info(line)
            elif 'error' in chunk:
                logger.error(chunk['error'])
                raise docker.errors.BuildError(chunk['error'], build_stream)
            elif 'status' in chunk:
                # Handle status updates (e.g., downloading layers)
                logger.debug(chunk['status'])
        
        logger.info(f"Successfully built image {image}")


@contextmanager
def vllm_server(
    config: "DockerServerConfig",
    timeout: int = 500,
    cleanup: bool = True,
):
    """Generic context manager for VLLM server deployment.

    Args:
        config: VLLMModelConfig with model-specific settings (includes port, gpu_device_ids, dockerfile_dir)
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

            _ensure_image_exists(client, config.docker_image, Path(config.dockerfile_dir))
        else:
            # Pull pre-built image (for standard VLLM images)
            try:
                client.images.get(config.docker_image)
                logger.info(f"Docker image {config.docker_image} found locally")
            except docker.errors.ImageNotFound:
                logger.info(f"Docker image {config.docker_image} not found locally, pulling...")
                client.images.pull(config.docker_image)
                logger.info(f"Successfully pulled {config.docker_image}")
        
        logger.info(f"Starting VLLM container for {config.model_name} on port {config.docker_port}")
        
        # Configure GPU access
        device_requests = None
        if config.gpu_device_ids:
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=config.gpu_device_ids,
                    capabilities=[["gpu"]]
                )
            ]
        else:
            device_requests = [
                docker.types.DeviceRequest(
                    count=-1,
                    capabilities=[["gpu"]]
                )
            ]
        
        # Build command
        model_key = ["--model"] if config.add_model_key_to_vllm_server else []

        command = model_key+[
            config.model_name,
            "--port", "8000",
        ] + config.command_args+["--served-model-name", config.default_model_name]
        
        # Setup volumes for model caching
        if config.hf_home_folder is not None:
            volumes = {
                str(Path(config.hf_home_folder).absolute()): {
                    'bind': '/root/.cache/huggingface',
                    'mode': 'rw'
                }
            }
            env = {"HF_HOME": config.hf_home_folder, "TRITON_CACHE_DIR": config.hf_home_folder}
        else:
            volumes = None
            env = None
        # Start container
        container = client.containers.run(
            config.docker_image,
            command=command,
            ports={"8000/tcp": config.docker_port},
            device_requests=device_requests,
            volumes=volumes,
            detach=True,
            remove=True,
            environment=env,
        )
        
        logger.info(f"Container {container.short_id} started, waiting for server to be ready...")
        
        # Wait for server to be ready
        start_time = time.time()
        server_ready = False
        last_log_position = 0
        
        while time.time() - start_time < timeout:
            try:
                container.reload()
            except docker.errors.NotFound:
                logger.error("Container stopped unexpectedly during startup")
                raise RuntimeError("Container crashed during initialization. Check Docker logs for details.")
            
            if container.status == "running":
                # Get all logs and display new ones
                all_logs = container.logs().decode("utf-8")
                
                # Display new log lines
                if len(all_logs) > last_log_position:
                    new_logs = all_logs[last_log_position:]
                    for line in new_logs.splitlines():
                        if line.strip():  # Only print non-empty lines
                            logger.info(f"[VLLM] {line}")
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
        
        base_url = f"http://localhost:{config.docker_port}/v1"
        logger.info(f"VLLM server ready at {base_url}\nConvert document with dparse convert --input /path/to/your/document.pdf --output /path/to/output/folder --model {config.model_name} --uri {base_url}")
        
        yield base_url, container
        
    finally:
        if cleanup and container:
            logger.info(f"Stopping container {container.short_id}")
            container.stop(timeout=10)
            logger.info("Container stopped")


