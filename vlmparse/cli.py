from typing import Literal

from loguru import logger


class DParseCLI:
    def serve(self, model: str, port: int | None = None, gpus: str | None = None):
        """Deploy a VLLM server in a Docker container.

        Args:
            model: Model name
            port: VLLM server port (default: 8056)
            gpus: Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, all GPUs will be used.
        """
        if port is None:
            port = 8056

        from vlmparse.registries import docker_config_registry

        docker_config = docker_config_registry.get(model)
        if docker_config is None:
            logger.warning(
                f"No Docker configuration found for model: {model}, using default configuration"
            )
            return

        docker_config.docker_port = port

        # Only override GPU configuration if explicitly specified
        # This preserves CPU-only settings from the config
        if gpus is not None:
            docker_config.gpu_device_ids = [g.strip() for g in str(gpus).split(",")]
        server = docker_config.get_server(auto_stop=False)

        # Deploy server and leave it running (cleanup=False)
        logger.info(
            f"Deploying VLLM server for {docker_config.model_name} on port {port}..."
        )

        base_url, container = server.start()

        logger.info(f"✓ VLLM server ready at {base_url}")
        logger.info(f"✓ Container ID: {container.id}")
        logger.info(f"✓ Container name: {container.name}")

    def convert(
        self,
        inputs: str | list[str],
        out_folder: str = ".",
        model: str | None = None,
        uri: str | None = None,
        gpus: str | None = None,
        mode: Literal["document", "md", "md_page"] = "document",
        with_vllm_server: bool = False,
        concurrency: int = 10,
        dpi: int | None = None,
    ):
        """Parse PDF documents and save results.

        Args:
            inputs: List of folders to process
            out_folder: Output folder for parsed documents
            pipe: Converter type ("vllm", "openai", or "lightonocr", default: "vllm")
            model: Model name. If not specified, the model will be inferred from the URI.
            uri: URI of the server, if not specified and the pipe is vllm, a local server will be deployed
            gpus: Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, all GPUs will be used.
            mode: Output mode - "document" (save as JSON zip), "md" (save as markdown file), "md_page" (save as folder of markdown pages)
            with_vllm_server: If True, a local VLLM server will be deployed if the model is not found in the registry. Note that if the model is in the registry and the uri is None, the server will be anyway deployed.
            dpi: DPI to use for the conversion. If not specified, the default DPI will be used.
        """
        from vlmparse.converter_with_server import ConverterWithServer

        converter_with_server = ConverterWithServer(
            model=model,
            uri=uri,
            gpus=gpus,
            with_vllm_server=with_vllm_server,
            concurrency=concurrency,
        )

        return converter_with_server.parse(
            inputs=inputs, out_folder=out_folder, mode=mode, dpi=dpi
        )
        # from vlmparse.registries import converter_config_registry

        # # Infer model from URI if provided
        # if uri is not None and model is None:
        #     import docker

        #     try:
        #         docker_client = docker.from_env()
        #         containers = docker_client.containers.list()
        #         for container in containers:
        #             # Check both exact match and match with/without trailing slash
        #             container_uri = container.labels.get("vlmparse_uri", "")
        #             if container_uri and (
        #                 container_uri == uri
        #                 or container_uri.rstrip("/") == uri.rstrip("/")
        #             ):
        #                 inferred_model = container.labels.get("vlmparse_model_name")
        #                 if inferred_model:
        #                     logger.info(
        #                         f"Inferred model {inferred_model} from URI {uri}"
        #                     )
        #                     model = inferred_model
        #                     break
        #     except Exception:
        #         # If Docker is not available or fails, just proceed with provided arguments
        #         pass

        # if mode not in ["document", "md", "md_page"]:
        #     logger.error(f"Invalid mode: {mode}. Must be one of: document, md, md_page")
        #     return

        # # Expand file paths from glob patterns
        # file_paths = []
        # if isinstance(inputs, str):
        #     inputs = [inputs]
        # for pattern in inputs:
        #     if "*" in pattern or "?" in pattern:
        #         file_paths.extend(glob(pattern, recursive=True))
        #     elif os.path.isdir(pattern):
        #         file_paths.extend(glob(os.path.join(pattern, "*.pdf"), recursive=True))
        #     elif os.path.isfile(pattern):
        #         file_paths.append(pattern)
        #     else:
        #         logger.error(f"Invalid input: {pattern}")

        # # Filter to only existing PDF files
        # file_paths = [f for f in file_paths if os.path.exists(f) and f.endswith(".pdf")]

        # if not file_paths:
        #     logger.error("No PDF files found matching the inputs patterns")
        #     return

        # logger.info(f"Processing {len(file_paths)} files with {model} converter")

        # gpu_device_ids = None
        # if gpus is not None:
        #     gpu_device_ids = [g.strip() for g in gpus.split(",")]

        # if uri is None:
        #     from vlmparse.registries import docker_config_registry

        #     docker_config = docker_config_registry.get(model, default=with_vllm_server)

        #     if docker_config is not None:
        #         docker_config.gpu_device_ids = gpu_device_ids
        #         server = docker_config.get_server(auto_stop=True)
        #         server.start()

        #         client = docker_config.get_client(
        #             save_folder=out_folder, save_mode=mode
        #         )
        #     else:
        #         client = converter_config_registry.get(model).get_client(
        #             save_folder=out_folder, save_mode=mode
        #         )

        # else:
        #     client_config = converter_config_registry.get(model, uri=uri)
        #     client = client_config.get_client(save_folder=out_folder, save_mode=mode)
        # client.num_concurrent_files = concurrency
        # client.num_concurrent_pages = concurrency
        # if dpi is not None:
        #     client.config.dpi = int(dpi)
        # documents = client.batch(file_paths)

        # if documents is not None:
        #     logger.info(f"Processed {len(documents)} documents to {out_folder}")
        # else:
        #     logger.info(f"Processed {len(file_paths)} documents to {out_folder}")

    def list(self):
        """List all containers whose name begins with vlmparse."""
        import docker

        try:
            client = docker.from_env()
            containers = client.containers.list()

            if not containers:
                logger.info("No running containers found")
                return

            # Filter for containers whose name begins with "vlmparse"
            vlmparse_containers = [
                container
                for container in containers
                if container.name.startswith("vlmparse")
            ]

            if not vlmparse_containers:
                logger.info("No vlmparse containers found")
                return

            # Prepare table data
            table_data = []
            for container in vlmparse_containers:
                # Extract port mappings
                ports = []
                if container.ports:
                    for _, host_bindings in container.ports.items():
                        if host_bindings:
                            for binding in host_bindings:
                                ports.append(f"{binding['HostPort']}")

                port_str = ", ".join(set(ports)) if ports else "N/A"
                uri = container.labels.get("vlmparse_uri", "N/A")
                gpu = container.labels.get("vlmparse_gpus", "N/A")

                table_data.append(
                    [
                        container.name,
                        container.status,
                        port_str,
                        gpu,
                        uri,
                    ]
                )

            # Display as table
            from tabulate import tabulate

            headers = ["Name", "Status", "Port(s)", "GPU", "URI"]
            table = tabulate(table_data, headers=headers, tablefmt="grid")

            logger.info(f"\nFound {len(vlmparse_containers)} vlmparse container(s):\n")
            print(table)

        except docker.errors.DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            logger.error(
                "Make sure Docker is running and you have the necessary permissions"
            )

    def stop(self, container: str | None = None):
        """Stop a Docker container by its ID or name.

        Args:
            container: Container ID or name to stop. If not specified, automatically stops the container if only one vlmparse container is running.
        """
        import docker

        try:
            client = docker.from_env()

            # If no container specified, try to auto-select
            if container is None:
                containers = client.containers.list()
                vlmparse_containers = [
                    c for c in containers if c.name.startswith("vlmparse")
                ]

                if len(vlmparse_containers) == 0:
                    logger.error("No vlmparse containers found")
                    return
                elif len(vlmparse_containers) > 1:
                    logger.error(
                        f"Multiple vlmparse containers found ({len(vlmparse_containers)}). "
                        "Please specify a container ID or name:"
                    )
                    for c in vlmparse_containers:
                        logger.info(f"  - {c.name} ({c.short_id})")
                    return
                else:
                    target_container = vlmparse_containers[0]
            else:
                # Try to get the specified container
                try:
                    target_container = client.containers.get(container)
                except docker.errors.NotFound:
                    logger.error(f"Container not found: {container}")
                    return

            # Stop the container
            logger.info(
                f"Stopping container: {target_container.name} ({target_container.short_id})"
            )
            target_container.stop()
            logger.info("✓ Container stopped successfully")

        except docker.errors.DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            logger.error(
                "Make sure Docker is running and you have the necessary permissions"
            )

    def log(self, container: str | None = None, follow: bool = True, tail: int = 500):
        """Show logs from a Docker container.

        Args:
            container: Container ID or name. If not specified, automatically selects the container if only one vlmparse container is running.
            follow: If True, follow log output (stream logs in real-time)
            tail: Number of lines to show from the end of the logs
        """
        import docker

        try:
            client = docker.from_env()

            # If no container specified, try to auto-select
            if container is None:
                containers = client.containers.list()
                vlmparse_containers = [
                    c for c in containers if c.name.startswith("vlmparse")
                ]

                if len(vlmparse_containers) == 0:
                    logger.error("No vlmparse containers found")
                    return
                elif len(vlmparse_containers) > 1:
                    logger.error(
                        f"Multiple vlmparse containers found ({len(vlmparse_containers)}). "
                        "Please specify a container ID or name:"
                    )
                    for c in vlmparse_containers:
                        logger.info(f"  - {c.name} ({c.short_id})")
                    return
                else:
                    target_container = vlmparse_containers[0]
                    logger.info(
                        f"Showing logs for: {target_container.name} ({target_container.short_id})"
                    )
            else:
                # Try to get the specified container
                try:
                    target_container = client.containers.get(container)
                except docker.errors.NotFound:
                    logger.error(f"Container not found: {container}")
                    return

            # Get and display logs
            if follow:
                logger.info("Following logs (press Ctrl+C to stop)...")
                try:
                    for log_line in target_container.logs(
                        stream=True, follow=True, tail=tail
                    ):
                        print(log_line.decode("utf-8", errors="replace"), end="")
                except KeyboardInterrupt:
                    logger.info("\nStopped following logs")
            else:
                logs = target_container.logs().decode("utf-8", errors="replace")
                print(logs)

        except docker.errors.DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            logger.error(
                "Make sure Docker is running and you have the necessary permissions"
            )

    def list_register(self):
        """List all model keys registered in client and server registries."""
        from vlmparse.registries import (
            converter_config_registry,
            docker_config_registry,
        )

        client_models = sorted(converter_config_registry.list_models())
        server_models = sorted(docker_config_registry.list_models())

        print("\nClient Models Registry:")
        for model in client_models:
            print(f"  - {model}")

        print("\nServer Models Registry:")
        for model in server_models:
            print(f"  - {model}")

    def view(self, folder):
        import subprocess
        import sys

        from streamlit import runtime

        from vlmparse.st_viewer.st_viewer import __file__ as st_viewer_file
        from vlmparse.st_viewer.st_viewer import run_streamlit

        if runtime.exists():
            run_streamlit(folder)
        else:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        st_viewer_file,
                        "--",
                        folder,
                    ],
                    check=True,
                )
            except KeyboardInterrupt:
                print("\nStreamlit app terminated by user.")
            except subprocess.CalledProcessError as e:
                print(f"Error while running Streamlit: {e}")


def main():
    import fire

    fire.Fire(DParseCLI)


if __name__ == "__main__":
    main()
