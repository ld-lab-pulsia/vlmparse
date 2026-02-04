# ruff: noqa: B008
from typing import Literal

import typer
from loguru import logger

app = typer.Typer(help="Parse PDF documents with VLMs.", pretty_exceptions_enable=False)


@app.command("serve")
def serve(
    model: str = typer.Argument(..., help="Model name"),
    port: int | None = typer.Option(None, help="VLLM server port (default: 8056)"),
    gpus: str | None = typer.Option(
        None,
        help='Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, all GPUs will be used.',
    ),
    vllm_args: list[str] | None = typer.Option(
        None,
        "--vllm-args",
        help="Additional keyword arguments to pass to the VLLM server.",
    ),
    forget_predefined_vllm_args: bool = typer.Option(
        False,
        help=(
            "If True, the predefined VLLM kwargs from the docker config will be replaced by vllm_args. "
            "Otherwise they will be updated with vllm_args (may overwrite keys)."
        ),
    ),
):
    """Deploy a VLLM server in a Docker container.

    Args:
        model: Model name
        port: VLLM server port (default: 8056)
        gpus: Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, GPU 0 will be used.
        vllm_args: Additional keyword arguments to pass to the VLLM server.
        forget_predefined_vllm_args: If True, the predefined VLLM kwargs from the docker config will be replaced by vllm_args otherwise the predefined kwargs will be updated with vllm_args with a risk of collision of argument names.
    """

    from vlmparse.converter_with_server import start_server

    base_url, container, _, _ = start_server(
        model=model,
        gpus=gpus,
        port=port,
        with_vllm_server=True,
        vllm_args=vllm_args,
        forget_predefined_vllm_args=forget_predefined_vllm_args,
        auto_stop=False,
    )

    logger.info(f"✓ VLLM server ready at {base_url}")
    if container is not None:
        logger.info(f"✓ Container ID: {container.id}")
        logger.info(f"✓ Container name: {container.name}")


@app.command("convert")
def convert(
    inputs: str = typer.Argument(..., help="List of folders to process"),
    out_folder: str = typer.Option(".", help="Output folder for parsed documents"),
    model: str | None = typer.Option(
        None, help="Model name. If not specified, inferred from the URI."
    ),
    uri: str | None = typer.Option(
        None,
        help=(
            "URI of the server. If not specified and the pipe is vllm, "
            "a local server will be deployed."
        ),
    ),
    gpus: str | None = typer.Option(
        None,
        help='Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, GPU 0 will be used.',
    ),
    mode: Literal["document", "md", "md_page"] = typer.Option(
        "document",
        help=(
            "Output mode - document (save as JSON zip), md (save as markdown file), "
            "md_page (save as folder of markdown pages)"
        ),
    ),
    conversion_mode: Literal[
        "ocr",
        "ocr_layout",
        "table",
        "image_description",
        "formula",
        "chart",
    ] = typer.Option(
        "ocr",
        help=(
            "Conversion mode - ocr (plain), ocr_layout (OCR with layout), table (table-centric), "
            "image_description (describe the image), formula (formula extraction), chart (chart recognition)"
        ),
    ),
    with_vllm_server: bool = typer.Option(
        False,
        help=(
            "If True, a local VLLM server will be deployed if the model is not found in the registry. "
            "Note that if the model is in the registry and uri is None, the server will be deployed."
        ),
    ),
    concurrency: int = typer.Option(10, help="Number of parallel requests"),
    dpi: int | None = typer.Option(None, help="DPI to use for the conversion"),
    debug: bool = typer.Option(False, help="Run in debug mode"),
    _return_documents: bool = typer.Option(False, hidden=True),
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
        conversion_mode: Conversion mode - "ocr" (plain), "ocr_layout" (OCR with layout), "table" (table-centric), "image_description" (describe the image), "formula" (formula extraction), "chart" (chart recognition)
        with_vllm_server: If True, a local VLLM server will be deployed if the model is not found in the registry. Note that if the model is in the registry and the uri is None, the server will be anyway deployed.
        dpi: DPI to use for the conversion. If not specified, the default DPI will be used.
        debug: If True, run in debug mode (single-threaded, no concurrency)
    """
    from vlmparse.converter_with_server import ConverterWithServer

    with ConverterWithServer(
        model=model,
        uri=uri,
        gpus=gpus,
        with_vllm_server=with_vllm_server,
        concurrency=concurrency,
        return_documents=_return_documents,
    ) as converter_with_server:
        return converter_with_server.parse(
            inputs=inputs,
            out_folder=out_folder,
            mode=mode,
            conversion_mode=conversion_mode,
            dpi=dpi,
            debug=debug,
        )


@app.command("list")
def containers():
    """List all containers whose name begins with vlmparse."""
    import docker

    from vlmparse.servers.utils import _get_container_labels

    try:
        client = docker.from_env()
        all_containers = client.containers.list(all=True)

        if not all_containers:
            logger.info("No containers found")
            return

        # Group containers by compose project or as standalone
        projects = {}  # project_name -> list of containers

        for container in all_containers:
            labels = _get_container_labels(container)
            project = labels.get("com.docker.compose.project") or labels.get(
                "vlmparse_compose_project"
            )

            # Include if name starts with vlmparse OR if it's in a vlmparse compose project
            if container.name.startswith("vlmparse"):
                if project:
                    projects.setdefault(project, []).append(container)
                else:
                    projects[container.name] = [container]
            elif project and project.startswith("vlmparse"):
                projects.setdefault(project, []).append(container)

        if not projects:
            logger.info("No vlmparse containers found")
            return

        # Prepare table data - one row per project/standalone container
        table_data = []
        for project_name, project_containers in projects.items():
            # Find the main container with vlmparse labels
            main_container = project_containers[0]
            for c in project_containers:
                labels = _get_container_labels(c)
                if labels.get("vlmparse_uri"):
                    main_container = c
                    break

            labels = _get_container_labels(main_container)

            # Extract port mappings from all containers in the project
            ports = []
            for container in project_containers:
                if container.ports:
                    for _, host_bindings in container.ports.items():
                        if host_bindings:
                            for binding in host_bindings:
                                ports.append(f"{binding['HostPort']}")

            port_str = ", ".join(sorted(set(ports))) if ports else "N/A"
            uri = labels.get("vlmparse_uri", "N/A")
            gpu = labels.get("vlmparse_gpus", "N/A")

            # Get all statuses
            statuses = list(set(c.status for c in project_containers))
            status_str = (
                statuses[0] if len(statuses) == 1 else f"mixed ({', '.join(statuses)})"
            )

            # Name: show project name if compose, otherwise container name
            is_compose = labels.get("com.docker.compose.project") or labels.get(
                "vlmparse_compose_project"
            )
            name = project_name if is_compose else main_container.name

            table_data.append([name, status_str, port_str, gpu, uri])

        # Display as table
        from tabulate import tabulate

        headers = ["Name", "Status", "Port(s)", "GPU", "URI"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        total = sum(len(containers) for containers in projects.values())
        logger.info(
            f"\nFound {len(projects)} vlmparse deployment(s) ({total} container(s)):\n"
        )
        print(table)

    except docker.errors.DockerException as e:
        logger.error(f"Failed to connect to Docker: {e}")
        logger.error(
            "Make sure Docker is running and you have the necessary permissions"
        )


@app.command("stop")
def stop(
    container: str | None = typer.Argument(None, help="Container ID or name to stop"),
):
    """Stop a Docker container by its ID or name.

    If the selected container belongs to a Docker Compose project, the whole
    compose stack is stopped and removed.

    Args:
        container: Container ID or name to stop. If not specified, automatically stops the container if only one vlmparse container is running.
    """
    import docker

    from vlmparse.servers.utils import _stop_compose_stack_for_container

    try:
        client = docker.from_env()

        # If no container specified, try to auto-select
        if container is None:
            from vlmparse.servers.utils import _get_container_labels

            all_containers = client.containers.list()

            # Group containers by compose project or as standalone
            projects = {}  # project_name -> list of containers
            for c in all_containers:
                labels = _get_container_labels(c)
                project = labels.get("com.docker.compose.project") or labels.get(
                    "vlmparse_compose_project"
                )

                # Include if name starts with vlmparse OR if it's in a vlmparse compose project
                if c.name.startswith("vlmparse"):
                    if project:
                        projects.setdefault(project, []).append(c)
                    else:
                        projects[c.name] = [c]
                elif project and project.startswith("vlmparse"):
                    projects.setdefault(project, []).append(c)

            if len(projects) == 0:
                logger.error("No vlmparse containers found")
                return
            elif len(projects) > 1:
                logger.error(
                    f"Multiple vlmparse deployments found ({len(projects)}). "
                    "Please specify a container ID or name:"
                )
                for project_name, containers in projects.items():
                    if len(containers) > 1:
                        logger.info(
                            f"  - {project_name} ({len(containers)} containers)"
                        )
                    else:
                        logger.info(
                            f"  - {containers[0].name} ({containers[0].short_id})"
                        )
                return
            else:
                # Only one project/deployment, pick any container from it
                target_container = list(projects.values())[0][0]
        else:
            # Try to get the specified container
            try:
                target_container = client.containers.get(container)
            except docker.errors.NotFound:
                logger.error(f"Container not found: {container}")
                return

        # If the container is part of a docker-compose stack, bring the whole stack down.
        if _stop_compose_stack_for_container(target_container):
            return

        # Stop + remove the container
        logger.info(
            f"Stopping container: {target_container.name} ({target_container.short_id})"
        )
        try:
            target_container.stop()
        except Exception:
            pass
        try:
            target_container.remove(force=True)
        except Exception:
            pass
        logger.info("✓ Container stopped and removed successfully")

    except docker.errors.DockerException as e:
        logger.error(f"Failed to connect to Docker: {e}")
        logger.error(
            "Make sure Docker is running and you have the necessary permissions"
        )


@app.command("log")
def log(
    container: str | None = typer.Argument(
        None, help="Container ID or name. If not specified, auto-selects."
    ),
    follow: bool = typer.Option(True, help="Follow log output"),
    tail: int = typer.Option(500, help="Number of lines to show from the end"),
):
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
            from vlmparse.servers.utils import _get_container_labels

            all_containers = client.containers.list()
            vlmparse_containers = []

            for c in all_containers:
                labels = _get_container_labels(c)
                project = labels.get("com.docker.compose.project") or labels.get(
                    "vlmparse_compose_project"
                )

                # Include if name starts with vlmparse OR if it's in a vlmparse compose project
                if c.name.startswith("vlmparse") or (
                    project and project.startswith("vlmparse")
                ):
                    vlmparse_containers.append(c)

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


@app.command("list-register")
def list_register():
    """List all model keys registered in client and server registries."""
    from vlmparse.registries import converter_config_registry, docker_config_registry

    client_models = sorted(converter_config_registry.list_models())
    server_models = sorted(docker_config_registry.list_models())

    print("\nClient Models Registry:")
    for model in client_models:
        print(f"  - {model}")

    print("\nServer Models Registry:")
    for model in server_models:
        print(f"  - {model}")


@app.command("view")
def view(folder: str = typer.Argument(..., help="Folder to visualize with Streamlit")):
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
    app()


if __name__ == "__main__":
    main()
