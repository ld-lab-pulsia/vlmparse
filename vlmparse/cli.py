# ruff: noqa: B008
from typing import Literal, cast

import typer
from loguru import logger

app = typer.Typer(help="Parse PDF documents with VLMs.", pretty_exceptions_enable=False)


@app.command("serve")
def serve(
    model: str = typer.Argument(..., help="Model name"),
    port: int | None = typer.Option(
        None, "-p", "--port", help="VLLM server port (default: 8056)"
    ),
    gpu: str | None = typer.Option(
        None,
        "-g",
        "--gpu",
        help='Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, all GPUs will be used.',
    ),
    provider: Literal["registry", "hf"] = typer.Option(
        "registry",
        "--provider",
        help="provider type for the model. 'registry' (default) or 'hf'.",
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
        server: Server type for the model. 'registry' (default) or 'hf'.
        vllm_args: Additional keyword arguments to pass to the VLLM server.
        forget_predefined_vllm_args: If True, the predefined VLLM kwargs from the docker config will be replaced by vllm_args otherwise the predefined kwargs will be updated with vllm_args with a risk of collision of argument names.
    """

    from vlmparse.converter_with_server import start_server

    base_url, container, _, _ = start_server(
        model=model,
        gpus=gpu,
        port=port,
        provider=provider,
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
    out_folder: str = typer.Option(
        ".", "-o", "--out-folder", help="Output folder for parsed documents"
    ),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model name. If not specified, inferred from the URI.",
    ),
    uri: str | None = typer.Option(
        None,
        "-u",
        "--uri",
        help=(
            "URI of the server. If not specified and the pipe is vllm, "
            "a local server will be deployed."
        ),
    ),
    gpu: str | None = typer.Option(
        None,
        "-g",
        "--gpu",
        help='Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, GPU 0 will be used.',
    ),
    save_mode: Literal["document", "md", "md_page"] = typer.Option(
        "document",
        "-s",
        "--save-mode",
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
        "-c",
        "--conversion-mode",
        help=(
            "Conversion mode - ocr (plain), ocr_layout (OCR with layout), table (table-centric), "
            "image_description (describe the image), formula (formula extraction), chart (chart recognition)"
        ),
    ),
    provider: Literal["registry", "hf", "google", "openai", "azure"] = typer.Option(
        "registry",
        "-p",
        "--provider",
        help="Server type for the model. Defaults to 'registry'.",
    ),
    with_vllm_server: bool = typer.Option(
        False,
        help=(
            "Deprecated. Use --server hf instead. "
            "If True, a local VLLM server will be deployed if the model is not found in the registry. "
            "Note that if the model is in the registry and uri is None, the server will be deployed."
        ),
    ),
    concurrency: int = typer.Option(
        10, "-n", "--concurrency", help="Number of parallel requests"
    ),
    dpi: int | None = typer.Option(
        None, "-d", "--dpi", help="DPI to use for the conversion"
    ),
    max_image_size: int | None = typer.Option(
        None,
        "--max-image-size",
        help=(
            "Maximum size (in pixels) for the longest edge of images during conversion. "
        ),
    ),
    use_response_api: bool = typer.Option(
        False,
        help=(
            "If True, uses the response API for conversion which returns a more detailed usage breakdown. This is mandatory for gpt-5 model on azure."
            "This is only applicable for compatible servers and may require additional configuration on the server side."
        ),
    ),
    image_description: bool = typer.Option(
        False,
        "--image-description/--no-image-description",
        help=(
            "Add an image-description post-processor that describes detected figures/charts "
            "using a VLM.  By default the same model and server as the main conversion are used."
        ),
    ),
    image_description_model: str | None = typer.Option(
        None,
        "--image-description-model",
        help=(
            "Model to use for image description.  Defaults to the main conversion model. "
            "Use 'deepseek-ai/DeepSeek-OCR', 'deepseek-ai/DeepSeek-OCR-2', or any "
            "OpenAI-compatible model name."
        ),
    ),
    image_description_uri: str | None = typer.Option(
        None,
        "--image-description-uri",
        help="URI of the server for image description.  Defaults to the main server URI.",
    ),
    image_description_api_key: str | None = typer.Option(
        None,
        "--image-description-api-key",
        help="API key for the image description server (if different from the main server).",
    ),
    image_description_categories: str | None = typer.Option(
        None,
        "--image-description-categories",
        help=(
            "Comma-separated list of item categories to describe "
            "(default: 'picture,image,figure,chart')."
        ),
    ),
    image_description_prompt: str | None = typer.Option(
        None,
        "--image-description-prompt",
        help="Custom prompt sent to the VLM for each image crop.  Uses model default if not set.",
    ),
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
        provider: provider type for the model. Defaults to 'registry'.
        dpi: DPI to use for the conversion. If not specified, the default DPI will be used.
        debug: If True, run in debug mode (single-threaded, no concurrency)
    """
    from vlmparse.converter_with_server import ConverterWithServer
    from vlmparse.model_endpoint_config import (
        ImageDescriptionConfig,
        ModelEndpointConfig,
    )

    if with_vllm_server and provider == "registry":
        provider = "hf"

    # Build ImageDescriptionConfig from flat CLI flags; None = disabled
    img_desc: ImageDescriptionConfig | None = None
    if image_description:
        categories = (
            [c.strip() for c in image_description_categories.split(",")]
            if image_description_categories
            else ["picture", "image", "figure", "chart"]
        )
        conn_override: ModelEndpointConfig | None = None
        if (
            image_description_model
            or image_description_uri
            or image_description_api_key
        ):
            model_name = cast(str, image_description_model or model)
            conn_override = ModelEndpointConfig(
                model_name=model_name,
                base_url=image_description_uri,
                api_key=image_description_api_key or "",
            )
        img_desc = ImageDescriptionConfig(
            connection=conn_override,
            categories=categories,
            prompt=image_description_prompt,
        )

    with ConverterWithServer(
        model=model,
        uri=uri,
        gpus=gpu,
        provider=provider,
        concurrency=concurrency,
        return_documents=_return_documents,
        use_response_api=use_response_api,
        image_description=img_desc,
    ) as converter_with_server:
        return converter_with_server.parse(
            inputs=inputs,
            out_folder=out_folder,
            mode=save_mode,
            conversion_mode=conversion_mode,
            dpi=dpi,
            debug=debug,
        )


@app.command("list")
def containers():
    """List all vlmparse deployments (networks, compose stacks, and containers)."""
    import docker

    from vlmparse.servers.utils import _get_container_labels, _get_vlmparse_groups

    try:
        groups, _network_keys = _get_vlmparse_groups(running_only=False)

        if not groups:
            logger.info("No vlmparse containers or networks found")
            return

        # One table row per group
        table_data = []
        for group_key, group_containers in groups.items():
            # Find the primary container (the one with vlmparse_uri label)
            main_container = group_containers[0]
            for c in group_containers:
                lbl = _get_container_labels(c)
                if lbl.get("vlmparse_uri"):
                    main_container = c
                    break

            labels = _get_container_labels(main_container)

            ports = []
            for c in group_containers:
                if c.ports:
                    for _, host_bindings in c.ports.items():
                        if host_bindings:
                            for binding in host_bindings:
                                ports.append(binding["HostPort"])
            port_str = ", ".join(sorted(set(ports))) if ports else "N/A"

            uri = labels.get("vlmparse_uri", "N/A")
            gpu = labels.get("vlmparse_gpus", "N/A")

            statuses = list(set(c.status for c in group_containers))
            status_str = (
                statuses[0] if len(statuses) == 1 else f"mixed ({', '.join(statuses)})"
            )

            table_data.append([group_key, status_str, port_str, gpu, uri])

        from tabulate import tabulate

        headers = ["Name", "Status", "Port(s)", "GPU", "URI"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        total = sum(len(c) for c in groups.values())
        logger.info(
            f"\nFound {len(groups)} vlmparse deployment(s) ({total} container(s)):\n"
        )
        print(table)

    except docker.errors.DockerException as e:  # type: ignore[name-defined]
        logger.error(f"Failed to connect to Docker: {e}")
        logger.error(
            "Make sure Docker is running and you have the necessary permissions"
        )


@app.command("stop")
def stop(
    container: str | None = typer.Argument(
        None, help="Container ID/name or network name to stop"
    ),
):
    """Stop a vlmparse deployment.

    Accepts a container ID/name, a Docker network name, or auto-selects when
    only one vlmparse deployment is running.  For ContainerGroupServer
    deployments the whole network (containers + network) is torn down.  For
    Docker Compose stacks the full stack is brought down.

    Args:
        container: Container ID/name or network name to stop. If not specified,
            automatically stops the deployment when exactly one is running.
    """
    import docker

    from vlmparse.servers.utils import (
        _get_container_vlmparse_network,
        _get_vlmparse_groups,
        _stop_compose_stack_for_container,
        _stop_network_group,
    )

    try:
        client = docker.from_env()
        target_container = None

        if container is None:
            # Auto-select among running deployments
            groups, network_keys = _get_vlmparse_groups(running_only=True)

            if len(groups) == 0:
                logger.error("No vlmparse containers or networks found")
                return
            elif len(groups) > 1:
                logger.error(
                    f"Multiple vlmparse deployments found ({len(groups)}). "
                    "Please specify a container ID, name, or network name:"
                )
                for key, conts in groups.items():
                    logger.info(f"  - {key} ({len(conts)} container(s))")
                return
            else:
                group_key = next(iter(groups))
                if group_key in network_keys:
                    _stop_network_group(group_key)
                    return
                target_container = groups[group_key][0]
        else:
            # Check if the argument is a Docker network starting with vlmparse
            try:
                network = client.networks.get(container)
                if network.name.startswith("vlmparse"):
                    _stop_network_group(network.name)
                    return
            except docker.errors.NotFound:  # type: ignore[name-defined]
                pass

            # Try to resolve as a container
            try:
                target_container = client.containers.get(container)
            except docker.errors.NotFound:  # type: ignore[name-defined]
                logger.error(f"Container or network not found: {container}")
                return

        # --- target_container is resolved; apply stop strategies in order ---

        # 1. Docker Compose stack
        if _stop_compose_stack_for_container(target_container):
            return

        # 2. ContainerGroupServer network group
        net_name = _get_container_vlmparse_network(target_container)
        if net_name:
            _stop_network_group(net_name)
            return

        # 3. Standalone container
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

    except docker.errors.DockerException as e:  # type: ignore[name-defined]
        logger.error(f"Failed to connect to Docker: {e}")
        logger.error(
            "Make sure Docker is running and you have the necessary permissions"
        )


@app.command("log")
def log(
    container: str | None = typer.Argument(
        None, help="Container ID or name. If not specified, auto-selects."
    ),
    follow: bool = typer.Option(True, "-f", "--follow", help="Follow log output"),
    tail: int = typer.Option(
        500, "-t", "--tail", help="Number of lines to show from the end"
    ),
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
            except docker.errors.NotFound:  # type: ignore[name-defined]
                logger.error(f"Container not found: {container}")
                return

        # Get and display logs
        if follow:
            logger.info("Following logs (press Ctrl+C to stop)...")
            try:
                for log_line in target_container.logs(
                    stream=True, follow=True, tail=tail
                ):
                    if "POST" not in log_line.decode("utf-8", errors="replace"):
                        print(log_line.decode("utf-8", errors="replace"), end="")
            except KeyboardInterrupt:
                logger.info("\nStopped following logs")
        else:
            logs = target_container.logs().decode("utf-8", errors="replace")
            print(logs)

    except docker.errors.DockerException as e:  # type: ignore[name-defined]
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
