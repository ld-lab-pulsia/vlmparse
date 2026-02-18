import datetime
import os
from pathlib import Path
from typing import Literal

from loguru import logger

from vlmparse.constants import DEFAULT_SERVER_PORT
from vlmparse.servers.utils import get_model_from_uri
from vlmparse.utils import get_file_paths


def start_server(
    model: str,
    gpus: str | None = None,
    port: None | int = None,
    provider: Literal["registry", "hf"] = "registry",
    vllm_args: list[str] | None = None,
    forget_predefined_vllm_args: bool = False,
    auto_stop: bool = False,
):
    from vlmparse.registries import docker_config_registry
    from vlmparse.servers.docker_server import (
        DEFAULT_MODEL_NAME,
        VLLMDockerServerConfig,
    )

    base_url = ""
    container = None
    docker_config = docker_config_registry.get(model)

    if vllm_args is None:
        vllm_args = []

    if port is None:
        port = int(DEFAULT_SERVER_PORT)

    if gpus is None:
        gpus = "0"

    if docker_config is None:
        if provider == "registry":
            print(f"DEBUG: Registry lookup failed for {model} (strict mode)")
            raise ValueError(
                f"Model '{model}' not found in registry and provider='registry'. Use provider='hf' to serve arbitrary HuggingFace models."
            )
        elif provider == "hf":
            docker_config = VLLMDockerServerConfig(
                model_name=model, default_model_name=DEFAULT_MODEL_NAME
            )
        else:
            logger.warning(
                f"No Docker configuration found for model: {model} and server type is undetermined."
            )
            return "", container, None, docker_config

    gpu_device_ids = None
    if gpus is not None:
        gpu_device_ids = [g.strip() for g in str(gpus).split(",")]

    if docker_config is not None:
        if port is not None:
            docker_config.docker_port = port
        docker_config.gpu_device_ids = gpu_device_ids
        if hasattr(docker_config, "update_command_args"):
            docker_config.update_command_args(
                vllm_args,
                forget_predefined_vllm_args=forget_predefined_vllm_args,
            )

        logger.info(
            f"Deploying server for {docker_config.model_name} on port {port}..."
        )
        provider = docker_config.get_server(auto_stop=auto_stop)
        if provider is None:
            logger.error(f"Model server not found for model: {model}")
            return "", container, None, docker_config

        base_url, container = provider.start()

    return base_url, container, provider, docker_config


def get_client_config(
    model: str,
    uri: str | None,
    provider: Literal["registry", "hf", "google", "openai", "azure"] = "registry",
    api_key: str | None = None,
    use_response_api: bool = False,
):
    from vlmparse.clients.openai_converter import OpenAIConverterConfig
    from vlmparse.registries import converter_config_registry

    if uri is not None and model is None and provider in ["registry", "hf"]:
        model = get_model_from_uri(uri)

    if provider == "hf":
        client_config = OpenAIConverterConfig(model_name=model, base_url=uri)

    elif provider == "registry":
        client_config = converter_config_registry.get(model, uri=uri)

    elif provider == "google":
        from vlmparse.registries import GOOGLE_API_BASE_URL

        client_config = OpenAIConverterConfig(
            model_name=model,
            base_url=GOOGLE_API_BASE_URL if uri is None else uri,
            api_key=api_key if api_key is not None else os.getenv("GOOGLE_API_KEY"),
            default_model_name=model,
        )

    elif provider == "openai":
        client_config = OpenAIConverterConfig(
            model_name=model,
            base_url=uri,
            api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"),
            default_model_name=model,
        )
    elif provider == "azure":
        client_config = OpenAIConverterConfig(
            model_name=model,
            base_url=uri if uri is not None else os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key
            if api_key is not None
            else os.getenv("AZURE_OPENAI_API_KEY"),
            is_azure=True,
            default_model_name=model,
            use_response_api=use_response_api,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return client_config


class ConverterWithServer:
    def __init__(
        self,
        model: str | None = None,
        uri: str | None = None,
        gpus: str | None = None,
        port: int | None = None,
        provider: Literal["registry", "hf", "google", "openai", "azure"] = "registry",
        concurrency: int = 10,
        vllm_args: list[str] | None = None,
        forget_predefined_vllm_args: bool = False,
        return_documents: bool = False,
        use_response_api: bool = False,
    ):
        if model is None and uri is None:
            raise ValueError("Either 'model' or 'uri' must be provided")

        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        self.model = model
        self.uri = uri
        self.port = port
        self.gpus = gpus
        self.provider = provider
        self.concurrency = concurrency
        self.vllm_args = vllm_args
        self.forget_predefined_vllm_args = forget_predefined_vllm_args
        self.return_documents = return_documents
        self.server = None
        self.client = None
        self.use_response_api = use_response_api

    def start_server_and_client(self):
        from vlmparse.registries import (
            converter_config_registry,
            docker_config_registry,
        )

        assert (
            self.model is not None
        ), "Model name must be determined from either 'model' or 'uri'"

        start_local_server = False
        if self.uri is None:
            if self.provider == "hf":
                start_local_server = True
            elif self.provider == "registry":
                if self.model in docker_config_registry.list_models():
                    start_local_server = True

        if start_local_server:
            server_arg = "hf" if self.provider == "hf" else "registry"
            _, _, self.server, docker_config = start_server(
                model=self.model,
                gpus=self.gpus,
                port=self.port,
                provider=server_arg,
                vllm_args=self.vllm_args,
                forget_predefined_vllm_args=self.forget_predefined_vllm_args,
                auto_stop=True,
            )

            if docker_config is not None:
                self.client = docker_config.get_client(
                    return_documents_in_batch_mode=self.return_documents
                )
            else:
                # Should not happen if start_server works as expected
                self.client = converter_config_registry.get(self.model).get_client(
                    return_documents_in_batch_mode=self.return_documents
                )
        else:
            client_config = get_client_config(
                self.model,
                self.uri,
                self.provider,
                use_response_api=self.use_response_api,
            )

            self.client = client_config.get_client(
                return_documents_in_batch_mode=self.return_documents
            )

    def stop_server(self):
        if self.server is not None and self.server.auto_stop:
            self.server.stop()

    def __enter__(self):
        self.start_server_and_client()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stop_server()
        except Exception as e:
            logger.warning(f"Error stopping server during cleanup: {e}")
        return False  # Don't suppress exceptions

    def parse(
        self,
        inputs: str | list[str],
        out_folder: str | Path | None = ".",
        mode: Literal["document", "md", "md_page"] = "document",
        conversion_mode: Literal[
            "ocr",
            "ocr_layout",
            "table",
            "image_description",
            "formula",
            "chart",
        ]
        | None = None,
        dpi: int | None = None,
        debug: bool = False,
        retrylast: bool = False,
        completion_kwargs: dict | None = None,
    ):
        from vlmparse.clients.openai_converter import OpenAIConverterConfig

        assert (
            self.client is not None
        ), "Client not initialized. Call start_server_and_client() first."
        file_paths = get_file_paths(inputs)

        if retrylast:
            assert (
                out_folder is not None
            ), "out_folder must be provided if retrylast is True"
            retry = Path(out_folder)
            previous_runs = sorted(os.listdir(retry))
            if len(previous_runs) > 0:
                retry = retry / previous_runs[-1]
            else:
                raise ValueError(
                    "No previous runs found, do not use the retrylast flag"
                )
            already_processed = [
                f.removesuffix(".zip") for f in os.listdir(retry / "results")
            ]
            file_paths = [
                f
                for f in file_paths
                if Path(f).name.removesuffix(".pdf") not in already_processed
            ]

            logger.debug(f"Number of files after filtering: {len(file_paths)}")

        else:
            if out_folder is not None:
                out_folder = Path(out_folder) / (
                    datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
                )

        if dpi is not None:
            self.client.config.dpi = int(dpi)

        if conversion_mode is not None:
            self.client.config.conversion_mode = conversion_mode

        if completion_kwargs is not None and isinstance(
            self.client.config, OpenAIConverterConfig
        ):
            self.client.config.completion_kwargs |= completion_kwargs

        if debug:
            self.client.debug = debug

        if out_folder is not None:
            self.client.save_folder = out_folder
            self.client.save_mode = mode
        self.client.num_concurrent_files = self.concurrency if not debug else 1
        self.client.num_concurrent_pages = self.concurrency if not debug else 1

        logger.info(f"Processing {len(file_paths)} files with {self.model} converter")

        documents = self.client.batch(file_paths)  # type: ignore

        if documents is not None:
            logger.info(f"Processed {len(documents)} documents to {out_folder}")
        else:
            logger.info(f"Processed {len(file_paths)} documents to {out_folder}")

        return documents

    def get_out_folder(self) -> str | None:
        if self.client is not None and hasattr(self.client, "save_folder"):
            return self.client.save_folder
