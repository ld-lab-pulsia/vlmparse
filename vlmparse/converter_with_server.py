import datetime
import os
import time
from pathlib import Path
from typing import Literal, cast

from loguru import logger

from vlmparse.constants import DEFAULT_SERVER_PORT
from vlmparse.model_endpoint_config import ImageDescriptionConfig
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
    model: str | None,
    uri: str | None,
    provider: Literal["registry", "hf", "google", "openai", "azure", "anthropic"] = "registry",
    api_key: str | None = None,
    use_response_api: bool = False,
):
    from vlmparse.registries import (
        _make_azure_factory,
        _make_google_factory,
        _make_hf_factory,
        _make_openai_factory,
        converter_config_registry,
    )

    if uri is not None and model is None and provider in ["registry", "hf"]:
        model = get_model_from_uri(uri)

    assert (
        model is not None
    ), "Model name could not be determined from parameters. Please provide a model name or a URI that includes the model name."

    if provider == "hf":
        client_config = _make_hf_factory(model, uri)

    elif provider == "registry":
        available = converter_config_registry.list_providers(model)
        if "registry" in available:
            client_config = converter_config_registry.get(
                model, uri=uri, provider="registry"
            )
        else:
            client_config = converter_config_registry.get(model, uri=uri)

    elif provider == "google":
        if "google" in converter_config_registry.list_providers(model):
            client_config = converter_config_registry.get(
                model, uri=uri, provider="google"
            )
        else:
            client_config = _make_google_factory(model, uri, api_key=api_key)

    elif provider == "openai":
        if "openai" in converter_config_registry.list_providers(model):
            client_config = converter_config_registry.get(
                model, uri=uri, provider="openai"
            )
        else:
            client_config = _make_openai_factory(model, uri, api_key=api_key)

    elif provider == "azure":
        client_config = _make_azure_factory(
            model, uri, api_key=api_key, use_response_api=use_response_api
        )

    elif provider == "anthropic":
        from vlmparse.clients.anthropic_converter import AnthropicConverterConfig

        if api_key is None:
            assert (
                os.getenv("ANTHROPIC_API_KEY") is not None
            ), "Anthropic API key must be provided via parameter or ANTHROPIC_API_KEY env variable"
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        client_config = AnthropicConverterConfig(
            model_name=model,
            api_key=api_key,
            default_model_name=model,
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
        provider: Literal["registry", "hf", "google", "openai", "azure", "anthropic"] = "registry",
        concurrency: int = 10,
        vllm_args: list[str] | None = None,
        forget_predefined_vllm_args: bool = False,
        return_documents: bool = False,
        use_response_api: bool = False,
        # ---- image description post-processor ----
        image_description: "ImageDescriptionConfig | None" = None,
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
        self.image_description = image_description

    def start_server_and_client(self):
        from vlmparse.registries import (
            converter_config_registry,
            docker_config_registry,
        )

        start_local_server = False
        if self.uri is None:
            assert (
                self.model is not None
            ), "Model must be specified if uri is not provided"
            if self.provider == "hf":
                start_local_server = True
            elif self.provider == "registry":
                if self.model in docker_config_registry.list_models():
                    start_local_server = True

        if start_local_server:
            assert (
                self.model is not None
            ), "Model must be specified to start local server"
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

    def _make_image_description_postproc(self):
        """Build the right PageProcessorConfig for item-level image description.

        Connection parameters come from ``self.image_description.connection``
        if set, otherwise fall back to the main converter's connection.
        """
        from vlmparse.clients.item_description_processors import (
            DeepSeekOCR2ItemDescriptionConfig,
            DeepSeekOCRItemDescriptionConfig,
            VLMItemDescriptionConfig,
        )

        assert self.image_description is not None
        desc = self.image_description
        client = cast(ConverterWithServer, self.client)
        # Start from the main converter's connection, then apply overrides
        conn = client.config.endpoint.model_copy()  # ty: ignore
        if desc.connection is not None:
            if desc.connection.base_url is not None:
                conn.base_url = desc.connection.base_url
            if desc.connection.api_key:
                conn.api_key = desc.connection.api_key
            if desc.connection.timeout is not None:
                conn.timeout = desc.connection.timeout
            if desc.connection.max_retries != 1:
                conn.max_retries = desc.connection.max_retries

        categories = desc.categories
        extra: dict = {}
        if desc.prompt is not None:
            extra["prompt"] = desc.prompt

        # Route to the right config class based on the HF model name;
        # if an explicit model_name is given in the override connection, use it.
        hf_model = (
            desc.connection.model_name
            if desc.connection is not None
            and desc.connection.model_name != conn.model_name
            else client.config.model_name  # ty: ignore
        )

        if hf_model == "deepseek-ai/DeepSeek-OCR":
            return DeepSeekOCRItemDescriptionConfig(
                connection=conn, categories=categories, **extra
            )
        elif hf_model == "deepseek-ai/DeepSeek-OCR-2":
            return DeepSeekOCR2ItemDescriptionConfig(
                connection=conn, categories=categories, **extra
            )
        else:
            # For a generic VLM the model_name in conn IS the served model id
            if desc.connection is not None:
                conn.model_name = desc.connection.model_name
            return VLMItemDescriptionConfig(
                connection=conn,
                completion_kwargs={"temperature": 0.2, "max_tokens": 512},
                categories=categories,
                **extra,
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
        max_image_size: int | None = None,
        debug: bool = False,
        retrylast: bool = False,
        completion_kwargs: dict | None = None,
        pages: list[int] | None = None,
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
        if max_image_size is not None:
            self.client.config.max_image_size = int(max_image_size)

        if conversion_mode is not None:
            self.client.config.conversion_mode = conversion_mode

        if completion_kwargs is not None and isinstance(
            self.client.config, OpenAIConverterConfig
        ):
            self.client.config.completion_kwargs |= completion_kwargs

        # ---- image description post-processor (idempotent) ----
        from vlmparse.clients.item_description_processors import (
            DeepSeekOCR2ItemDescriptionConfig,
            DeepSeekOCRItemDescriptionConfig,
            VLMItemDescriptionConfig,
        )

        _desc_types = (
            DeepSeekOCRItemDescriptionConfig,
            DeepSeekOCR2ItemDescriptionConfig,
            VLMItemDescriptionConfig,
        )
        self.client.config.page_postproc = [
            p
            for p in self.client.config.page_postproc
            if not isinstance(p, _desc_types)
        ]
        if self.image_description is not None:
            self.client.config.page_postproc.append(
                self._make_image_description_postproc()
            )

        if debug:
            self.client.debug = debug

        if pages is not None:
            self.client.pages = pages

        if out_folder is not None:
            self.client.save_folder = out_folder
            self.client.save_mode = mode
        self.client.num_concurrent_files = self.concurrency if not debug else 1
        self.client.num_concurrent_pages = self.concurrency if not debug else 1

        logger.debug(f"Processing {len(file_paths)} files with {self.model} converter")
        tic = time.perf_counter()

        documents = self.client.batch(file_paths)

        toc = time.perf_counter()
        logger.debug(
            f"Processed {len(file_paths)} documents to {out_folder} in {toc - tic:.2f} seconds"
        )

        return documents

    def get_out_folder(self) -> str | None:
        if self.client is not None and hasattr(self.client, "save_folder"):
            return self.client.save_folder
