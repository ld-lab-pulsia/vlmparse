import datetime
import os
from pathlib import Path
from typing import Literal

from loguru import logger

from vlmparse.servers.utils import get_model_from_uri
from vlmparse.utils import get_file_paths


class ConverterWithServer:
    def __init__(
        self,
        model: str,
        uri: str | None = None,
        gpus: str | None = None,
        port: int | None = None,
        with_vllm_server: bool = False,
        concurrency: int = 10,
    ):
        from vlmparse.registries import (
            converter_config_registry,
            docker_config_registry,
        )

        self.model = model
        self.uri = uri
        self.port = port
        self.gpus = gpus
        self.with_vllm_server = with_vllm_server
        self.concurrency = concurrency

        if self.uri is not None and self.model is None:
            self.model = get_model_from_uri(self.uri)

        gpu_device_ids = None
        if self.gpus is not None:
            gpu_device_ids = [g.strip() for g in self.gpus.split(",")]

        if self.uri is None:
            docker_config = docker_config_registry.get(
                self.model, default=self.with_vllm_server
            )
            if self.port is not None:
                docker_config.docker_port = self.port

            if docker_config is not None:
                docker_config.gpu_device_ids = gpu_device_ids
                server = docker_config.get_server(auto_stop=True)
                server.start()

                self.client = docker_config.get_client()
            else:
                self.client = converter_config_registry.get(self.model).get_client()

        else:
            client_config = converter_config_registry.get(self.model, uri=self.uri)
            self.client = client_config.get_client()

    def parse(
        self,
        inputs: str | list[str],
        out_folder: str = ".",
        mode: Literal["document", "md", "md_page"] = "document",
        dpi: int | None = None,
        debug: bool = False,
        retrylast: bool = False,
    ):
        file_paths = get_file_paths(inputs)
        assert (
            out_folder is not None
        ), "out_folder must be provided if retrylast is True"
        if retrylast:
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
            out_folder = Path(out_folder) / (
                datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            )

        if dpi is not None:
            self.client.config.dpi = int(dpi)

        if debug:
            self.client.debug = debug

        self.client.save_folder = out_folder
        self.client.save_mode = mode
        self.client.num_concurrent_files = self.concurrency if not debug else 1
        self.client.num_concurrent_pages = self.concurrency if not debug else 1

        logger.info(f"Processing {len(file_paths)} files with {self.model} converter")

        documents = self.client.batch(file_paths)

        if documents is not None:
            logger.info(f"Processed {len(documents)} documents to {out_folder}")
        else:
            logger.info(f"Processed {len(file_paths)} documents to {out_folder}")

        return documents

    def get_out_folder(self) -> Path:
        return self.client.save_folder
