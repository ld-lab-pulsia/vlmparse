from typing import Literal

import httpx
from loguru import logger
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import Page
from vlmparse.servers.docker_server import DockerServerConfig


class DoclingDockerServerConfig(DockerServerConfig):
    """Configuration for Docling Serve using official image."""

    model_name: str = "docling"
    docker_image: str = Field(default="")
    cpu_only: bool = False
    command_args: list[str] = Field(default_factory=list)
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: ["Application startup complete", "Uvicorn running"]
    )
    enable_ui: bool = False
    docker_port: int = 5001
    container_port: int = 5001
    environment: dict[str, str] = Field(
        default_factory=lambda: {
            "DOCLING_SERVE_HOST": "0.0.0.0",
            "DOCLING_SERVE_PORT": "5001",
        }
    )

    def model_post_init(self, __context):
        """Set docker_image and gpu_device_ids based on cpu_only if not explicitly provided."""
        if not self.docker_image:
            if self.cpu_only:
                self.docker_image = "quay.io/docling-project/docling-serve-cpu:latest"
            else:
                self.docker_image = "quay.io/docling-project/docling-serve:latest"

        # For CPU-only mode, explicitly disable GPU by setting empty list
        if self.cpu_only and self.gpu_device_ids is None:
            self.gpu_device_ids = []

        if self.enable_ui:
            self.command_args.append("--enable-ui")

    @property
    def client_config(self):
        return DoclingConverterConfig(base_url=f"http://localhost:{self.docker_port}")


class DoclingConverterConfig(ConverterConfig):
    """Configuration for Docling converter client."""

    base_url: str = "http://localhost:5001"
    timeout: int = 300
    api_kwargs: dict = {"output_format": "markdown", "image_export_mode": "referenced"}

    def get_client(self, **kwargs) -> "DoclingConverter":
        return DoclingConverter(config=self, **kwargs)


class DoclingConverter(BaseConverter):
    """Client for Docling Serve API using httpx."""

    def __init__(
        self,
        config: DoclingConverterConfig,
        num_concurrent_files: int = 10,
        num_concurrent_pages: int = 10,
        save_folder: str | None = None,
        save_mode: Literal["document", "md", "md_page"] = "document",
        debug: bool = False,
        return_documents_in_batch_mode: bool = False,
    ):
        super().__init__(
            config=config,
            num_concurrent_files=num_concurrent_files,
            num_concurrent_pages=num_concurrent_pages,
            save_folder=save_folder,
            save_mode=save_mode,
            debug=debug,
            return_documents_in_batch_mode=return_documents_in_batch_mode,
        )
        self.client = httpx.AsyncClient(timeout=self.config.timeout)

    async def async_call_inside_page(self, page: Page) -> Page:
        """Process a single page using Docling Serve API."""
        from io import BytesIO

        image = page.image

        # Convert image to bytes for file upload
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        files = {"files": ("image.png", img_bytes, "image/png")}
        data = self.config.api_kwargs

        url = f"{self.config.base_url}/v1/convert/file"
        logger.debug(f"Calling Docling API at: {url}")

        try:
            response = await self.client.post(
                url, files=files, data=data, headers={"Accept": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Docling API response status: {response.status_code}")

            # Extract text from the response
            # The response structure depends on the output format
            if self.config.api_kwargs["output_format"] == "markdown":
                text = result["document"]["md_content"]

            elif self.config.api_kwargs["output_format"] == "text":
                text = result["document"]["md_content"]

            else:  # json or other formats
                text = str(result)

            logger.info(f"Extracted text length: {len(text)}")

            # Clean and convert the response
            text = clean_response(text)
            text = html_to_md_keep_tables(text)
            page.text = text

        except Exception as e:
            logger.error(f"Error processing page with Docling: {e}")
            page.text = f"Error: {str(e)}"

        return page

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    def __del__(self):
        """Cleanup when the converter is destroyed."""
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.client.aclose())
            else:
                asyncio.run(self.client.aclose())
        except Exception:
            pass
