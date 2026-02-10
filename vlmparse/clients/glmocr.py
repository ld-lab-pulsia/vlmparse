import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
import orjson
from loguru import logger
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import BoundingBox, Item, Page
from vlmparse.servers.docker_compose_server import DockerComposeServerConfig
from vlmparse.utils import to_base64

DOCKER_PIPELINE_DIR = (
    Path(__file__).parent.parent.parent / "docker_pipelines" / "glmocr"
)


class GLMOCRDockerServerConfig(DockerComposeServerConfig):
    """Docker Compose configuration for GLM-OCR server."""

    model_name: str = "GLM-OCR"
    aliases: list[str] = Field(default_factory=lambda: ["glmocr", "glm-ocr"])
    compose_file: str = str(DOCKER_PIPELINE_DIR / "compose.yaml")
    server_service: str = "glmocr-api"
    compose_services: list[str] = Field(
        default_factory=lambda: ["glmocr-api", "glmocr-vllm-server"]
    )
    gpu_service_names: list[str] = Field(default_factory=lambda: ["glmocr-vllm-server"])
    docker_port: int = 5002
    container_port: int = 5002
    environment: dict[str, str] = Field(
        default_factory=lambda: {
            "VLM_BACKEND": "vllm",
            "API_PORT": "8080",
        }
    )
    environment_services: list[str] = Field(default_factory=lambda: ["glmocr-api"])
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: ["Running on", "Application startup complete"]
    )

    def model_post_init(self, __context):
        if not self.compose_env:
            compose_env = {}
            for key in [
                "API_IMAGE_TAG_SUFFIX",
                "VLM_IMAGE_TAG_SUFFIX",
                "VLM_BACKEND",
            ]:
                value = os.getenv(key)
                if value:
                    compose_env[key] = value
            if compose_env:
                self.compose_env = compose_env

    @property
    def client_config(self):
        return GLMOCRConverterConfig(
            **self._create_client_kwargs(f"http://localhost:{self.docker_port}")
        )


class GLMOCRConverterConfig(ConverterConfig):
    """Configuration for GLM-OCR API client."""

    model_name: str = "GLM-OCR"
    aliases: list[str] = Field(default_factory=lambda: ["glmocr", "glm-ocr"])
    timeout: int = 600

    endpoint_parse: str = "/glmocr/parse"

    # GLM-OCR specific configuration

    # Output format: "json", "markdown", or "both"
    output_format: str = "both"

    # Enable layout detection (PP-DocLayout)
    enable_layout: bool = True

    # GLM-OCR model parameters
    max_tokens: int = 16384
    temperature: float = 0.01
    image_format: str = "JPEG"
    min_pixels: int = 12544
    max_pixels: int = 71372800

    # Backward-compat escape hatch: if set, applied last to the payload.
    request_overrides: dict[str, Any] = Field(default_factory=dict)

    def get_client(self, **kwargs) -> "GLMOCRConverter":
        return GLMOCRConverter(config=self, **kwargs)


class GLMOCRConverter(BaseConverter):
    """GLM-OCR HTTP API converter."""

    config: GLMOCRConverterConfig

    def _build_parse_payload(self, file_content_b64: str) -> dict:
        """Build the request payload for the GLM-OCR parse endpoint.

        Args:
            file_content_b64: Base64 encoded image content

        Returns:
            Dictionary payload for the API request
        """
        # Wrap base64 in data URI format as expected by GLM-OCR
        # Format: data:image/png;base64,<base64_data>
        data_uri = f"data:image/png;base64,{file_content_b64}"

        payload: dict[str, Any] = {
            "images": [data_uri]  # GLM-OCR expects a list
        }

        # Apply any request overrides
        if self.config.request_overrides:
            payload.update(self.config.request_overrides)

        return payload

    async def _post_json(self, endpoint: str, payload: dict) -> dict:
        """Make a POST request to the GLM-OCR API.

        Args:
            endpoint: API endpoint path
            payload: Request payload

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If the API returns an error
        """
        headers = {}
        assert self.config.base_url is not None, "Base URL is required for API calls"
        async with httpx.AsyncClient(
            base_url=self.config.base_url, timeout=self.config.timeout, headers=headers
        ) as client:
            response = await client.post(endpoint, json=payload)

        response.raise_for_status()
        data = response.json()

        # Check for error in response
        if "error" in data:
            raise RuntimeError(data.get("error", "Unknown error"))

        return data

    def _apply_markdown(self, page: Page, markdown_text: str | None):
        """Apply markdown text to the page.

        Args:
            page: Page object to update
            markdown_text: Markdown content from GLM-OCR
        """
        text = markdown_text or ""
        text = clean_response(text)
        text = html_to_md_keep_tables(text)
        logger.debug(f"Converted markdown text:\n{text}")
        page.text = text

    def _apply_items(self, page: Page, json_result: list[dict] | None):
        """Apply structured items to the page from JSON result.

        Args:
            page: Page object to update
            json_result: List of detected regions from GLM-OCR
        """
        if not json_result:
            return

        items: list[Item] = []

        for block in json_result:
            bbox = block.get("bbox_2d")
            if not bbox or len(bbox) != 4:
                # If no bbox, skip this item
                continue

            x1, y1, x2, y2 = bbox
            text = block.get("content") or ""
            label = block.get("label") or ""

            items.append(
                Item(
                    text=text,
                    box=BoundingBox(l=x1, t=y1, r=x2, b=y2),
                    category=label,
                )
            )

        page.items = items

    async def async_call_inside_page(self, page: Page) -> Page:
        """Process a single page through the GLM-OCR API.

        Args:
            page: Page object containing the image to process

        Returns:
            Updated Page object with OCR results
        """
        image = page.image
        assert image is not None, "Page image is required for processing"

        # Convert image to base64
        file_content_b64 = await asyncio.to_thread(to_base64, image, "PNG")

        # Build request payload
        payload = self._build_parse_payload(file_content_b64)

        # Call the GLM-OCR API
        data = await self._post_json(self.config.endpoint_parse, payload)

        # GLM-OCR returns results as a list (one per document)
        # Since we send one image, we get one document result
        result = data.get("markdown_result", None)

        if result:
            # Get markdown output if available
            markdown_result = result
            if markdown_result:
                self._apply_markdown(page, markdown_result)

            # Get JSON output if available and layout detection is enabled
            json_result = data.get("json_result")
            if json_result and isinstance(json_result, list) and len(json_result) > 0:
                # json_result is a list of pages, take the first page
                page_result = (
                    json_result[0] if isinstance(json_result[0], list) else json_result
                )
                self._apply_items(page, page_result)

            # Store raw response
            page.raw_response = orjson.dumps(result).decode("utf-8")

        return page
