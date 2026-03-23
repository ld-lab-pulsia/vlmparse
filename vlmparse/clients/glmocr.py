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
from vlmparse.servers.container_group_server import (
    ContainerGroupServerConfig,
    ServiceDefinition,
)
from vlmparse.utils import to_base64

DOCKER_PIPELINE_DIR = (
    Path(__file__).parent.parent.parent / "docker_pipelines" / "glmocr"
)


class GLMOCRDockerServerConfig(ContainerGroupServerConfig):
    """Container group configuration for GLM-OCR server."""

    model_name: str = "GLM-OCR"
    aliases: list[str] = Field(default_factory=lambda: ["glmocr", "glm-ocr"])
    server_service: str = "glmocr-api"
    docker_port: int = 5002
    gpu_services: list[str] = Field(default_factory=lambda: ["glmocr-vllm-server"])
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: ["Running on", "Application startup complete"]
    )
    services: dict[str, ServiceDefinition] = Field(default_factory=dict)

    def model_post_init(self, __context):
        hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")

        # The glmocr-api container resolves the vllm server by its container name
        # on the shared bridge network.  The container name is
        # "{resolved_group_name}-glmocr-vllm-server".
        vllm_container_name = f"{self.resolved_group_name}-glmocr-vllm-server"

        self.services = {
            "glmocr-vllm-server": ServiceDefinition(
                image="vlmparse-glmocr-vllm-server:latest",
                dockerfile_dir=DOCKER_PIPELINE_DIR / "vllm-server",
                internal_port=8080,
                ready_indicators=["Application startup complete."],
                entrypoint=["/bin/bash", "-c"],
                command=[
                    "vllm serve zai-org/GLM-OCR"
                    " --served-model-name glm-ocr"
                    " --allowed-local-media-path /"
                    " --port 8080"
                    ' --speculative-config \'{"method": "mtp", "num_speculative_tokens": 1}\''
                ],
                environment={
                    "CUDA_VISIBLE_DEVICES": cuda_devices,
                    "HF_HOME": "/root/.cache/huggingface",
                },
                volumes={
                    hf_home: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                },
                shm_size="16g",
            ),
            "glmocr-api": ServiceDefinition(
                image="vlmparse-glmocr-api:latest",
                dockerfile_dir=DOCKER_PIPELINE_DIR,
                internal_port=5002,
                entrypoint=["/bin/bash", "-c"],
                command=[
                    # Copy the mounted config to a writable location, patch
                    # api_host so it resolves to the vLLM sibling container
                    # on the shared bridge network, then start the server.
                    "cp /app/config-mount/config.yaml /app/GLM-OCR/glmocr/config.yaml"
                    f" && sed -i 's/api_host: glmocr-vllm-server/api_host: {vllm_container_name}/'"
                    " /app/GLM-OCR/glmocr/config.yaml"
                    " && python -m glmocr.server --log-level ${LOG_LEVEL:-INFO}"
                ],
                environment={
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                },
                volumes={
                    str(DOCKER_PIPELINE_DIR / "config.yaml"): {
                        "bind": "/app/config-mount/config.yaml",
                        "mode": "ro",
                    },
                },
            ),
        }

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
    max_tokens: int = 4096
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
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
            "images": [data_uri],  # GLM-OCR expects a list
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
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
        assert (
            self.config.endpoint.base_url is not None
        ), "Base URL is required for API calls"
        async with httpx.AsyncClient(
            base_url=self.config.endpoint.base_url,
            timeout=self.config.timeout,
            headers=headers,
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

        image = page.image
        if image is None:
            return

        img_width, img_height = image.size
        items: list[Item] = []

        for block in json_result:
            bbox = block.get("bbox_2d")
            if not bbox or len(bbox) != 4:
                # If no bbox, skip this item
                continue

            # bbox_2d is in 0-1000 normalized coordinates; convert to pixels
            x1 = bbox[0] * img_width / 1000
            y1 = bbox[1] * img_height / 1000
            x2 = bbox[2] * img_width / 1000
            y2 = bbox[3] * img_height / 1000
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
