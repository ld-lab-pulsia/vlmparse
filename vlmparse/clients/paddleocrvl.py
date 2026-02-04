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
    Path(__file__).parent.parent.parent / "docker_pipelines" / "paddleocrvl"
)


class PaddleOCRVLDockerServerConfig(DockerComposeServerConfig):
    """Docker Compose configuration for PaddleOCR-VL server."""

    model_name: str = "PaddleOCR-VL-1.5"
    aliases: list[str] = Field(
        default_factory=lambda: ["paddleocrvl1.5", "paddleocr-vl-1.5"]
    )
    compose_file: str = str(DOCKER_PIPELINE_DIR / "compose.yaml")
    server_service: str = "paddleocr-vl-api"
    compose_services: list[str] = Field(
        default_factory=lambda: ["paddleocr-vl-api", "paddleocr-vlm-server"]
    )
    gpu_service_names: list[str] = Field(
        default_factory=lambda: ["paddleocr-vl-api", "paddleocr-vlm-server"]
    )
    docker_port: int = 8080
    container_port: int = 8080
    environment: dict[str, str] = Field(
        default_factory=lambda: {
            "VLM_BACKEND": "vllm",
        }
    )
    environment_services: list[str] = Field(
        default_factory=lambda: ["paddleocr-vl-api"]
    )
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: ["Application startup complete", "Uvicorn running"]
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
        return PaddleOCRVLConverterConfig(
            **self._create_client_kwargs(f"http://localhost:{self.docker_port}")
        )


class PaddleOCRVLConverterConfig(ConverterConfig):
    """Configuration for PaddleOCR-VL API client."""

    model_name: str = "PaddleOCR-VL-1.5"
    aliases: list[str] = Field(
        default_factory=lambda: ["paddleocrvl1.5", "paddleocr-vl-1.5"]
    )
    timeout: int = 600

    endpoint_layout_parsing: str = "/layout-parsing"
    endpoint_restructure_pages: str = "/restructure-pages"

    # Dict of PaddleOCR-VL API args.
    # Keys should match the PaddleOCR-VL API JSON fields (camelCase), e.g.
    # {"useLayoutDetection": true, "promptLabel": "..."}.
    paddleocr_args: dict[str, Any] = Field(
        default_factory=lambda: {
            # Preserve previous default behavior (these were always sent before).
            "prettifyMarkdown": True,
            "showFormulaNumber": False,
            "restructurePages": False,
        }
    )

    # Optional args for the /restructure-pages endpoint (if/when used).
    restructure_args: dict[str, Any] = Field(default_factory=dict)

    # Backward-compat escape hatch: if set, applied last to the payload.
    request_overrides: dict[str, Any] = Field(default_factory=dict)

    def get_client(self, **kwargs) -> "PaddleOCRVLConverter":
        return PaddleOCRVLConverter(config=self, **kwargs)


class PaddleOCRVLConverter(BaseConverter):
    """PaddleOCR-VL HTTP API converter."""

    config: PaddleOCRVLConverterConfig

    def _build_layout_payload(self, file_content_b64: str, file_type: int | None):
        payload: dict[str, Any] = {"file": file_content_b64}

        if self.config.paddleocr_args:
            payload.update(self.config.paddleocr_args)

        if file_type is not None:
            payload["fileType"] = file_type

        if self.config.request_overrides:
            payload.update(self.config.request_overrides)

        return payload

    def _build_restructure_payload(self, layout_results: list[dict]) -> dict:
        pages = []
        for page_result in layout_results:
            pruned = page_result.get("prunedResult")
            markdown = page_result.get("markdown") or {}
            if pruned is None:
                continue
            pages.append(
                {
                    "prunedResult": pruned,
                    "markdownImages": markdown.get("images"),
                }
            )

        payload: dict[str, Any] = {"pages": pages}

        if self.config.restructure_args:
            payload.update(self.config.restructure_args)

        return payload

    async def _post_json(self, endpoint: str, payload: dict) -> dict:
        async with httpx.AsyncClient(
            base_url=self.config.base_url, timeout=self.config.timeout
        ) as client:
            response = await client.post(endpoint, json=payload)

        response.raise_for_status()
        data = response.json()
        if data.get("errorCode", 0) != 0:
            raise RuntimeError(data.get("errorMsg", "Unknown error"))
        return data

    def _apply_markdown(self, page: Page, markdown_text: str | None):
        text = markdown_text or ""
        text = clean_response(text)
        text = html_to_md_keep_tables(text)
        logger.debug(f"Converted markdown text: {text}...")
        page.text = text

    def _apply_items(self, page: Page, pruned_result: dict | None):
        if not pruned_result:
            return
        parsing_res_list = pruned_result.get("parsing_res_list") or []
        items: list[Item] = []
        for block in parsing_res_list:
            bbox = block.get("block_bbox")
            if not bbox or len(bbox) != 4:
                logger.warning(f"Invalid bbox in block: {block}")
                continue
            l, t, r, b = bbox
            text = block.get("block_content") or ""
            items.append(
                Item(
                    text=text,
                    box=BoundingBox(l=l, t=t, r=r, b=b),
                    category=block.get("block_label") or "",
                )
            )

        page.items = items

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image
        file_content_b64 = await asyncio.to_thread(to_base64, image, "PNG")
        payload = self._build_layout_payload(file_content_b64, 1)

        data = await self._post_json(self.config.endpoint_layout_parsing, payload)
        result = data.get("result", {})
        layout_results = result.get("layoutParsingResults", [])
        if layout_results:
            first = layout_results[0]

            markdown = first.get("markdown") or {}
            self._apply_markdown(page, markdown.get("text"))
            self._apply_items(page, first.get("prunedResult"))
            page.raw_response = orjson.dumps(first).decode("utf-8")

        return page
