import asyncio
import base64
import os
from pathlib import Path

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

    file_type: int | None = None
    use_doc_orientation_classify: bool | None = None
    use_doc_unwarping: bool | None = None
    use_layout_detection: bool | None = None
    use_chart_recognition: bool | None = None
    use_seal_recognition: bool | None = None
    use_ocr_for_image_block: bool | None = None
    layout_threshold: float | dict | None = None
    layout_nms: bool | None = None
    layout_unclip_ratio: float | list | dict | None = None
    layout_merge_bboxes_mode: str | dict | None = None
    layout_shape_mode: str | None = None
    prompt_label: str | None = None
    format_block_content: bool | None = None
    repetition_penalty: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    max_new_tokens: int | None = None
    merge_layout_blocks: bool | None = None
    markdown_ignore_labels: list | None = None
    vlm_extra_args: dict | None = None
    prettify_markdown: bool | None = True
    show_formula_number: bool | None = False
    restructure_pages: bool | None = False
    merge_tables: bool | None = None
    relevel_titles: bool | None = None
    concatenate_pages: bool | None = None
    visualize: bool | None = None
    request_overrides: dict = Field(default_factory=dict)

    def get_client(self, **kwargs) -> "PaddleOCRVLConverter":
        return PaddleOCRVLConverter(config=self, **kwargs)


class PaddleOCRVLConverter(BaseConverter):
    """PaddleOCR-VL HTTP API converter."""

    config: PaddleOCRVLConverterConfig

    def _build_layout_payload(self, file_content_b64: str, file_type: int | None):
        payload: dict = {"file": file_content_b64}
        if file_type is not None:
            payload["fileType"] = file_type

        mapping = {
            "use_doc_orientation_classify": "useDocOrientationClassify",
            "use_doc_unwarping": "useDocUnwarping",
            "use_layout_detection": "useLayoutDetection",
            "use_chart_recognition": "useChartRecognition",
            "use_seal_recognition": "useSealRecognition",
            "use_ocr_for_image_block": "useOcrForImageBlock",
            "layout_threshold": "layoutThreshold",
            "layout_nms": "layoutNms",
            "layout_unclip_ratio": "layoutUnclipRatio",
            "layout_merge_bboxes_mode": "layoutMergeBboxesMode",
            "layout_shape_mode": "layoutShapeMode",
            "prompt_label": "promptLabel",
            "format_block_content": "formatBlockContent",
            "repetition_penalty": "repetitionPenalty",
            "temperature": "temperature",
            "top_p": "topP",
            "min_pixels": "minPixels",
            "max_pixels": "maxPixels",
            "max_new_tokens": "maxNewTokens",
            "merge_layout_blocks": "mergeLayoutBlocks",
            "markdown_ignore_labels": "markdownIgnoreLabels",
            "vlm_extra_args": "vlmExtraArgs",
            "prettify_markdown": "prettifyMarkdown",
            "show_formula_number": "showFormulaNumber",
            "restructure_pages": "restructurePages",
            "merge_tables": "mergeTables",
            "relevel_titles": "relevelTitles",
            "visualize": "visualize",
        }

        for attr, key in mapping.items():
            value = getattr(self.config, attr)
            if value is not None:
                payload[key] = value

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

        payload = {"pages": pages}

        if self.config.merge_tables is not None:
            payload["mergeTables"] = self.config.merge_tables
        if self.config.relevel_titles is not None:
            payload["relevelTitles"] = self.config.relevel_titles
        if self.config.concatenate_pages is not None:
            payload["concatenatePages"] = self.config.concatenate_pages
        if self.config.prettify_markdown is not None:
            payload["prettifyMarkdown"] = self.config.prettify_markdown
        if self.config.show_formula_number is not None:
            payload["showFormulaNumber"] = self.config.show_formula_number

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

    @staticmethod
    def _file_to_base64(file_path: str | Path) -> str:
        file_path = Path(file_path)
        content = file_path.read_bytes()
        return base64.b64encode(content).decode("utf-8")

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
