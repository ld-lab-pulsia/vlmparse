import asyncio
from io import BytesIO
from typing import Literal, cast

import httpx
from loguru import logger
from PIL import Image
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.box import BoundingBox
from vlmparse.data_model.document import Item, Page
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
            "LOG_LEVEL": "DEBUG",
            "DOCLING_SERVE_ENG_LOC_NUM_WORKERS": "16",
            "DOCLING_NUM_THREADS": "32",
        }
    )

    def model_post_init(self, __context):
        if not self.docker_image:
            if self.cpu_only:
                self.docker_image = "quay.io/docling-project/docling-serve-cpu:latest"
            else:
                self.docker_image = "quay.io/docling-project/docling-serve:latest"
        if self.cpu_only and self.gpu_device_ids is None:
            self.gpu_device_ids = []
        if self.enable_ui:
            self.command_args.append("--enable-ui")

    @property
    def client_config(self):
        return DoclingConverterConfig(base_url=f"http://localhost:{self.docker_port}")


class DoclingConverterConfig(ConverterConfig):
    """Configuration for Docling converter client."""

    model_name: str = "docling"
    timeout: int = 300
    api_kwargs: dict = {
        "to_formats": ["md", "json"],
        "image_export_mode": "placeholder",
        "do_picture_classification": True,
    }

    def get_client(self, **kwargs) -> "DoclingConverter":
        return DoclingConverter(config=self, **kwargs)


def image_to_bytes(image: Image.Image) -> bytes:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def _resolve_ref(doc_json: dict, ref: str) -> dict | None:
    """Resolve a $ref like '#/texts/0' or '#/tables/0' into the actual item dict."""
    try:
        parts = ref.lstrip("#/").split("/")
        obj = doc_json
        for part in parts:
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        return obj
    except (KeyError, IndexError, ValueError):
        return None


def _iter_body_items(doc_json: dict):
    """Yield leaf doc items in body reading order, resolving groups recursively."""

    def _iter_children(children: list):
        for child_ref in children:
            ref = child_ref.get("$ref", "")
            item = _resolve_ref(doc_json, ref)
            if item is None:
                continue
            # Groups have children but no prov — recurse into them
            if "prov" not in item or not item.get("prov"):
                yield from _iter_children(item.get("children", []))
            else:
                yield item

    yield from _iter_children(doc_json.get("body", {}).get("children", []))


def _convert_bbox(bbox: dict, page_height: float | None) -> BoundingBox | None:
    """Convert a docling bbox dict (BOTTOMLEFT) to a vlmparse BoundingBox (TOPLEFT)."""
    try:
        l = bbox["l"]
        t = bbox["t"]
        r = bbox["r"]
        b = bbox["b"]
        if bbox.get("coord_origin") == "BOTTOMLEFT" and page_height:
            new_t = page_height - t
            new_b = page_height - b
            t, b = min(new_t, new_b), max(new_t, new_b)
        return BoundingBox(l=max(l, 0), t=max(t, 0), r=max(r, 0), b=max(b, 0))
    except Exception as e:
        logger.warning(f"Could not convert bbox {bbox}: {e}")
        return None


def table_to_html(table_item: dict) -> str:
    """Render a docling table item (with structured cell data) as an HTML table."""
    data = table_item.get("data", {})
    if not data:
        return ""

    num_rows = data.get("num_rows", 0)
    num_cols = data.get("num_cols", 0)
    cells = data.get("table_cells", [])

    if not cells or not num_rows or not num_cols:
        return ""

    # Build a sparse grid: (row, col) -> cell dict, tracking occupied slots
    occupied: set[tuple[int, int]] = set()
    sorted_cells = sorted(
        cells,
        key=lambda c: (
            c.get("start_row_offset_idx", 0),
            c.get("start_col_offset_idx", 0),
        ),
    )

    # Group cells by row
    rows: dict[int, list[dict]] = {}
    for cell in sorted_cells:
        row = cell.get("start_row_offset_idx", 0)
        rows.setdefault(row, []).append(cell)

    html = ["<table>"]
    for row_idx in range(num_rows):
        html.append("<tr>")
        col_cursor = 0
        for cell in rows.get(row_idx, []):
            start_col = cell.get("start_col_offset_idx", 0)
            row_span = cell.get("row_span", 1)
            col_span = cell.get("col_span", 1)
            text = cell.get("text", "")
            is_header = cell.get("column_header", False) or cell.get(
                "row_header", False
            )
            tag = "th" if is_header else "td"

            # Skip cells already occupied by a previous rowspan
            while (row_idx, col_cursor) in occupied:
                col_cursor += 1

            attrs = ""
            if row_span > 1:
                attrs += f' rowspan="{row_span}"'
            if col_span > 1:
                attrs += f' colspan="{col_span}"'

            html.append(f"<{tag}{attrs}>{text}</{tag}>")

            # Mark occupied slots
            for r in range(row_idx, row_idx + row_span):
                for c in range(start_col, start_col + col_span):
                    occupied.add((r, c))
            col_cursor = start_col + col_span

        html.append("</tr>")
    html.append("</table>")
    return "\n".join(html)


def extract_items_from_docling_json(
    doc_json: dict, page_height: float | None
) -> list[Item]:
    """Extract layout Items with bounding boxes in body reading order."""
    items = []
    for doc_item in _iter_body_items(doc_json):
        label = doc_item.get("label", "text")
        text = doc_item.get("orig") or doc_item.get("text") or ""
        self_ref = doc_item.get("self_ref")

        # Extract top predicted class from picture classification annotations
        class_name: str | None = None
        confidence: float | None = None
        if label == "picture":
            for annotation in doc_item.get("annotations", []):
                if annotation.get("kind") == "classification":
                    predicted = annotation.get("predicted_classes", [])
                    if predicted:
                        class_name = predicted[0].get("class_name")
                        confidence = predicted[0].get("confidence")
                    break

        for prov in doc_item.get("prov", []):
            bbox_dict = prov.get("bbox")
            if not bbox_dict:
                continue
            box = _convert_bbox(bbox_dict, page_height)
            if box is None:
                continue
            items.append(
                Item(
                    category=label,
                    text=text,
                    box=box,
                    id=self_ref,
                    class_name=class_name,
                    confidence=confidence,
                )
            )

        # Add captions as separate items referencing this figure
        if label in ("picture", "table") and self_ref:
            for cap_ref in doc_item.get("captions", []):
                ref = cap_ref.get("$ref", "")
                cap_item = _resolve_ref(doc_json, ref)
                if cap_item is None:
                    continue
                cap_text = cap_item.get("orig") or cap_item.get("text") or ""
                for prov in cap_item.get("prov", []):
                    bbox_dict = prov.get("bbox")
                    if not bbox_dict:
                        continue
                    box = _convert_bbox(bbox_dict, page_height)
                    if box is None:
                        continue
                    items.append(
                        Item(
                            category="caption",
                            text=cap_text,
                            box=box,
                            parent=self_ref,
                        )
                    )

    return items


def extract_text_from_docling_json(doc_json: dict) -> str:
    """Reconstruct markdown text in reading order; tables are rendered as HTML."""
    parts = []
    for doc_item in _iter_body_items(doc_json):
        label = doc_item.get("label", "")
        text = doc_item.get("orig") or doc_item.get("text") or ""

        if label == "table":
            html = table_to_html(doc_item)
            if html:
                parts.append(html)
        elif label == "section_header":
            level = doc_item.get("level", 1)
            if text.strip():
                parts.append(f"{'#' * level} {text}")
        elif label == "list_item":
            if text.strip():
                marker = doc_item.get("marker") or "-"
                parts.append(f"{marker} {text}")
        elif label == "picture":
            parts.append("![image]")
        else:
            if text.strip():
                parts.append(text)

    return "\n\n".join(parts)


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

    async def async_call_inside_page(self, page: Page) -> Page:
        """Process a single page using Docling Serve API."""
        assert page.image is not None, "Page image is required for processing"
        self.config = cast(DoclingConverterConfig, self.config)
        img_bytes = await asyncio.to_thread(image_to_bytes, page.image)

        data = self.config.api_kwargs
        url = f"{self.config.base_url}/v1/convert/file"
        logger.debug(f"Calling Docling API at: {url}")
        files = {"files": ("image.png", img_bytes, "image/png")}

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    url, files=files, data=data, headers={"Accept": "application/json"}
                )
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Docling API response status: {response.status_code}")

            doc_json = result["document"]["json_content"]

            # pages is a dict with string keys e.g. {'1': {'size': {...}, ...}}
            page_height: float | None = None
            pages = doc_json.get("pages", {})
            if pages:
                first_page = next(iter(pages.values()))
                page_height = first_page.get("size", {}).get("height")

            # Extract layout items with bounding boxes in reading order
            page.items = extract_items_from_docling_json(doc_json, page_height)
            logger.debug(f"Extracted {len(page.items)} layout items")

            # Extract text in reading order (tables rendered as HTML)
            text = extract_text_from_docling_json(doc_json)
            logger.debug(f"Extracted text length: {len(text)}")

            text = clean_response(text)
            text = html_to_md_keep_tables(text)
            page.text = text

        except Exception as e:
            logger.error(f"Error processing page with Docling: {e}")
            page.text = f"Error: {str(e)}"

        return page
