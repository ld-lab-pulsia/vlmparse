"""Base converter for layout detection + per-region VLM recognition."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Literal

import httpx
from loguru import logger
from PIL import Image

from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import BoundingBox, Item, Page, TextCell
from vlmparse.utils import to_base64

NativeTextStrategy = Literal["always", "any_overlap", "edge_overlap"]


def _assemble_text_cells(cells: list[TextCell]) -> str:
    """Reconstruct reading-order text from a list of word-level TextCells."""
    if not cells:
        return ""
    # Sort top-to-bottom, left-to-right
    cells = sorted(cells, key=lambda c: (c.box.t, c.box.l))
    lines: list[list[TextCell]] = []
    current: list[TextCell] = [cells[0]]
    for cell in cells[1:]:
        prev = current[-1]
        prev_cy = (prev.box.t + prev.box.b) / 2
        cell_cy = (cell.box.t + cell.box.b) / 2
        cell_h = max(cell.box.b - cell.box.t, 1)
        if abs(cell_cy - prev_cy) < cell_h * 0.6:
            current.append(cell)
        else:
            lines.append(sorted(current, key=lambda c: c.box.l))
            current = [cell]
    lines.append(sorted(current, key=lambda c: c.box.l))
    return "\n".join(" ".join(c.to_markdown() for c in line) for line in lines)


class LayoutVLMConverterConfig(ConverterConfig):
    """Config for a layout-detection + VLM-recognition pipeline.

    Fields:
        base_url:          URL of the layout detection server (LitServe).
        vlm_base_url:      URL of the VLM server (vLLM / OpenAI-compatible).
        vlm_model_id:      Model name to send in the VLM API request.
        layout_endpoint:   Endpoint path on the layout server.
        max_tokens:        Max tokens for VLM generation.
        temperature:       Sampling temperature for VLM generation.
        timeout:           HTTP timeout (seconds) for both servers.
    """

    model_name: str = "layout-vlm"
    vlm_base_url: str | None = None
    vlm_model_id: str = "default"
    layout_endpoint: str = "/predict"
    max_tokens: int = 4096
    temperature: float = 0.8
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    timeout: int = 300
    native_text_strategy: NativeTextStrategy = "always"
    """How to use docling-parse native text for *text* regions.

    - ``"always"``      – always call the VLM (default, no change).
    - ``"any_overlap"`` – skip VLM if any native text cell overlaps the region.
    - ``"edge_overlap"``– skip VLM only when cells are present at the left,
                          right **and** top edges of the region (indicating the
                          box is fully covered with selectable text).

    Not applied to table, formula or image/chart regions.
    """

    def get_client(self, **kwargs) -> "LayoutVLMConverter":
        return LayoutVLMConverter(config=self, **kwargs)


class LayoutVLMConverter(BaseConverter):
    """Converter that detects layout regions then calls a VLM on each region.

    Override points for subclasses:
        _get_label_task(label) -> str
            Classify a detected layout label as 'text' (recognize), 'skip'
            (keep region but do not OCR) or 'abandon' (drop region entirely).
            Default: return 'text' for every label.

        _get_prompt(label) -> str | None
            Return an instruction string to append after the image in the VLM
            message, or None for image-only input.
            Default: return None.
    """

    config: LayoutVLMConverterConfig

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def _get_label_task(self, label: str) -> str:
        """Classify a layout label.

        Returns one of:
            'text'    – crop and send to VLM for recognition
            'skip'    – keep as an Item but do not call VLM
            'abandon' – discard entirely
        """
        return "text"

    def _get_prompt(self, label: str) -> str | None:
        """Return a task-specific prompt for the given label, or None."""
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _detect_layout(self, image: "Image.Image") -> list[dict]:
        """Call the layout server and return a list of detected regions.

        Each region is a dict: {label: str, score: float, bbox: [x1, y1, x2, y2]}.
        """
        b64 = await asyncio.to_thread(to_base64, image, "JPEG")
        async with httpx.AsyncClient(
            base_url=self.config.base_url or "", timeout=self.config.timeout
        ) as client:
            response = await client.post(
                self.config.layout_endpoint, json={"image": b64}
            )
        response.raise_for_status()
        return response.json()["regions"]

    @staticmethod
    def _smart_resize_image(image: "Image.Image") -> "Image.Image":
        """Resize image to fit within GLM-OCR's max_pixels budget.

        Matches the reference implementation's PageLoaderConfig defaults:
            max_pixels = 14 * 14 * 4 * 1280  (= 1,003,520)
            factor     = 28  (= 14 * 2 * patch_expand_factor(1))
        """
        max_pixels = 14 * 14 * 4 * 1280
        min_pixels = 112 * 112
        factor = 28

        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        h_bar = round(h / factor) * factor
        w_bar = round(w / factor) * factor

        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h * w) / max_pixels)
            h_bar = math.floor(h / beta / factor) * factor
            w_bar = math.floor(w / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (h * w))
            h_bar = math.ceil(h * beta / factor) * factor
            w_bar = math.ceil(w * beta / factor) * factor

        if (h_bar, w_bar) != (h, w):
            image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)
        return image

    def _collect_native_text(
        self,
        bbox: list,
        text_cells: list[TextCell],
    ) -> str | None:
        """Return assembled native text for *bbox* according to the configured
        strategy, or ``None`` if the VLM should be called instead."""
        strategy = self.config.native_text_strategy
        if strategy == "always" or not text_cells:
            return None

        x1, y1, x2, y2 = bbox
        overlapping = [
            cell
            for cell in text_cells
            if cell.box.r > x1
            and cell.box.l < x2
            and cell.box.b > y1
            and cell.box.t < y2
        ]
        if not overlapping:
            return None

        if strategy == "any_overlap":
            return _assemble_text_cells(overlapping)

        # edge_overlap: require cells at left, right AND top edges
        tol_x = (x2 - x1) * 0.05
        tol_y = (y2 - y1) * 0.05
        has_left = any(cell.box.l <= x1 + tol_x for cell in overlapping)
        has_right = any(cell.box.r >= x2 - tol_x for cell in overlapping)
        has_top = any(cell.box.t <= y1 + tol_y for cell in overlapping)
        if has_left and has_right and has_top:
            return _assemble_text_cells(overlapping)
        return None

    async def _recognize_region(self, image: "Image.Image", prompt: str | None) -> str:
        """Send a cropped region to the VLM and return the generated text."""
        image = await asyncio.to_thread(self._smart_resize_image, image)
        b64 = await asyncio.to_thread(to_base64, image, "JPEG")
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        ]
        if prompt:
            content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.config.vlm_model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            payload["top_k"] = self.config.top_k
        if self.config.repetition_penalty is not None:
            payload["repetition_penalty"] = self.config.repetition_penalty
        assert (
            self.config.vlm_base_url
        ), "vlm_base_url is required for region recognition"
        async with httpx.AsyncClient(
            base_url=self.config.vlm_base_url, timeout=self.config.timeout
        ) as client:
            response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image
        import time

        assert image is not None, "Page image is required"
        t0 = time.perf_counter()
        regions = await self._detect_layout(image)
        t1 = time.perf_counter()
        logger.debug(f"Layout: {t1-t0:.3f}s  ({len(regions)} regions)")
        if not regions:
            logger.debug("No layout regions detected, falling back to full-page OCR")
            page.text = await self._recognize_region(image, self._get_prompt("text"))
            return page

        # Classify and filter regions
        width, height = image.size
        classified: list[tuple[str, list, str]] = []  # (label, bbox, task)
        for region in regions:
            label = region["label"]
            task = self._get_label_task(label)
            if task != "abandon":
                x1, y1, x2, y2 = region["bbox"]  # [x1, y1, x2, y2]
                x1 = max(0, min(width, x1))
                y1 = max(0, min(height, y1))
                x2 = max(0, min(width, x2))
                y2 = max(0, min(height, y2))
                if x1 < x2 and y1 < y2:
                    classified.append((label, [x1, y1, x2, y2], task))

        if not classified:
            logger.debug("All regions abandoned, falling back to full-page OCR")
            page.text = await self._recognize_region(image, self._get_prompt("text"))
            return page

        # Crop + recognize in parallel (skip regions get empty text)
        n_skipped = 0
        n_native = 0
        n_vlm = 0

        async def process_region(label: str, bbox: list, task: str) -> str:
            nonlocal n_skipped, n_native, n_vlm
            if task == "skip":
                n_skipped += 1
                return ""
            if task == "text" and page.text_cells:
                native = self._collect_native_text(bbox, page.text_cells)
                if native is not None:
                    n_native += 1
                    return native
            n_vlm += 1
            x1, y1, x2, y2 = bbox
            cropped = image.crop((x1, y1, x2, y2))
            return await self._recognize_region(cropped, self._get_prompt(label))

        texts = await asyncio.gather(
            *[process_region(label, bbox, task) for label, bbox, task in classified]
        )
        t2 = time.perf_counter()
        logger.debug(
            "VLM (parallel, {n_vlm} regions): {dt:.3f}s  "
            "[native={n_native}, skipped={n_skipped}]",
            n_vlm=n_vlm,
            dt=t2 - t1,
            n_native=n_native,
            n_skipped=n_skipped,
        )
        items = []
        for (label, bbox, _), text in zip(classified, texts, strict=False):
            x1, y1, x2, y2 = bbox
            items.append(
                Item(
                    text=text,
                    box=BoundingBox(l=x1, t=y1, r=x2, b=y2),
                    category=label,
                )
            )

        page.items = items
        return page
