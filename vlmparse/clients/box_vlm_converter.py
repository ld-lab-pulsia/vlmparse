"""Generalist VLM converters with bounding-box detection.

Two variants:

* **BoxVLMConverterClient** – full layout extraction.  The VLM returns a JSON
  array where every detected region has a category, bounding box, and text.

* **ImageBoxVLMConverterClient** – image-only box extraction.  The VLM does a
  normal markdown transcription but wraps every image/figure region in a
  ``<region>`` tag that carries the bounding box.  A regex post-processes the
  output to split it into ``Item`` objects (images get boxes, text segments
  don't).
"""

from __future__ import annotations

import json
import re

from loguru import logger
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.data_model.document import BoundingBox, Item, Page
from vlmparse.utils import to_base64

from .openai_converter import OpenAIConverterClient, OpenAIConverterConfig

# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)


def _gemini_box_to_bbox(coords: list[float]) -> BoundingBox:
    """Convert Gemini-format [y_min, x_min, y_max, x_max] (0-1000) to a BoundingBox."""
    y_min, x_min, y_max, x_max = coords
    return BoundingBox(
        l=float(x_min),
        t=float(y_min),
        r=float(x_max),
        b=float(y_max),
        coord_origin="TOPLEFT",
    )


def _build_messages(config, image, prompt_text):
    """Build the OpenAI-style messages list from config, image and prompt."""
    if config.use_response_api:
        text_key = "input_text"
        image_payload = {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{to_base64(image)}",
        }
    else:
        text_key = "text"
        image_payload = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{to_base64(image)}"},
        }

    preprompt = (
        [{"role": "system", "content": [{"type": text_key, "text": config.preprompt}]}]
        if config.preprompt
        else []
    )

    return [
        *preprompt,
        {
            "role": "user",
            "content": [
                image_payload,
                {"type": text_key, "text": prompt_text},
            ],
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Version 1 – Full layout extraction  (JSON array of all regions)
# ═══════════════════════════════════════════════════════════════════════════

_FULL_LAYOUT_PROMPT_TEMPLATE = """\
Analyze the document image and detect all content regions.
For each region, return a JSON object with:
- "category": one of the categories listed below
- "box": the bounding box as [y_min, x_min, y_max, x_max] with coordinates \
normalized to 0-1000 (top-left origin, x horizontal, y vertical)
- "text": the text content of the region (use markdown, LaTeX for formulas, \
HTML <table> for tables). For images, provide a short description.

Categories:
{categories}

Return ONLY a JSON array of objects. No commentary, no markdown fences.\
"""


def _build_full_layout_prompt(categories: dict[str, str]) -> str:
    lines = "\n".join(f'- "{cat}": {desc}' for cat, desc in categories.items())
    return _FULL_LAYOUT_PROMPT_TEMPLATE.format(categories=lines)


def _parse_full_layout_response(raw: str, valid_categories: set[str]) -> list[Item]:
    """Parse a JSON array of detected regions into ``Item`` objects."""
    text = raw.strip()
    m = _JSON_BLOCK_RE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    items: list[Item] = []
    for entry in json.loads(text):
        box_coords = entry.get("box") or entry.get("bbox")
        category = entry.get("category", "other")
        region_text = entry.get("text", "")

        if not box_coords or len(box_coords) != 4:
            logger.warning(f"Skipping entry with invalid box: {entry}")
            continue

        if category not in valid_categories:
            logger.debug(f"Category '{category}' not in config; keeping as-is.")

        items.append(
            Item(
                category=category,
                box=_gemini_box_to_bbox(box_coords),
                text=region_text,
            )
        )
    return items


class BoxVLMConverterConfig(OpenAIConverterConfig):
    """Config for full-layout VLM detection (all regions as JSON).

    Attributes:
        categories: ``{category: description}`` dict sent to the VLM.
        box_prompt: Optional override for the auto-built prompt.
    """

    categories: dict[str, str] = Field(
        default_factory=lambda: {
            "text": "Regular text paragraph or block",
            "title": "Section heading or document title",
            "table": "Tabular data",
            "figure": "Image, chart, diagram, or graph",
            "formula": "Mathematical equation or formula",
            "caption": "Caption associated with a figure or table",
            "header": "Page header",
            "footer": "Page footer or page number",
            "list": "Bulleted or numbered list",
        }
    )
    box_prompt: str | None = None

    def get_client(self, **kwargs) -> "BoxVLMConverterClient":
        return BoxVLMConverterClient(config=self, **kwargs)


class BoxVLMConverterClient(OpenAIConverterClient):
    """Full-layout VLM client – returns classified bounding boxes for every region."""

    config: BoxVLMConverterConfig

    def _get_detection_prompt(self) -> str:
        return self.config.box_prompt or _build_full_layout_prompt(
            self.config.categories
        )

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image
        assert image is not None, "Page image is required for conversion"

        messages = _build_messages(self.config, image, self._get_detection_prompt())
        response, usage = await self._get_chat_completion(messages)
        logger.debug("Box detection response: " + str(response))
        page.raw_response = response

        try:
            items = _parse_full_layout_response(response, set(self.config.categories))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error(f"Failed to parse box response: {exc}")
            items = []

        page.items = items
        page.text = "\n\n".join(
            it.text for it in sorted(items, key=lambda i: (i.box.t, i.box.l)) if it.text
        )
        page = self.add_usage(page, usage)
        return page


# ═══════════════════════════════════════════════════════════════════════════
# Version 2 – Image-only box extraction  (markdown + <region> tags)
# ═══════════════════════════════════════════════════════════════════════════

_IMAGE_BOX_PROMPT_TEMPLATE = """\
Please convert the document image to Markdown following these rules:

1. Accurately recognize all text and convert it to well-structured Markdown.
2. Use LaTeX ($ for inline, $$ for block) for mathematical formulas.
3. Use HTML <table> tags for tables.
4. For EVERY image, figure, chart, diagram or graph in the document, output \
a <region> tag instead of a normal image placeholder. The tag format is:
   <region box="y_min,x_min,y_max,x_max" category="CATEGORY">description of the image</region>
   - The box coordinates must be normalized to 0-1000 (top-left origin).
   - CATEGORY must be one of: {categories}
   - The description should summarize the visual content.
5. Do NOT wrap your output in code fences.
6. If the page is blank, return only <blank>.\
"""


def _build_image_box_prompt(categories: dict[str, str]) -> str:
    cat_list = ", ".join(f'"{c}"' for c in categories)
    return _IMAGE_BOX_PROMPT_TEMPLATE.format(categories=cat_list)


# Regex that matches <region box="..." category="...">...</region>
_REGION_TAG_RE = re.compile(
    r'<region\s+box="(?P<box>[^"]+)"\s+category="(?P<cat>[^"]+)">'
    r"(?P<text>.*?)</region>",
    re.DOTALL,
)


# Placeholder box for text items (no positional info from the VLM)
_TEXT_PLACEHOLDER_BOX = BoundingBox(l=0, t=0, r=1000, b=1000, coord_origin="TOPLEFT")


def _text_to_items(text: str) -> list[Item]:
    """Split a text block on double-newlines into text Items."""
    items: list[Item] = []
    for chunk in re.split(r"\n{2,}", text):
        chunk = chunk.strip()
        if chunk:
            items.append(
                Item(
                    category="text",
                    box=_TEXT_PLACEHOLDER_BOX.model_copy(),
                    text=chunk,
                )
            )
    return items


def _parse_image_box_response(
    raw: str,
    valid_categories: set[str],
) -> tuple[list[Item], str]:
    """Split a markdown+<region> response into text and image Items.

    The response is split at every ``<region>`` tag.  Segments between tags are
    further split on double-newlines into ``text`` Items.  Each ``<region>``
    becomes an image Item with its bounding box.

    Returns ``(items, markdown_text)`` where *items* is the ordered list of
    **all** content (text + images) and *markdown_text* is the cleaned markdown
    with ``<region>`` tags replaced by ``![description][image-placeholder]``.
    """
    items: list[Item] = []
    cleaned = raw
    last_end = 0

    for m in _REGION_TAG_RE.finditer(raw):
        # Text segment before this <region>
        text_before = raw[last_end : m.start()]
        items.extend(_text_to_items(text_before))

        # Image item from <region>
        box_str = m.group("box")
        category = m.group("cat")
        description = m.group("text").strip()

        coords = [float(c) for c in box_str.split(",")]
        if len(coords) != 4:
            logger.warning(f"Skipping region with bad box: {box_str}")
        else:
            if category not in valid_categories:
                logger.debug(f"Category '{category}' not in config; keeping as-is.")
            items.append(
                Item(
                    category=category,
                    box=_gemini_box_to_bbox(coords),
                    text=description,
                )
            )

        cleaned = cleaned.replace(
            m.group(0),
            f"![{description}][image-placeholder]",
        )
        last_end = m.end()

    # Trailing text after the last <region>
    items.extend(_text_to_items(raw[last_end:]))

    return items, cleaned


class ImageBoxVLMConverterConfig(OpenAIConverterConfig):
    """Config for image-only box detection (markdown output with <region> tags).

    Attributes:
        image_categories: ``{category: description}`` for image-like regions.
        image_box_prompt: Optional override for the auto-built prompt.
    """

    image_categories: dict[str, str] = Field(
        default_factory=lambda: {
            "figure": "Photograph, illustration, or generic image",
            "chart": "Statistical chart or graph (bar, line, pie, etc.)",
            "diagram": "Technical diagram, flowchart, or schematic",
            "map": "Geographic map",
            "logo": "Logo or brand mark",
        }
    )
    image_box_prompt: str | None = None

    def get_client(self, **kwargs) -> "ImageBoxVLMConverterClient":
        return ImageBoxVLMConverterClient(config=self, **kwargs)


class ImageBoxVLMConverterClient(OpenAIConverterClient):
    """Image-only box VLM client – normal markdown OCR with detected image boxes.

    The VLM produces standard markdown but wraps images in ``<region>`` tags.
    Post-processing splits the response into an ordered list of Items:

    * **image Items** have their bounding box and category from ``<region>``
    * **text Items** are split on double-newlines with a placeholder box

    ``page.text`` contains the cleaned markdown with image placeholders.
    """

    config: ImageBoxVLMConverterConfig

    def _get_detection_prompt(self) -> str:
        return self.config.image_box_prompt or _build_image_box_prompt(
            self.config.image_categories,
        )

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image
        assert image is not None, "Page image is required for conversion"

        messages = _build_messages(self.config, image, self._get_detection_prompt())
        response, usage = await self._get_chat_completion(messages)
        logger.debug("Image-box response: " + str(response))
        page.raw_response = response

        try:
            items, md_text = _parse_image_box_response(
                response,
                set(self.config.image_categories),
            )
        except Exception as exc:
            logger.error(f"Failed to parse image-box response: {exc}")
            items, md_text = [], response

        page.items = items if items else None
        md_text = clean_response(md_text)
        page.text = html_to_md_keep_tables(md_text)
        page = self.add_usage(page, usage)
        return page
