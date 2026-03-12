"""Unified category mapping for document layout elements across OCR backends.

All raw category strings from the various clients are normalised
(lower-cased, hyphens/spaces → underscores) and looked up in a single
consolidated dict.  Unknown categories fall back to ``"other"``.

Unified categories
------------------
table, image, text, footer, header, list_item, title, footnote, caption,
formula, other
"""

from typing import Literal

UnifiedCategory = Literal[
    "table",
    "image",
    "text",
    "footer",
    "header",
    "list_item",
    "title",
    "footnote",
    "caption",
    "formula",
    "other",
]

# Maps normalised raw category → unified category.
# Normalisation: raw.lower().replace("-", "_").replace(" ", "_")
_CATEGORY_MAP: dict[str, str] = {
    # ── table ─────────────────────────────────────────────────────────────
    "table": "table",
    # ── image / figure / picture / chart ──────────────────────────────────
    "image": "image",
    "picture": "image",
    "chart": "image",
    "figure": "image",
    "footer_image": "image",
    "header_image": "image",
    # ── caption ───────────────────────────────────────────────────────────
    "caption": "caption",
    "figure_caption": "caption",  # MinerU
    "figure_title": "caption",  # GLM-OCR / PP-DocLayout
    "table_caption": "caption",  # MinerU
    # ── footnote ──────────────────────────────────────────────────────────
    "footnote": "footnote",
    "vision_footnote": "footnote",  # GLM-OCR
    "table_footnote": "footnote",  # MinerU
    # ── footer ────────────────────────────────────────────────────────────
    "footer": "footer",
    "page_footer": "footer",
    # ── header ────────────────────────────────────────────────────────────
    "header": "header",
    "page_header": "header",
    # ── title / section header ────────────────────────────────────────────
    "title": "title",
    "section_header": "title",  # Docling / DotsOCR normalised
    "doc_title": "title",  # GLM-OCR / PP-DocLayout
    "paragraph_title": "title",  # GLM-OCR / PP-DocLayout
    # ── list item ─────────────────────────────────────────────────────────
    "list_item": "list_item",
    "list_group": "list_item",  # Chandra
    # ── formula / equation ────────────────────────────────────────────────
    "formula": "formula",
    "display_formula": "formula",  # GLM-OCR / PP-DocLayout
    "inline_formula": "formula",  # GLM-OCR / PP-DocLayout
    "equation_block": "formula",  # Chandra
    "equation": "formula",
    "formula_number": "formula",  # GLM-OCR / PP-DocLayout
    "isolate_formula": "formula",  # MinerU
    "embeded_formula": "formula",  # MinerU (typo preserved intentionally)
    "interline_equation": "formula",  # MinerU
    # ── text (generic prose / paragraphs) ─────────────────────────────────
    "text": "text",
    "paragraph": "text",
    "abstract": "text",
    "algorithm": "text",
    "content": "text",
    "reference_content": "text",
    "vertical_text": "text",
    "aside_text": "text",
    "reference": "text",
    # ── other / miscellaneous ─────────────────────────────────────────────
    "number": "other",
    "seal": "other",
    "complex_block": "other",
    "code_block": "other",
    "code": "other",
    "form": "other",
    "table_of_contents": "other",
    "abandon": "other",
    "other": "other",
    "unknown": "other",
}


def normalize_category(category: str) -> str:
    """Lowercase and replace hyphens/spaces with underscores."""
    return category.lower().replace("-", "_").replace(" ", "_")


def map_to_unified_category(category: str) -> str:
    """Return the unified category for *category*, falling back to ``"other"``."""
    return _CATEGORY_MAP.get(normalize_category(category), "other")
