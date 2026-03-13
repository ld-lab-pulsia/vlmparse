"""Native PDF text extraction using docling-parse.

Extracts word-level bounding boxes and text from PDF pages without any VLM.
Coordinates in the returned TextCell objects are in TOPLEFT origin, expressed
in PDF points.  The converter scales them to image-pixel space before storing
them on the page, so ``TextCell.box`` and ``Item.box`` share the same space.
"""

from pathlib import Path

from loguru import logger

from vlmparse.constants import PDF_EXTENSION
from vlmparse.data_model.box import BoundingBox
from vlmparse.data_model.document import TextCell


def extract_page_text_cells(
    file_path: str | Path, page_idx: int
) -> tuple[list[TextCell], float, float] | tuple[None, None, None]:
    """Extract word-level text cells from a single PDF page.

    Uses docling-parse to retrieve native PDF text with bounding boxes.
    This function is synchronous and is intended to be called via
    ``asyncio.to_thread`` to avoid blocking the event loop.

    Parameters
    ----------
    file_path:
        Path to the PDF file.
    page_idx:
        Zero-based page index.

    Returns
    -------
    (cells, pdf_page_width, pdf_page_height) on success, or
    (None, None, None) when the file is not a PDF or extraction fails.
    """
    if Path(file_path).suffix.lower() != PDF_EXTENSION:
        return None, None, None

    try:
        from docling_parse.pdf_parsers import (  # type: ignore[import]
            DecodePageConfig,
            pdf_parser,
        )
    except ImportError:
        logger.warning(
            "docling-parse is not installed; skipping native text extraction; install with `pip install vlmparse[docling-parse]`"
        )
        return None, None, None

    try:
        parser = pdf_parser("fatal")
        key = str(file_path)
        loaded = False
        try:
            parser.load_document(key, key)
            loaded = True

            config = DecodePageConfig()
            config.create_word_cells = True

            decoder = parser.get_page_decoder(key, page_idx, config)

            media_bbox = decoder.get_page_dimension().get_media_bbox()
            pdf_page_width = float(media_bbox[2] - media_bbox[0])
            pdf_page_height = float(media_bbox[3] - media_bbox[1])

            raw_cells = decoder.get_word_cells()
            cells: list[TextCell] = []
            for i in range(len(raw_cells)):
                cell = raw_cells[i]
                # docling-parse uses BOTTOMLEFT origin; convert to TOPLEFT.
                l = float(cell.x0)
                r = float(cell.x1)
                t = pdf_page_height - float(cell.y1)
                b = pdf_page_height - float(cell.y0)
                if l < r and t < b and cell.text.strip():
                    cells.append(
                        TextCell(
                            box=BoundingBox(l=l, t=t, r=r, b=b),
                            text=cell.text,
                        )
                    )

            # --- hyperlink parsing & merging ---
            # Each hyperlink covers a rectangular area (BOTTOMLEFT origin).
            # Collect all word cells whose centre falls inside a hyperlink bbox,
            # then merge them into a single TextCell carrying the URI.
        raw_hyperlinks = decoder.get_page_hyperlinks()
        hyperlinks: list[dict] = []
        for i in range(len(raw_hyperlinks)):
            h = raw_hyperlinks[i]
            hl_l = float(h.x0)
            hl_r = float(h.x1)
            hl_t = pdf_page_height - float(h.y1)
            hl_b = pdf_page_height - float(h.y0)
            uri: str | None = h.uri or None
            if hl_l < hl_r and hl_t < hl_b and uri:
                hyperlinks.append(
                    {"l": hl_l, "t": hl_t, "r": hl_r, "b": hl_b, "uri": uri}
                )

        if hyperlinks:
            merged_into_hyperlink: set[int] = set()
            merged_cells: list[TextCell] = []
            for hl in hyperlinks:
                group = [
                    idx
                    for idx, c in enumerate(cells)
                    if idx not in merged_into_hyperlink
                    and hl["l"] <= (c.box.l + c.box.r) / 2 <= hl["r"]
                    and hl["t"] <= (c.box.t + c.box.b) / 2 <= hl["b"]
                ]
                if not group:
                    continue
                merged_into_hyperlink.update(group)
                # Sort by reading order: top-to-bottom then left-to-right.
                grouped = sorted(
                    [cells[i] for i in group], key=lambda c: (c.box.t, c.box.l)
                )
                merged_cells.append(
                    TextCell(
                        box=BoundingBox(
                            l=min(c.box.l for c in grouped),
                            t=min(c.box.t for c in grouped),
                            r=max(c.box.r for c in grouped),
                            b=max(c.box.b for c in grouped),
                        ),
                        text=" ".join(c.text for c in grouped),
                        uri=hl["uri"],
                    )
                )
            # Preserve original relative order of non-merged cells.
            remaining = [
                c for i, c in enumerate(cells) if i not in merged_into_hyperlink
            ]
            cells = merged_cells + remaining
        # --- end hyperlink merging ---

        parser.unload_document(key)
        return cells, pdf_page_width, pdf_page_height

    except Exception:
        logger.opt(exception=True).debug(
            "docling-parse extraction failed for {file_path} page {page_idx}",
            file_path=str(file_path),
            page_idx=page_idx,
        )
        return None, None, None
        finally:
            if loaded:
                parser.unload_document(key)
