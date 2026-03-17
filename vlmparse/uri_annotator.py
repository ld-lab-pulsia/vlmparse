"""Annotate Item text with markdown URI links from native text cells."""

import difflib
import urllib.parse

from .data_model.document import Item, Page, TextCell


def _find_fuzzy_match(
    text: str, query: str, threshold: float
) -> tuple[int, int] | None:
    """Find the best fuzzy match position of *query* inside *text*.

    Tries an exact case-insensitive substring first; falls back to a sliding-
    window SequenceMatcher scan.  Returns ``(start, end)`` character indices
    into *text*, or ``None`` if no match meets *threshold*.
    """
    lower_text = text.lower()
    lower_query = query.lower()

    # Fast path: exact substring
    idx = lower_text.find(lower_query)
    if idx >= 0:
        return idx, idx + len(query)

    qlen = len(query)
    if qlen > len(text):
        return None

    best_ratio = threshold
    best_pos: tuple[int, int] | None = None

    for i in range(len(text) - qlen + 1):
        window = text[i : i + qlen]
        ratio = difflib.SequenceMatcher(
            None, window.lower(), lower_query, autojunk=False
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = (i, i + qlen)

    return best_pos


def _escape_uri_for_markdown(uri: str) -> str:
    """Return *uri* percent-encoded for safe use in markdown link destinations.

    Escapes characters like spaces and ``)`` that would otherwise terminate
    the markdown link, while preserving common URL punctuation.
    """
    # Do not escape common URL characters, but ensure ')' and spaces are encoded.
    return urllib.parse.quote(uri, safe=":/?#[]@!$&'(*+,;=%")


def annotate_item_with_uris(
    item: Item,
    uri_cells: list[TextCell],
    overlap_threshold: float = 0.3,
    fuzzy_threshold: float = 0.7,
) -> str:
    """Return ``item.text`` with markdown URI annotations.

    For each *TextCell* (with a URI) in *uri_cells* whose bounding box
    overlaps sufficiently with *item.box*, the best fuzzy match of the
    cell's text inside the item text is wrapped as ``[match](uri)``.

    Multiple non-overlapping matches are all applied in a single pass so
    replacements never interfere with each other.

    Args:
        item: The layout item whose text should be annotated.
        uri_cells: TextCells that carry a non-None URI (pre-filtered for
            efficiency when processing many items on the same page).
        overlap_threshold: Minimum ``intersection_over_self_area`` of the
            TextCell box vs the Item box to be considered a candidate.
        fuzzy_threshold: Minimum SequenceMatcher ratio for a fuzzy match.

    Returns:
        Annotated text string (original text if nothing matched).
    """
    if not item.text or not uri_cells:
        return item.text

    # 1. Find cells whose box overlaps with the item box
    candidates: list[TextCell] = []
    for cell in uri_cells:
        try:
            overlap = cell.box.intersection_over_self_area(item.box)
        except ZeroDivisionError:
            continue
        if overlap >= overlap_threshold:
            candidates.append(cell)

    if not candidates:
        return item.text

    # 2. Sort by text length descending so longer phrases get priority
    candidates.sort(key=lambda c: len(c.text), reverse=True)

    text = item.text

    # 3. Collect non-overlapping matches on the *original* text
    matches: list[tuple[int, int, str]] = []  # (start, end, uri)
    for cell in candidates:
        query = cell.text.strip()
        if not query:
            continue
        pos = _find_fuzzy_match(text, query, fuzzy_threshold)
        if pos is None:
            continue
        start, end = pos
        # Skip if this range overlaps with an already-recorded match
        if any(start < me and end > ms for ms, me, _ in matches) or cell.uri is None:
            continue
        matches.append((start, end, cell.uri))

    if not matches:
        return text

    # 4. Build the annotated string in a single left-to-right pass
    matches.sort(key=lambda x: x[0])
    parts: list[str] = []
    prev = 0
    for start, end, uri in matches:
        parts.append(text[prev:start])
        safe_uri = _escape_uri_for_markdown(uri)
        parts.append(f"[{text[start:end]}]({safe_uri})")
        prev = end
    parts.append(text[prev:])

    return "".join(parts)


def annotate_page_items_with_uris(
    page: Page,
    overlap_threshold: float = 0.3,
    fuzzy_threshold: float = 0.7,
) -> None:
    """Annotate all items on *page* with URI markdown links, in-place.

    Requires ``page.items`` and ``page.text_cells`` to be populated.
    TextCells without a URI are ignored.
    """
    if not page.items or not page.text_cells:
        return

    uri_cells = [c for c in page.text_cells if c.uri]
    if not uri_cells:
        return

    for item in page.items:
        item.text = annotate_item_with_uris(
            item, uri_cells, overlap_threshold, fuzzy_threshold
        )
