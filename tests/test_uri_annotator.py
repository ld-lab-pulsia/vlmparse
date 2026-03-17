"""Tests for URI annotation on items and page text."""

from vlmparse.data_model.box import BoundingBox
from vlmparse.data_model.document import Item, Page, TextCell
from vlmparse.uri_annotator import (
    _build_uri_phrases,
    _find_fuzzy_match,
    annotate_item_with_uris,
    annotate_page_items_with_uris,
    annotate_page_text_with_uris,
)


def _box(l=0, t=0, r=100, b=50):
    return BoundingBox(l=l, t=t, r=r, b=b)


def _cell(text, uri, l=0, t=0, r=100, b=50):
    return TextCell(box=_box(l, t, r, b), text=text, uri=uri)


# ---------- _find_fuzzy_match ----------


class TestFindFuzzyMatch:
    def test_exact_match(self):
        assert _find_fuzzy_match("hello world", "world", 0.7) == (6, 11)

    def test_case_insensitive_exact(self):
        assert _find_fuzzy_match("Hello World", "hello", 0.7) == (0, 5)

    def test_no_match(self):
        assert _find_fuzzy_match("hello world", "zzzzz", 0.7) is None

    def test_fuzzy_match(self):
        result = _find_fuzzy_match("the documntation is here", "documentation", 0.6)
        assert result is not None
        start, end = result
        assert "documntation" in "the documntation is here"[start:end]

    def test_query_longer_than_text(self):
        assert _find_fuzzy_match("hi", "a very long query", 0.7) is None


# ---------- annotate_item_with_uris ----------


class TestAnnotateItemWithUris:
    def test_single_uri(self):
        item = Item(category="text", box=_box(), text="Visit the OpenAI docs now.")
        cells = [_cell("OpenAI docs", "https://openai.com")]
        result = annotate_item_with_uris(item, cells)
        assert result == "Visit the [OpenAI docs](https://openai.com) now."

    def test_no_overlap_skips(self):
        item = Item(
            category="text",
            box=_box(l=0, t=0, r=50, b=50),
            text="Visit the OpenAI docs now.",
        )
        cells = [_cell("OpenAI docs", "https://openai.com", l=500, t=500, r=600, b=550)]
        result = annotate_item_with_uris(item, cells)
        assert result == "Visit the OpenAI docs now."

    def test_multiple_uris(self):
        item = Item(
            category="text",
            box=_box(),
            text="See docs and examples for details.",
        )
        cells = [
            _cell("docs", "https://docs.example.com"),
            _cell("examples", "https://examples.example.com"),
        ]
        result = annotate_item_with_uris(item, cells)
        assert "[docs](https://docs.example.com)" in result
        assert "[examples](https://examples.example.com)" in result

    def test_empty_text_returns_as_is(self):
        item = Item(category="text", box=_box(), text="")
        assert annotate_item_with_uris(item, [_cell("x", "http://x")]) == ""

    def test_no_cells_returns_original(self):
        item = Item(category="text", box=_box(), text="hello")
        assert annotate_item_with_uris(item, []) == "hello"


# ---------- _build_uri_phrases ----------


class TestBuildUriPhrases:
    def test_groups_by_uri(self):
        cells = [
            _cell("click", "https://a.com", l=0, t=0, r=30, b=10),
            _cell("here", "https://a.com", l=35, t=0, r=60, b=10),
        ]
        phrases = _build_uri_phrases(cells)
        assert len(phrases) == 1
        assert phrases[0] == ("click here", "https://a.com")

    def test_multiple_uris(self):
        cells = [
            _cell("alpha", "https://a.com"),
            _cell("beta gamma", "https://b.com"),
        ]
        phrases = _build_uri_phrases(cells)
        assert len(phrases) == 2
        # Sorted longest first
        assert phrases[0][0] == "beta gamma"

    def test_empty_text_skipped(self):
        cells = [_cell("", "https://a.com"), _cell("word", "https://a.com")]
        phrases = _build_uri_phrases(cells)
        assert phrases[0][0] == "word"


# ---------- annotate_page_text_with_uris ----------


class TestAnnotatePageTextWithUris:
    def test_basic(self):
        page = Page(
            text="Read the full documentation for more info.",
            text_cells=[_cell("full documentation", "https://docs.example.com")],
        )
        annotate_page_text_with_uris(page)
        assert (
            page.text
            == "Read the [full documentation](https://docs.example.com) for more info."
        )

    def test_multiple_links(self):
        page = Page(
            text="See the guide and the FAQ page.",
            text_cells=[
                _cell("guide", "https://guide.com"),
                _cell("FAQ", "https://faq.com"),
            ],
        )
        annotate_page_text_with_uris(page)
        assert "[guide](https://guide.com)" in page.text
        assert "[FAQ](https://faq.com)" in page.text

    def test_no_text_noop(self):
        page = Page(text=None, text_cells=[_cell("x", "http://x")])
        annotate_page_text_with_uris(page)
        assert page.text is None

    def test_no_cells_noop(self):
        page = Page(text="hello", text_cells=None)
        annotate_page_text_with_uris(page)
        assert page.text == "hello"

    def test_no_uri_cells_noop(self):
        page = Page(
            text="hello",
            text_cells=[TextCell(box=_box(), text="hello", uri=None)],
        )
        annotate_page_text_with_uris(page)
        assert page.text == "hello"

    def test_phrase_built_from_multiple_cells(self):
        page = Page(
            text="Please click here to continue.",
            text_cells=[
                _cell("click", "https://link.com", l=0, t=0, r=30, b=10),
                _cell("here", "https://link.com", l=35, t=0, r=60, b=10),
            ],
        )
        annotate_page_text_with_uris(page)
        assert "[click here](https://link.com)" in page.text


# ---------- annotate_page_items_with_uris (integration) ----------


class TestAnnotatePageItemsWithUris:
    def test_annotates_both_items_and_text(self):
        page = Page(
            text="Visit the docs for help.",
            items=[Item(category="text", box=_box(), text="Visit the docs for help.")],
            text_cells=[_cell("docs", "https://docs.example.com")],
        )
        annotate_page_items_with_uris(page)
        assert "[docs](https://docs.example.com)" in page.items[0].text
        assert "[docs](https://docs.example.com)" in page.text

    def test_no_text_cells_noop(self):
        page = Page(
            text="hello",
            items=[Item(category="text", box=_box(), text="hello")],
            text_cells=None,
        )
        annotate_page_items_with_uris(page)
        assert page.text == "hello"
        assert page.items[0].text == "hello"

    def test_only_text_no_items(self):
        page = Page(
            text="Check the API reference.",
            items=None,
            text_cells=[_cell("API reference", "https://api.example.com")],
        )
        annotate_page_items_with_uris(page)
        assert "[API reference](https://api.example.com)" in page.text
