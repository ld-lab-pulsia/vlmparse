"""
Test all converter configs with mocked OpenAI clients.
This avoids the need to deploy actual Docker servers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from vlmparse.data_model.document import Document, Page
from vlmparse.registries import converter_config_registry

# Mock response for different model types
MOCK_RESPONSES = {
    "default": "# Test Document\n\nThis is a test page with some content.",
    "dotsocr_layout": '[{"bbox": [10, 10, 100, 50], "category": "Text", "text": "Test content"}]',
    "dotsocr_ocr": "Test content from DotsOCR",
}


# Note: mock_openai_client and dotsocr_mock_client fixtures are replaced by the
# unified mock_openai_api fixture from conftest.py


# List of all models registered in converter_config_registry
ALL_MODELS = [
    "gemini-2.5-flash-lite",
    "lightonocr",
    "dotsocr",
    "nanonets/Nanonets-OCR2-3B",
]


class TestConverterConfigs:
    """Test suite for all converter configs."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_config_retrieval(self, model_name):
        """Test that all registered models can be retrieved from registry."""
        config = converter_config_registry.get(model_name)
        assert config is not None, f"Config for {model_name} should not be None"

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_config_has_get_client(self, model_name):
        """Test that all configs have get_client method."""
        config = converter_config_registry.get(model_name)
        assert hasattr(config, "get_client"), f"{model_name} config missing get_client"

    @pytest.mark.parametrize(
        "model_name",
        [
            "gemini-2.5-flash-lite",
            "lightonocr",
            "nanonets/Nanonets-OCR2-3B",
        ],
    )
    def test_converter_basic_processing(self, file_path, model_name, mock_openai_api):
        """Test basic document processing for OpenAI-compatible converters."""
        with mock_openai_api() as openai_client:
            config = converter_config_registry.get(model_name)
            converter = config.get_client(num_concurrent_pages=2, debug=True)

            # Process document
            document = converter(file_path)

            # Verify document structure
            assert isinstance(document, Document)
            assert document.file_path == str(file_path)
            assert (
                len(document.pages) == 2
            ), f"Expected 2 pages, got {len(document.pages)}"

            # Verify pages
            for page in document.pages:
                assert isinstance(page, Page)
                assert page.text is not None, "Page text should not be None"
                assert len(page.text) > 0, "Page text should not be empty"

            # Verify API was called
            assert openai_client.chat.completions.create.call_count == 2

    def test_converter_image_processing(self, datadir, mock_openai_api):
        """Test processing of a single image file."""
        with mock_openai_api() as openai_client:
            model_name = "gemini-2.5-flash-lite"
            image_path = datadir / "page_with_formula.png"

            config = converter_config_registry.get(model_name)
            converter = config.get_client(debug=True)

            # Process image
            document = converter(image_path)

            # Verify document structure
            assert isinstance(document, Document)
            assert document.file_path == str(image_path)
            assert (
                len(document.pages) == 1
            ), f"Expected 1 page, got {len(document.pages)}"

            # Verify page
            page = document.pages[0]
            assert isinstance(page, Page)
            assert page.text is not None
            assert len(page.text) > 0

            # Verify API was called once
            assert openai_client.chat.completions.create.call_count == 1

    def test_dotsocr_ocr_mode(self, file_path, mock_openai_api):
        """Test DotsOCR converter in OCR mode."""
        with mock_openai_api(content=MOCK_RESPONSES["dotsocr_ocr"]) as openai_client:
            config = converter_config_registry.get("dotsocr")
            converter = config.get_client(num_concurrent_pages=2, debug=True)

            # Process document
            document = converter(file_path)

            # Verify document structure
            assert isinstance(document, Document)
            assert len(document.pages) == 2

            for page in document.pages:
                assert isinstance(page, Page)
                assert page.text is not None
                assert len(page.text) > 0

            # Verify API was called
            assert openai_client.chat.completions.create.call_count == 2

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_converter_error_handling(self, file_path, model_name, mock_openai_api):
        """Test that converters handle errors gracefully."""
        with mock_openai_api(side_effect=Exception("API Error")):
            config = converter_config_registry.get(model_name)
            converter = config.get_client(debug=False)

            # Process should not crash
            document = converter(file_path)

            # Document should have error info in pages
            assert isinstance(document, Document)
            # Check that pages have errors
            for page in document.pages:
                assert page.error is not None


class TestConverterBatchProcessing:
    """Test batch processing capabilities."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "gemini-2.5-flash-lite",
            "lightonocr",
        ],
    )
    def test_batch_processing(self, file_path, model_name, mock_openai_api):
        """Test batch processing of multiple files."""
        with mock_openai_api():
            config = converter_config_registry.get(model_name)
            converter = config.get_client(
                num_concurrent_files=2,
                num_concurrent_pages=2,
                return_documents_in_batch_mode=True,
                debug=True,
            )

            # Process multiple files (same file for testing)
            file_paths = [file_path, file_path]
            documents = converter.batch(file_paths)

            # Verify results
            assert len(documents) == 2
            for doc in documents:
                assert isinstance(doc, Document)
                assert len(doc.pages) == 2


@pytest.fixture
def mineru_mock_httpx_client():
    """Mock the httpx AsyncClient used by MinerUConverter."""
    with patch("httpx.AsyncClient") as mock_async_client:
        mock_client = MagicMock()
        mock_async_client.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = orjson.dumps(
            [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "content": "<p>Hello MinerU</p>",
                    "type": "Text",
                },
                {
                    "bbox": [0.5, 0.6, 0.7, 0.8],
                    "content": "<p>Second block</p>",
                    "type": "Text",
                },
            ]
        )

        mock_client.post = AsyncMock(return_value=mock_response)
        yield mock_client


class TestMinerUConverterMockedApi:
    def test_mineru_converter_repeated_call(self, file_path, mineru_mock_httpx_client):
        """Repeated `__call__` should keep working and call API each page."""
        from vlmparse.clients.mineru import MinerUConverterConfig

        config = MinerUConverterConfig(base_url="http://mineru.test")
        converter = config.get_client(num_concurrent_pages=2, debug=True)

        with (
            patch("vlmparse.clients.mineru.clean_response", lambda x: x),
            patch("vlmparse.clients.mineru.html_to_md_keep_tables", lambda x: x),
        ):
            doc1 = converter(file_path)
            doc2 = converter(file_path)

        assert isinstance(doc1, Document)
        assert isinstance(doc2, Document)
        assert len(doc1.pages) == 2
        assert len(doc2.pages) == 2

        for page in doc1.pages + doc2.pages:
            assert isinstance(page, Page)
            assert page.text is not None and len(page.text) > 0
            assert page.items is not None
            assert len(page.items) == 2

        # 2 pages per doc * 2 docs
        assert mineru_mock_httpx_client.post.call_count == 4

    def test_mineru_converter_batch_processing(
        self, file_path, mineru_mock_httpx_client
    ):
        """Batch mode should return documents and call API for each page."""
        from vlmparse.clients.mineru import MinerUConverterConfig

        config = MinerUConverterConfig(base_url="http://mineru.test")
        converter = config.get_client(
            num_concurrent_files=2,
            num_concurrent_pages=2,
            return_documents_in_batch_mode=True,
            debug=True,
        )

        with (
            patch("vlmparse.clients.mineru.clean_response", lambda x: x),
            patch("vlmparse.clients.mineru.html_to_md_keep_tables", lambda x: x),
        ):
            docs = converter.batch([file_path, file_path])

        assert isinstance(docs, list)
        assert len(docs) == 2
        for doc in docs:
            assert isinstance(doc, Document)
            assert len(doc.pages) == 2

        # 2 pages per doc * 2 docs
        assert mineru_mock_httpx_client.post.call_count == 4


class TestCustomURI:
    """Test converter initialization with custom URIs."""

    def test_custom_uri_config(self, mock_openai_api, file_path):
        """Test that converters can be initialized with custom URIs."""
        with mock_openai_api():
            custom_uri = "http://localhost:8000/v1"
            config = converter_config_registry.get(
                "gemini-2.5-flash-lite", uri=custom_uri
            )

            assert config.llm_params.base_url == custom_uri

            # Test it works
            converter = config.get_client(debug=True)
            document = converter(file_path)

            assert isinstance(document, Document)
            assert len(document.pages) == 2


class TestConcurrency:
    """Test concurrent processing settings."""

    @pytest.mark.parametrize("model_name", ["gemini-2.5-flash-lite", "lightonocr"])
    def test_concurrent_page_processing(self, file_path, model_name, mock_openai_api):
        """Test that concurrent page processing limits are respected."""
        with mock_openai_api() as openai_client:
            config = converter_config_registry.get(model_name)
            converter = config.get_client(num_concurrent_pages=1, debug=True)

            document = converter(file_path)

            assert len(document.pages) == 2
            # With concurrency=1, calls should be sequential
            assert openai_client.chat.completions.create.call_count == 2
