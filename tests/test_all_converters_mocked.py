"""
Test all converter configs with mocked OpenAI clients.
This avoids the need to deploy actual Docker servers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vlmparse.data_model.document import Document, Page
from vlmparse.registries import converter_config_registry

# Mock response for different model types
MOCK_RESPONSES = {
    "default": "# Test Document\n\nThis is a test page with some content.",
    "dotsocr_layout": '[{"bbox": [10, 10, 100, 50], "category": "Text", "text": "Test content"}]',
    "dotsocr_ocr": "Test content from DotsOCR",
}


@pytest.fixture
def mock_openai_client():
    """Mock the AsyncOpenAI client used by all converters."""
    with patch("openai.AsyncOpenAI") as mock_client:
        # Create mock response object
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = MOCK_RESPONSES["default"]

        # Configure the async method
        mock_instance = MagicMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def dotsocr_mock_client():
    """Mock for DotsOCR with different response types."""
    with patch("openai.AsyncOpenAI") as mock_client:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = MOCK_RESPONSES["dotsocr_ocr"]

        mock_instance = MagicMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_instance

        yield mock_instance


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
    def test_converter_basic_processing(
        self, file_path, model_name, mock_openai_client
    ):
        """Test basic document processing for OpenAI-compatible converters."""
        config = converter_config_registry.get(model_name)
        converter = config.get_client(num_concurrent_pages=2)

        # Process document
        document = converter(file_path)

        # Verify document structure
        assert isinstance(document, Document)
        assert document.file_path == str(file_path)
        assert len(document.pages) == 2, f"Expected 2 pages, got {len(document.pages)}"

        # Verify pages
        for page in document.pages:
            assert isinstance(page, Page)
            assert page.text is not None, "Page text should not be None"
            assert len(page.text) > 0, "Page text should not be empty"

        # Verify API was called
        assert mock_openai_client.chat.completions.create.call_count == 2

    def test_converter_image_processing(self, datadir, mock_openai_client):
        """Test processing of a single image file."""
        model_name = "gemini-2.5-flash-lite"
        image_path = datadir / "page_with_formula.png"

        config = converter_config_registry.get(model_name)
        converter = config.get_client()

        # Process image
        document = converter(image_path)

        # Verify document structure
        assert isinstance(document, Document)
        assert document.file_path == str(image_path)
        assert len(document.pages) == 1, f"Expected 1 page, got {len(document.pages)}"

        # Verify page
        page = document.pages[0]
        assert isinstance(page, Page)
        assert page.text is not None
        assert len(page.text) > 0

        # Verify API was called once
        assert mock_openai_client.chat.completions.create.call_count == 1

    def test_dotsocr_ocr_mode(self, file_path, dotsocr_mock_client):
        """Test DotsOCR converter in OCR mode."""
        config = converter_config_registry.get("dotsocr")
        converter = config.get_client(num_concurrent_pages=2)

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
        assert dotsocr_mock_client.chat.completions.create.call_count == 2

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_converter_error_handling(self, file_path, model_name):
        """Test that converters handle errors gracefully."""
        with patch("openai.AsyncOpenAI") as mock_client:
            # Configure mock to raise an exception
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_client.return_value = mock_instance

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
    def test_batch_processing(self, file_path, model_name, mock_openai_client):
        """Test batch processing of multiple files."""
        config = converter_config_registry.get(model_name)
        converter = config.get_client(
            num_concurrent_files=2,
            num_concurrent_pages=2,
            return_documents_in_batch_mode=True,
        )

        # Process multiple files (same file for testing)
        file_paths = [file_path, file_path]
        documents = converter.batch(file_paths)

        # Verify results
        assert len(documents) == 2
        for doc in documents:
            assert isinstance(doc, Document)
            assert len(doc.pages) == 2


class TestCustomURI:
    """Test converter initialization with custom URIs."""

    def test_custom_uri_config(self, mock_openai_client, file_path):
        """Test that converters can be initialized with custom URIs."""
        custom_uri = "http://localhost:8000/v1"
        config = converter_config_registry.get("gemini-2.5-flash-lite", uri=custom_uri)

        assert config.llm_params.base_url == custom_uri

        # Test it works
        converter = config.get_client()
        document = converter(file_path)

        assert isinstance(document, Document)
        assert len(document.pages) == 2


class TestConcurrency:
    """Test concurrent processing settings."""

    @pytest.mark.parametrize("model_name", ["gemini-2.5-flash-lite", "lightonocr"])
    def test_concurrent_page_processing(
        self, file_path, model_name, mock_openai_client
    ):
        """Test that concurrent page processing limits are respected."""
        config = converter_config_registry.get(model_name)
        converter = config.get_client(num_concurrent_pages=1)

        document = converter(file_path)

        assert len(document.pages) == 2
        # With concurrency=1, calls should be sequential
        assert mock_openai_client.chat.completions.create.call_count == 2
