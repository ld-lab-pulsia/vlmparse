"""
Test CLI commands while mocking the server side.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from vlmparse.cli import DParseCLI
from vlmparse.data_model.document import Document, Page

# Mock pypdfium2 to return fake images
mock_pdfium = MagicMock()
mock_pdf = MagicMock()
# Create fake PIL images for the pages
fake_image = Image.new("RGB", (100, 100), color="white")
mock_pdf.__len__ = MagicMock(return_value=2)  # 2 pages
mock_pdf.__getitem__ = MagicMock(
    return_value=MagicMock(
        render=MagicMock(
            return_value=MagicMock(to_pil=MagicMock(return_value=fake_image))
        )
    )
)
mock_pdfium.PdfDocument = MagicMock(return_value=mock_pdf)
sys.modules["pypdfium2"] = mock_pdfium


@pytest.fixture
def cli():
    """Create a CLI instance for testing."""
    return DParseCLI()


@pytest.fixture
def mock_docker_server():
    """Mock Docker server for serve tests."""
    with patch("vlmparse.registries.docker_config_registry") as mock_registry:
        # Create mock Docker config
        mock_config = MagicMock()
        mock_server = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test_container_id"
        mock_container.name = "test_container_name"

        # Configure server.start() to return base_url and container
        mock_server.start.return_value = ("http://localhost:8056", mock_container)
        mock_config.get_server.return_value = mock_server
        mock_registry.get.return_value = mock_config

        yield mock_registry, mock_config, mock_server, mock_container


@pytest.fixture
def mock_converter_client():
    """Mock converter client for convert tests."""
    with patch("vlmparse.registries.converter_config_registry") as mock_registry:
        # Create mock converter (not client, CLI calls get_client)
        mock_converter = MagicMock()
        mock_config = MagicMock()

        # Batch now returns None by default (documents are saved to disk instead)
        mock_converter.batch.return_value = None
        mock_config.get_client.return_value = mock_converter
        mock_registry.get.return_value = mock_config

        yield mock_registry, mock_config, mock_converter


class TestServeCommand:
    """Test the 'serve' command."""

    def test_serve_default_port(self, cli, mock_docker_server):
        """Test serve command with default port."""
        mock_registry, mock_config, mock_server, mock_container = mock_docker_server

        cli.serve(model="lightonocr")

        # Verify registry was called with correct model
        mock_registry.get.assert_called_once_with("lightonocr")

        # Verify port was set to default
        assert mock_config.docker_port == 8056

        # # Verify gpu_device_ids was None
        # assert mock_config.gpu_device_ids is None

        # Verify server was created and started
        mock_config.get_server.assert_called_once_with(auto_stop=False)
        mock_server.start.assert_called_once()

    def test_serve_custom_port(self, cli, mock_docker_server):
        """Test serve command with custom port."""
        mock_registry, mock_config, mock_server, mock_container = mock_docker_server

        cli.serve(model="lightonocr", port=9000)

        # Verify custom port was set
        assert mock_config.docker_port == 9000
        mock_server.start.assert_called_once()

    def test_serve_with_gpus(self, cli, mock_docker_server):
        """Test serve command with GPU configuration."""
        mock_registry, mock_config, mock_server, mock_container = mock_docker_server

        cli.serve(model="lightonocr", port=8056, gpus="0,1,2")

        # Verify GPU device IDs were parsed correctly
        assert mock_config.gpu_device_ids == ["0", "1", "2"]
        mock_server.start.assert_called_once()

    def test_serve_single_gpu(self, cli, mock_docker_server):
        """Test serve command with single GPU."""
        mock_registry, mock_config, mock_server, mock_container = mock_docker_server

        cli.serve(model="lightonocr", gpus="0")

        # Verify single GPU was parsed correctly
        assert mock_config.gpu_device_ids == ["0"]

    def test_serve_unknown_model(self, cli):
        """Test serve command with unknown model (should warn and return)."""
        with patch("vlmparse.registries.docker_config_registry") as mock_registry:
            mock_registry.get.return_value = None

            # Should not raise an exception, just log warning
            cli.serve(model="unknown_model")

            mock_registry.get.assert_called_once_with("unknown_model")


class TestConvertCommand:
    """Test the 'convert' command."""

    def test_convert_single_file(self, cli, file_path, mock_converter_client):
        """Test convert with a single PDF file."""
        mock_registry, mock_config, mock_converter = mock_converter_client

        with patch("vlmparse.registries.docker_config_registry"):
            cli.convert(
                inputs=[str(file_path)],
                model="lightonocr",
                uri="http://localhost:8000/v1",
            )

        # Verify config was retrieved
        mock_registry.get.assert_called_once()

        # Verify batch was called with correct file
        mock_converter.batch.assert_called_once()
        call_args = mock_converter.batch.call_args[0][0]
        assert len(call_args) == 1
        assert str(file_path) in call_args

    def test_convert_multiple_files(self, cli, file_path, mock_converter_client):
        """Test convert with multiple PDF files."""
        mock_registry, mock_config, mock_converter = mock_converter_client

        with patch("vlmparse.registries.docker_config_registry"):
            cli.convert(
                inputs=[str(file_path), str(file_path)],
                model="lightonocr",
                uri="http://localhost:8000/v1",
            )

        # Verify batch was called with both files
        call_args = mock_converter.batch.call_args[0][0]
        assert len(call_args) == 2

    def test_convert_with_glob_pattern(self, cli, file_path, mock_converter_client):
        """Test convert with glob pattern."""
        mock_registry, mock_config, mock_converter = mock_converter_client

        # Use the parent directory with a glob pattern
        pattern = str(file_path.parent / "*.pdf")

        with patch("vlmparse.registries.docker_config_registry"):
            cli.convert(
                inputs=[pattern], model="lightonocr", uri="http://localhost:8000/v1"
            )

        # Verify at least one file was found
        call_args = mock_converter.batch.call_args[0][0]
        assert len(call_args) >= 1

    def test_convert_no_files_found(self, cli):
        """Test convert with no matching files."""
        mock_registry = MagicMock()

        with patch("vlmparse.registries.converter_config_registry", mock_registry):
            with patch("vlmparse.registries.docker_config_registry"):
                # Should return early without calling batch
                cli.convert(
                    inputs=["/nonexistent/*.pdf"],
                    model="lightonocr",
                    uri="http://localhost:8000/v1",
                )

    def test_convert_with_custom_uri(self, cli, file_path, mock_converter_client):
        """Test convert with custom URI (no Docker server needed)."""
        mock_registry, mock_config, mock_converter = mock_converter_client

        custom_uri = "http://custom-server:9000/v1"

        with patch("vlmparse.registries.docker_config_registry"):
            cli.convert(inputs=[str(file_path)], model="lightonocr", uri=custom_uri)

        # Verify registry was called with custom URI
        mock_registry.get.assert_called_once()
        _, kwargs = mock_registry.get.call_args
        assert kwargs.get("uri") == custom_uri

    def test_convert_without_uri_starts_server(self, cli, file_path):
        """Test convert without URI starts a Docker server."""
        with patch("vlmparse.registries.converter_config_registry"):
            with patch(
                "vlmparse.registries.docker_config_registry"
            ) as mock_docker_registry:
                # Setup mocks
                mock_docker_config = MagicMock()
                mock_server = MagicMock()
                mock_converter = MagicMock()
                mock_converter.batch.return_value = None

                mock_docker_config.get_server.return_value = mock_server
                mock_docker_config.get_client.return_value = mock_converter
                mock_docker_registry.get.return_value = mock_docker_config

                cli.convert(inputs=[str(file_path)], model="lightonocr")

                # Verify Docker server was started
                mock_docker_registry.get.assert_called_once_with(
                    "lightonocr", default=False
                )
                mock_docker_config.get_server.assert_called_once_with(auto_stop=True)
                mock_server.start.assert_called_once()
                mock_converter.batch.assert_called_once()

    def test_convert_with_gpus(self, cli, file_path):
        """Test convert with GPU configuration."""
        with patch("vlmparse.registries.converter_config_registry"):
            with patch(
                "vlmparse.registries.docker_config_registry"
            ) as mock_docker_registry:
                mock_docker_config = MagicMock()
                mock_server = MagicMock()
                mock_converter = MagicMock()
                mock_converter.batch.return_value = None

                mock_docker_config.get_server.return_value = mock_server
                mock_docker_config.get_client.return_value = mock_converter
                mock_docker_registry.get.return_value = mock_docker_config

                cli.convert(inputs=[str(file_path)], model="lightonocr", gpus="0,1")

                # Verify GPU device IDs were set
                assert mock_docker_config.gpu_device_ids == ["0", "1"]

    def test_convert_with_output_folder(self, cli, file_path, mock_converter_client):
        """Test convert with custom output folder."""
        mock_registry, mock_config, mock_converter = mock_converter_client

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("vlmparse.registries.docker_config_registry"):
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=tmpdir,
                    model="lightonocr",
                    uri="http://localhost:8000/v1",
                )

            # Just verify the command completes successfully
            mock_converter.batch.assert_called_once()

    def test_convert_string_inputs(self, cli, file_path, mock_converter_client):
        """Test convert with string inputs (not list)."""
        mock_registry, mock_config, mock_converter = mock_converter_client

        with patch("vlmparse.registries.docker_config_registry"):
            # Pass string instead of list
            cli.convert(
                inputs=str(file_path),
                model="lightonocr",
                uri="http://localhost:8000/v1",
            )

        # Should convert to list internally and process
        call_args = mock_converter.batch.call_args[0][0]
        assert len(call_args) == 1

    def test_convert_filters_non_pdf_files(self, cli, mock_converter_client):
        """Test that convert filters out non-PDF files."""
        mock_registry, mock_config, mock_converter = mock_converter_client

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-PDF file
            txt_file = Path(tmpdir) / "test.txt"
            txt_file.write_text("test")

            with patch("vlmparse.registries.docker_config_registry"):
                cli.convert(
                    inputs=[str(txt_file)],
                    model="lightonocr",
                    uri="http://localhost:8000/v1",
                )


class TestConvertWithDifferentModels:
    """Test convert command with different model types."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "gemini-2.5-flash-lite",
            "lightonocr",
            "dotsocr",
            "nanonets/Nanonets-OCR2-3B",
        ],
    )
    def test_convert_with_various_models(self, cli, file_path, model_name):
        """Test convert with different registered models."""
        with patch("vlmparse.registries.converter_config_registry") as mock_registry:
            with patch("vlmparse.registries.docker_config_registry"):
                mock_config = MagicMock()
                mock_converter = MagicMock()
                mock_converter.batch.return_value = None
                mock_config.get_client.return_value = mock_converter
                mock_registry.get.return_value = mock_config

                cli.convert(
                    inputs=[str(file_path)],
                    model=model_name,
                    uri="http://localhost:8000/v1",
                )

                # Verify correct model was requested
                call_args = mock_registry.get.call_args
                assert call_args[0][0] == model_name
                mock_converter.batch.assert_called_once()


class TestCLIIntegration:
    """Integration tests for CLI with mocked server."""

    def test_full_workflow_without_uri(self, cli, file_path):
        """Test full conversion workflow without providing URI."""
        with patch("vlmparse.registries.converter_config_registry"):
            with patch(
                "vlmparse.registries.docker_config_registry"
            ) as mock_docker_registry:
                # Setup complete mock chain
                mock_docker_config = MagicMock()
                mock_server = MagicMock()
                mock_converter = MagicMock()

                mock_converter.batch.return_value = None

                mock_docker_config.get_server.return_value = mock_server
                mock_docker_config.get_client.return_value = mock_converter
                mock_docker_registry.get.return_value = mock_docker_config

                # Run conversion
                cli.convert(inputs=[str(file_path)], model="lightonocr")

                # Verify full workflow
                mock_docker_registry.get.assert_called_once()
                mock_server.start.assert_called_once()
                mock_converter.batch.assert_called_once()

    def test_serve_then_convert_scenario(self, cli, file_path):
        """Test scenario where server is started first, then convert is called."""
        with patch(
            "vlmparse.registries.docker_config_registry"
        ) as mock_docker_registry:
            # Mock for serve
            mock_docker_config = MagicMock()
            mock_server = MagicMock()
            mock_container = MagicMock()
            mock_container.id = "test_id"
            mock_container.name = "test_name"
            mock_server.start.return_value = ("http://localhost:8056", mock_container)
            mock_docker_config.get_server.return_value = mock_server
            mock_docker_registry.get.return_value = mock_docker_config

            # First serve
            cli.serve(model="lightonocr", port=8056)

            # Verify serve worked
            mock_server.start.assert_called_once()

        # Then convert with URI pointing to the served model
        with patch(
            "vlmparse.registries.converter_config_registry"
        ) as mock_converter_registry:
            with patch("vlmparse.registries.docker_config_registry"):
                mock_config = MagicMock()
                mock_converter = MagicMock()
                mock_converter.batch.return_value = None
                mock_config.get_client.return_value = mock_converter
                mock_converter_registry.get.return_value = mock_config

                cli.convert(
                    inputs=[str(file_path)],
                    model="lightonocr",
                    uri="http://localhost:8056/v1",
                )

                # Verify convert used the URI
                mock_converter.batch.assert_called_once()


class TestCLIConvertInDepth:
    """In-depth tests for CLI convert with real converters, mocking only OpenAI API and server."""

    @pytest.fixture
    def mock_openai_api(self):
        """Mock the AsyncOpenAI client at API level."""
        # Patch at the openai module level where it's defined
        with patch("openai.AsyncOpenAI") as mock_client_class:
            # Create mock response object
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[
                0
            ].message.content = "# Test Document\n\nPage content here."

            # Configure the async method
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_client_class.return_value = mock_instance

            yield mock_instance

    @pytest.fixture
    def mock_pdf_to_images(self):
        """Mock PDF to image conversion."""
        from PIL import Image

        # Create fake PIL images for the pages
        fake_images = [Image.new("RGB", (100, 100), color="white") for _ in range(2)]

        with patch("vlmparse.converter.convert_specific_page_to_image") as mock_convert:
            mock_convert.return_value = fake_images[0]
            yield mock_convert

    @pytest.fixture
    def mock_docker_server_operations(self):
        """Mock Docker server operations without mocking converter logic."""
        with patch(
            "vlmparse.registries.docker_config_registry"
        ) as mock_docker_registry:
            mock_docker_config = MagicMock()
            mock_server = MagicMock()
            mock_container = MagicMock()
            mock_container.id = "test_container_id"
            mock_container.name = "test_container"

            mock_server.start.return_value = (
                "http://localhost:8000/v1",
                mock_container,
            )
            mock_docker_config.get_server.return_value = mock_server

            # Return None for models that shouldn't start a server
            def get_docker_config(model_name, default=False):
                if model_name.startswith("gemini"):
                    return None
                return mock_docker_config

            mock_docker_registry.get.side_effect = get_docker_config

            yield mock_docker_registry, mock_docker_config, mock_server

    def test_convert_with_real_converter_gemini(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test convert with real Gemini converter and mocked OpenAI API."""
        cli.convert(
            inputs=[str(file_path)],
            model="gemini-2.5-flash-lite",
            uri="http://mocked-api/v1",
        )

        # Verify OpenAI API was called (2 pages in test PDF)
        assert mock_openai_api.chat.completions.create.call_count == 2

        # Verify the model parameter was correct
        call_args = mock_openai_api.chat.completions.create.call_args_list[0]
        assert call_args[1]["model"] == "gemini-2.5-flash-lite"

    def test_convert_with_real_converter_lightonocr(
        self, cli, file_path, mock_openai_api, mock_docker_server_operations
    ):
        """Test convert with real LightOnOCR converter, auto-starting mocked server."""
        mock_docker_registry, mock_docker_config, mock_server = (
            mock_docker_server_operations
        )

        # Need to also mock get_client since we're auto-starting server
        mock_client = MagicMock()
        mock_doc = Document(file_path=str(file_path))
        mock_doc.pages = [Page(text="Page 1"), Page(text="Page 2")]
        mock_client.batch.return_value = [mock_doc]
        mock_docker_config.get_client.return_value = mock_client

        cli.convert(inputs=[str(file_path)], model="lightonocr")

        # Verify server was started
        mock_server.start.assert_called_once()

        # Verify client batch was called
        mock_client.batch.assert_called_once()

    def test_convert_batch_multiple_files(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test batch conversion of multiple files with real converter."""
        cli.convert(
            inputs=[str(file_path), str(file_path)],
            model="gemini-2.5-flash-lite",
            uri="http://mocked-api/v1",
        )

        # Should process 2 files × 2 pages = 4 API calls
        assert mock_openai_api.chat.completions.create.call_count == 4

    def test_convert_verifies_document_structure(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images, tmp_path
    ):
        """Test that converted documents have correct structure."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock successful response with detailed content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "# Page Title\n\nPage content with text."
        mock_openai_api.chat.completions.create = AsyncMock(return_value=mock_response)

        cli.convert(
            inputs=[str(file_path)],
            out_folder=str(output_dir),
            model="gemini-2.5-flash-lite",
            uri="http://mocked-api/v1",
        )

        # Verify conversion happened (2 pages)
        assert mock_openai_api.chat.completions.create.call_count == 2

    def test_convert_handles_api_errors_gracefully(
        self, cli, file_path, mock_pdf_to_images
    ):
        """Test that converter handles API errors without crashing."""
        with patch("openai.AsyncOpenAI") as mock_client_class:
            # Configure mock to raise an exception
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_client_class.return_value = mock_instance

            # Should not raise, but handle gracefully
            cli.convert(
                inputs=[str(file_path)],
                model="gemini-2.5-flash-lite",
                uri="http://mocked-api/v1",
            )

            # Verify it attempted to call API (2 pages)
            assert mock_instance.chat.completions.create.call_count == 2

    @pytest.mark.parametrize(
        "model_name",
        [
            "lightonocr",
            "gemini-2.5-flash-lite",
            "nanonets/Nanonets-OCR2-3B",
        ],
    )
    def test_convert_uses_correct_model_name(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images, model_name
    ):
        """Test that each converter uses the correct model name in API calls."""
        cli.convert(
            inputs=[str(file_path)], model=model_name, uri="http://mocked-api/v1"
        )

        # Check that model parameter is passed
        call_args = mock_openai_api.chat.completions.create.call_args_list[0]
        assert "model" in call_args[1]
        # Model name can be the original or derived from config
        assert call_args[1]["model"] in [
            model_name,
            "vllm-model",
            "lightonai/LightOnOCR-1B-1025",
            "nanonets/Nanonets-OCR2-3B",
        ]

    def test_convert_respects_concurrency(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test that concurrent processing is working."""
        import asyncio

        call_times = []

        async def track_call(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test content"
            return mock_response

        mock_openai_api.chat.completions.create = AsyncMock(side_effect=track_call)

        cli.convert(
            inputs=[str(file_path)],
            model="gemini-2.5-flash-lite",
            uri="http://mocked-api/v1",
        )

        # Verify calls were made
        assert len(call_times) == 2

    def test_convert_with_dotsocr_model(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test convert with DotsOCR which has different prompt modes."""
        cli.convert(
            inputs=[str(file_path)], model="dotsocr", uri="http://mocked-api/v1"
        )

        # Verify API was called (2 pages)
        assert mock_openai_api.chat.completions.create.call_count == 2

        # Check that messages were sent (DotsOCR uses specific prompt format)
        call_args = mock_openai_api.chat.completions.create.call_args_list[0]
        assert "messages" in call_args[1]

    def test_convert_image_preprocessing(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test that image preprocessing happens correctly."""
        messages_sent = []

        async def capture_messages(*args, **kwargs):
            messages_sent.append(kwargs.get("messages", []))
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            return mock_response

        mock_openai_api.chat.completions.create = AsyncMock(
            side_effect=capture_messages
        )

        cli.convert(
            inputs=[str(file_path)],
            model="gemini-2.5-flash-lite",
            uri="http://mocked-api/v1",
        )

        # Verify images were sent in messages
        assert len(messages_sent) == 2
        for messages in messages_sent:
            assert len(messages) > 0
            # Check that image data is present
            assert any("image_url" in str(msg) for msg in messages)

    def test_convert_with_different_dpi_settings(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test that different models use their configured DPI settings."""
        # LightOnOCR uses DPI=200
        cli.convert(
            inputs=[str(file_path)], model="lightonocr", uri="http://mocked-api/v1"
        )

        lightonocr_calls = mock_openai_api.chat.completions.create.call_count
        assert lightonocr_calls == 2

        mock_openai_api.reset_mock()

        # Gemini uses DPI=175 (default)
        cli.convert(
            inputs=[str(file_path)],
            model="gemini-2.5-flash-lite",
            uri="http://mocked-api/v1",
        )

        gemini_calls = mock_openai_api.chat.completions.create.call_count
        assert gemini_calls == 2

    def test_convert_with_max_image_size_limit(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test that max_image_size limit is respected for models that have it."""
        # LightOnOCR has max_image_size=1540
        cli.convert(
            inputs=[str(file_path)], model="lightonocr", uri="http://mocked-api/v1"
        )

        assert mock_openai_api.chat.completions.create.call_count == 2

        mock_openai_api.reset_mock()

        # Nanonets has no max_image_size limit
        cli.convert(
            inputs=[str(file_path)],
            model="nanonets/Nanonets-OCR2-3B",
            uri="http://mocked-api/v1",
        )

        assert mock_openai_api.chat.completions.create.call_count == 2

    def test_convert_with_glob_pattern_real_converter(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test glob pattern expansion with real converter."""
        pattern = str(file_path.parent / "*.pdf")

        cli.convert(
            inputs=[pattern], model="gemini-2.5-flash-lite", uri="http://mocked-api/v1"
        )

        # At least one file should be found and processed
        assert mock_openai_api.chat.completions.create.call_count >= 2

    def test_convert_checks_completion_kwargs(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test that converter processes pages correctly."""
        cli.convert(
            inputs=[str(file_path)], model="lightonocr", uri="http://mocked-api/v1"
        )

        # Check that API was called (2 pages)
        assert mock_openai_api.chat.completions.create.call_count == 2

        # Verify messages were sent to API
        call_args = mock_openai_api.chat.completions.create.call_args_list[0]
        assert "messages" in call_args[1]
        assert len(call_args[1]["messages"]) > 0

    def test_convert_processes_all_pages(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test that all pages in PDF are processed."""
        page_contents = []

        async def capture_page(*args, **kwargs):
            page_contents.append(f"Page {len(page_contents) + 1} content")
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = page_contents[-1]
            return mock_response

        mock_openai_api.chat.completions.create = AsyncMock(side_effect=capture_page)

        cli.convert(
            inputs=[str(file_path)],
            model="gemini-2.5-flash-lite",
            uri="http://mocked-api/v1",
        )

        # Test PDF has 2 pages
        assert len(page_contents) == 2
        assert "Page 1" in page_contents[0]
        assert "Page 2" in page_contents[1]

    def test_convert_with_nanonet_postprompt(
        self, cli, file_path, mock_openai_api, mock_pdf_to_images
    ):
        """Test that Nanonet model uses its postprompt correctly."""
        from vlmparse.registries import converter_config_registry

        messages_sent = []

        async def capture_messages(*args, **kwargs):
            messages_sent.append(kwargs.get("messages", []))
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Extracted text"
            return mock_response

        mock_openai_api.chat.completions.create = AsyncMock(
            side_effect=capture_messages
        )

        cli.convert(
            inputs=[str(file_path)],
            model="nanonets/Nanonets-OCR2-3B",
            uri="http://mocked-api/v1",
        )

        # Verify messages were sent (2 pages)
        assert len(messages_sent) == 2

        # Verify config has postprompt set
        config = converter_config_registry.get(
            "nanonets/Nanonets-OCR2-3B", uri="http://mocked-api/v1"
        )
        assert config.postprompt is not None
        assert len(config.postprompt) > 0
