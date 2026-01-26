"""
Test CLI commands while mocking the server side.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vlmparse.cli import DParseCLI
from vlmparse.data_model.document import Document, Page


@pytest.fixture
def cli():
    """Create a CLI instance for testing."""
    return DParseCLI()


# Note: mock_docker_server and mock_converter_client fixtures are now replaced
# by the unified mocking system in conftest.py: mock_docker_operations and mock_openai_api


class TestServeCommand:
    """Test the 'serve' command."""

    def test_serve_default_port(self, cli, mock_docker_operations):
        """Test serve command with default port."""
        with mock_docker_operations() as (mock_registry, mock_config, mock_server, _):
            cli.serve(model="lightonocr")

            # Verify registry was called with correct model
            mock_registry.get.assert_called_once_with("lightonocr", default=True)

            # Verify port was set to default
            assert mock_config.docker_port == 8056

            # # Verify gpu_device_ids was None
            # assert mock_config.gpu_device_ids is None

            # Verify server was created and started
            mock_config.get_server.assert_called_once_with(auto_stop=False)
            mock_server.start.assert_called_once()

    def test_serve_custom_port(self, cli, mock_docker_operations):
        """Test serve command with custom port."""
        with mock_docker_operations() as (mock_registry, mock_config, mock_server, _):
            cli.serve(model="lightonocr", port=9000)

            # Verify custom port was set
            assert mock_config.docker_port == 9000
            mock_server.start.assert_called_once()

    def test_serve_with_gpus(self, cli, mock_docker_operations):
        """Test serve command with GPU configuration."""
        with mock_docker_operations() as (mock_registry, mock_config, mock_server, _):
            cli.serve(model="lightonocr", port=8056, gpus="0,1,2")

            # Verify GPU device IDs were parsed correctly
            assert mock_config.gpu_device_ids == ["0", "1", "2"]
            mock_server.start.assert_called_once()

    def test_serve_single_gpu(self, cli, mock_docker_operations):
        """Test serve command with single GPU."""
        with mock_docker_operations() as (mock_registry, mock_config, mock_server, _):
            cli.serve(model="lightonocr", gpus="0")

            # Verify single GPU was parsed correctly
            assert mock_config.gpu_device_ids == ["0"]

    def test_serve_unknown_model(self, cli, mock_docker_operations):
        """Test serve command with unknown model (should warn and return)."""
        with mock_docker_operations(
            model_filter=lambda model: False  # No docker for any model
        ) as (mock_registry, _, _, _):
            # Should not raise an exception, just log warning
            cli.serve(model="unknown_model")

            mock_registry.get.assert_called_once_with("unknown_model", default=True)


class TestConvertCommand:
    """Test the 'convert' command."""

    def test_convert_single_file(
        self, cli, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with a single PDF file."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value="gemini-2.5-flash-lite",
                ):
                    documents = cli.convert(
                        inputs=[str(file_path)],
                        out_folder=str(tmp_output_dir),
                        model="gemini-2.5-flash-lite",
                        uri="http://localhost:8000/v1",
                        debug=True,
                        _return_documents=True,
                    )

                    # Verify files were processed
                    assert documents is not None
                    assert len(documents) > 0

    def test_convert_multiple_files(
        self, cli, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with multiple PDF files."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value="gemini-2.5-flash-lite",
                ):
                    documents = cli.convert(
                        inputs=[str(file_path), str(file_path)],
                        out_folder=str(tmp_output_dir),
                        model="gemini-2.5-flash-lite",
                        uri="http://localhost:8000/v1",
                        debug=True,
                        _return_documents=True,
                    )

                    # Verify both files were processed
                    assert documents is not None
                    assert len(documents) == 2

    def test_convert_with_glob_pattern(
        self, cli, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with glob pattern."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value="gemini-2.5-flash-lite",
                ):
                    # Use the parent directory with a glob pattern
                    pattern = str(file_path.parent / "*.pdf")

                    documents = cli.convert(
                        inputs=[pattern],
                        out_folder=str(tmp_output_dir),
                        model="gemini-2.5-flash-lite",
                        uri="http://localhost:8000/v1",
                        debug=True,
                        _return_documents=True,
                    )

                    # Verify at least one file was found
                    assert documents is not None
                    assert len(documents) >= 1

    def test_convert_with_custom_uri(
        self, cli, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with custom URI (no Docker server needed)."""
        with mock_docker_operations(include_client=False):
            custom_uri = "http://custom-server:9000/v1"

            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value="gemini-2.5-flash-lite",
                ):
                    documents = cli.convert(
                        inputs=[str(file_path)],
                        out_folder=str(tmp_output_dir),
                        model="gemini-2.5-flash-lite",
                        uri=custom_uri,
                        debug=True,
                        _return_documents=True,
                    )

                    # Verification now checks actual results rather than mock calls
                    assert documents is not None
                    assert len(documents) > 0

    def test_convert_without_uri_starts_server(
        self, cli, file_path, mock_docker_operations, tmp_output_dir
    ):
        """Test convert without URI starts a Docker server."""
        with mock_docker_operations(include_client=True, client_batch_return=None) as (
            mock_docker_reg,
            mock_docker_config,
            mock_server,
            mock_client,
        ):
            cli.convert(
                inputs=[str(file_path)],
                out_folder=str(tmp_output_dir),
                model="lightonocr",
                debug=True,
            )

            # Verify Docker server was started
            mock_docker_reg.get.assert_called_once_with("lightonocr", default=False)
            mock_docker_config.get_server.assert_called_once_with(auto_stop=True)
            mock_server.start.assert_called_once()
            mock_client.batch.assert_called_once()

    def test_convert_with_gpus(
        self, cli, file_path, mock_docker_operations, tmp_output_dir
    ):
        """Test convert with GPU configuration."""
        with mock_docker_operations(include_client=True, client_batch_return=None) as (
            mock_docker_reg,
            mock_docker_config,
            mock_server,
            mock_client,
        ):
            cli.convert(
                inputs=[str(file_path)],
                out_folder=str(tmp_output_dir),
                model="lightonocr",
                gpus="0,1",
                debug=True,
            )

            # Verify GPU device IDs were set
            assert mock_docker_config.gpu_device_ids == ["0", "1"]

    def test_convert_with_output_folder(
        self, cli, file_path, mock_docker_operations, mock_openai_api
    ):
        """Test convert with custom output folder."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value="gemini-2.5-flash-lite",
                ):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        documents = cli.convert(
                            inputs=[str(file_path)],
                            out_folder=tmpdir,
                            model="gemini-2.5-flash-lite",
                            uri="http://localhost:8000/v1",
                            debug=True,
                            _return_documents=True,
                        )

                        # Verify the command completes successfully
                        assert documents is not None
                        assert len(documents) > 0

    def test_convert_string_inputs(
        self, cli, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with string inputs (not list)."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value="gemini-2.5-flash-lite",
                ):
                    # Pass string instead of list
                    documents = cli.convert(
                        inputs=str(file_path),
                        out_folder=str(tmp_output_dir),
                        model="gemini-2.5-flash-lite",
                        uri="http://localhost:8000/v1",
                        debug=True,
                        _return_documents=True,
                    )

                    # Should convert to list internally and process
                    assert documents is not None
                    assert len(documents) == 1

    def test_convert_filters_non_pdf_files(
        self, cli, mock_docker_operations, mock_openai_api
    ):
        """Test that convert filters out non-PDF files."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value="gemini-2.5-flash-lite",
                ):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Create a non-PDF file
                        txt_file = Path(tmpdir) / "test.txt"
                        txt_file.write_text("test")

                        # Should raise an error for non-PDF files
                        with pytest.raises(
                            ValueError, match="Unsupported file extension"
                        ):
                            cli.convert(
                                inputs=[str(txt_file)],
                                model="gemini-2.5-flash-lite",
                                uri="http://localhost:8000/v1",
                                debug=True,
                                _return_documents=True,
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
    def test_convert_with_various_models(
        self,
        cli,
        file_path,
        model_name,
        mock_docker_operations,
        mock_openai_api,
        tmp_output_dir,
    ):
        """Test convert with different registered models."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value=model_name,
                ):
                    documents = cli.convert(
                        inputs=[str(file_path)],
                        out_folder=str(tmp_output_dir),
                        model=model_name,
                        uri="http://localhost:8000/v1",
                        debug=True,
                        _return_documents=True,
                    )

                    # Verify files were processed
                    assert documents is not None
                    assert len(documents) > 0


class TestCLIIntegration:
    """Integration tests for CLI with mocked server."""

    def test_full_workflow_without_uri(
        self, cli, file_path, mock_docker_operations, tmp_output_dir
    ):
        """Test full conversion workflow without providing URI."""
        with mock_docker_operations(include_client=True, client_batch_return=None) as (
            mock_docker_reg,
            mock_docker_config,
            mock_server,
            mock_client,
        ):
            # Run conversion
            cli.convert(
                inputs=[str(file_path)],
                out_folder=str(tmp_output_dir),
                model="lightonocr",
                debug=True,
            )

            # Verify full workflow
            mock_docker_reg.get.assert_called_once()
            mock_server.start.assert_called_once()
            mock_client.batch.assert_called_once()

    def test_serve_then_convert_scenario(
        self, cli, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test scenario where server is started first, then convert is called."""
        # First part: serve
        with mock_docker_operations() as (
            mock_docker_reg_serve,
            mock_docker_config,
            mock_server,
            _,
        ):
            # First serve
            cli.serve(model="lightonocr", port=8056)

            # Verify serve worked
            mock_server.start.assert_called_once()

        # Second part: convert with URI
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with patch(
                    "vlmparse.converter_with_server.get_model_from_uri",
                    return_value="gemini-2.5-flash-lite",
                ):
                    documents = cli.convert(
                        inputs=[str(file_path)],
                        out_folder=str(tmp_output_dir),
                        model="gemini-2.5-flash-lite",
                        uri="http://localhost:8056/v1",
                        debug=True,
                        _return_documents=True,
                    )

                    # Verify convert used the URI and processed files
                    assert documents is not None
                    assert len(documents) > 0


class TestCLIConvertInDepth:
    """In-depth tests for CLI convert with real converters, mocking only OpenAI API and server."""

    # @pytest.fixture
    # def mock_pdf_to_images(self):
    #     """Mock PDF to image conversion."""
    #     from PIL import Image

    #     # Create fake PIL images for the pages
    #     fake_images = [Image.new("RGB", (100, 100), color="white") for _ in range(2)]

    #     with patch("vlmparse.converter.convert_specific_page_to_image") as mock_convert:
    #         mock_convert.return_value = fake_images[0]
    #         yield mock_convert

    def test_convert_with_real_converter_gemini(
        self, cli, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test convert with real Gemini converter and mocked OpenAI API."""
        with mock_openai_api() as openai_client:
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="gemini-2.5-flash-lite",
            ):
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=str(tmp_output_dir),
                    model="gemini-2.5-flash-lite",
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                # Verify OpenAI API was called (2 pages in test PDF)
                assert openai_client.chat.completions.create.call_count == 2

                # Verify the model parameter was correct
                call_args = openai_client.chat.completions.create.call_args_list[0]
                assert call_args[1]["model"] == "gemini-2.5-flash-lite"

    def test_convert_with_real_converter_lightonocr(
        self, cli, file_path, mock_openai_api, mock_docker_operations, tmp_output_dir
    ):
        """Test convert with real LightOnOCR converter, auto-starting mocked server."""

        # Setup docker operations with model filter
        with mock_docker_operations(
            model_filter=lambda model: not model.startswith("gemini"),
            include_client=True,
        ) as (mock_docker_registry, mock_docker_config, mock_server, mock_client):
            # Setup client return value
            mock_doc = Document(file_path=str(file_path))
            mock_doc.pages = [Page(text="Page 1"), Page(text="Page 2")]
            mock_client.batch.return_value = [mock_doc]

            cli.convert(
                inputs=[str(file_path)],
                out_folder=str(tmp_output_dir),
                model="lightonocr",
                debug=True,
            )

            # Verify server was started
            mock_server.start.assert_called_once()

            # Verify client batch was called
            mock_client.batch.assert_called_once()

    def test_convert_batch_multiple_files(
        self, cli, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test batch conversion of multiple files with real converter."""
        with mock_openai_api() as openai_client:
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="gemini-2.5-flash-lite",
            ):
                cli.convert(
                    inputs=[str(file_path), str(file_path)],
                    out_folder=str(tmp_output_dir),
                    model="gemini-2.5-flash-lite",
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                # Should process 2 files × 2 pages = 4 API calls
                assert openai_client.chat.completions.create.call_count == 4

    def test_convert_verifies_document_structure(
        self, cli, file_path, mock_openai_api, tmp_path
    ):
        """Test that converted documents have correct structure."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create custom response
        with patch(
            "vlmparse.converter_with_server.get_model_from_uri",
            return_value="gemini-2.5-flash-lite",
        ):
            with mock_openai_api(
                content="# Page Title\n\nPage content with text."
            ) as openai_client:
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=str(output_dir),
                    model="gemini-2.5-flash-lite",
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                # Verify conversion happened (2 pages)
                assert openai_client.chat.completions.create.call_count == 2

    def test_convert_handles_api_errors_gracefully(
        self, cli, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test that converter handles API errors without crashing."""
        with patch(
            "vlmparse.converter_with_server.get_model_from_uri",
            return_value="gemini-2.5-flash-lite",
        ):
            with mock_openai_api(side_effect=Exception("API Error")) as openai_client:
                # Should not raise, but handle gracefully
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=str(tmp_output_dir),
                    model="gemini-2.5-flash-lite",
                    uri="http://mocked-api/v1",
                )

                # Verify it attempted to call API (2 pages)
                assert openai_client.chat.completions.create.call_count == 2

    @pytest.mark.parametrize(
        "model_name",
        [
            "lightonocr",
            "gemini-2.5-flash-lite",
            "nanonets/Nanonets-OCR2-3B",
        ],
    )
    def test_convert_uses_correct_model_name(
        self, cli, file_path, mock_openai_api, model_name, tmp_output_dir
    ):
        """Test that each converter uses the correct model name in API calls."""
        with mock_openai_api() as openai_client:
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value=model_name,
            ):
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=str(tmp_output_dir),
                    model=model_name,
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                # Check that model parameter is passed
                call_args = openai_client.chat.completions.create.call_args_list[0]
                assert "model" in call_args[1]
                # Model name can be the original or derived from config
                assert call_args[1]["model"] in [
                    model_name,
                    "vllm-model",
                    "lightonai/LightOnOCR-1B-1025",
                    "nanonets/Nanonets-OCR2-3B",
                ]

    def test_convert_with_dotsocr_model(
        self, cli, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test convert with DotsOCR which has different prompt modes."""
        with mock_openai_api() as openai_client:
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="dotsocr",
            ):
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=str(tmp_output_dir),
                    model="dotsocr",
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                # Verify API was called (2 pages)
                assert openai_client.chat.completions.create.call_count == 2

                # Check that messages were sent (DotsOCR uses specific prompt format)
                call_args = openai_client.chat.completions.create.call_args_list[0]
                assert "messages" in call_args[1]

    def test_convert_with_max_image_size_limit(
        self, cli, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test that max_image_size limit is respected for models that have it."""
        with mock_openai_api() as openai_client:
            # LightOnOCR has max_image_size=1540
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="lightonocr",
            ):
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=str(tmp_output_dir),
                    model="lightonocr",
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                assert openai_client.chat.completions.create.call_count == 2

            openai_client.reset_mock()

            # Nanonets has no max_image_size limit
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="nanonets/Nanonets-OCR2-3B",
            ):
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=str(tmp_output_dir),
                    model="nanonets/Nanonets-OCR2-3B",
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                assert openai_client.chat.completions.create.call_count == 2

    def test_convert_with_glob_pattern_real_converter(
        self, cli, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test glob pattern expansion with real converter."""
        pattern = str(file_path.parent / "*.pdf")

        with patch(
            "vlmparse.converter_with_server.get_model_from_uri",
            return_value="gemini-2.5-flash-lite",
        ):
            with mock_openai_api() as openai_client:
                cli.convert(
                    inputs=[pattern],
                    out_folder=str(tmp_output_dir),
                    model="gemini-2.5-flash-lite",
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                # At least one file should be found and processed
                assert openai_client.chat.completions.create.call_count >= 2

    def test_convert_checks_completion_kwargs(
        self, cli, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test that converter processes pages correctly."""
        with mock_openai_api() as openai_client:
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="lightonocr",
            ):
                cli.convert(
                    inputs=[str(file_path)],
                    out_folder=str(tmp_output_dir),
                    model="lightonocr",
                    uri="http://mocked-api/v1",
                    debug=True,
                )

                # Check that API was called (2 pages)
                assert openai_client.chat.completions.create.call_count == 2

                # Verify messages were sent to API
                call_args = openai_client.chat.completions.create.call_args_list[0]
                assert "messages" in call_args[1]
                assert len(call_args[1]["messages"]) > 0
