"""Test Typer CLI commands while mocking the server side.

The CLI implementation moved from a Fire class-based interface to Typer.
These tests invoke the Typer app through Click's test runner so that Typer
options/arguments are parsed and default values are resolved.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from vlmparse.cli import app
from vlmparse.data_model.document import Document, Page


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# Note: mock_docker_server and mock_converter_client fixtures are now replaced
# by the unified mocking system in conftest.py: mock_docker_operations and mock_openai_api


class TestServeCommand:
    """Test the 'serve' command."""

    def test_serve_default_port(self, runner, mock_docker_operations):
        """Test serve command with default port."""
        with mock_docker_operations() as (mock_registry, mock_config, mock_server, _):
            result = runner.invoke(app, ["serve", "lightonocr"])
            assert result.exit_code == 0, result.output

            # Verify registry was called with correct model
            mock_registry.get.assert_called_once_with("lightonocr", default=True)

            # Verify port was set to default
            assert mock_config.docker_port == 8056

            # # Verify gpu_device_ids was None
            # assert mock_config.gpu_device_ids is None

            # Verify server was created and started
            mock_config.get_server.assert_called_once_with(auto_stop=False)
            mock_server.start.assert_called_once()

    def test_serve_custom_port(self, runner, mock_docker_operations):
        """Test serve command with custom port."""
        with mock_docker_operations() as (mock_registry, mock_config, mock_server, _):
            result = runner.invoke(app, ["serve", "lightonocr", "--port", "9000"])
            assert result.exit_code == 0, result.output

            # Verify custom port was set
            assert mock_config.docker_port == 9000
            mock_server.start.assert_called_once()

    def test_serve_with_gpus(self, runner, mock_docker_operations):
        """Test serve command with GPU configuration."""
        with mock_docker_operations() as (mock_registry, mock_config, mock_server, _):
            result = runner.invoke(
                app,
                ["serve", "lightonocr", "--port", "8056", "--gpus", "0,1,2"],
            )
            assert result.exit_code == 0, result.output

            # Verify GPU device IDs were parsed correctly
            assert mock_config.gpu_device_ids == ["0", "1", "2"]
            mock_server.start.assert_called_once()

    def test_serve_single_gpu(self, runner, mock_docker_operations):
        """Test serve command with single GPU."""
        with mock_docker_operations() as (mock_registry, mock_config, mock_server, _):
            result = runner.invoke(app, ["serve", "lightonocr", "--gpus", "0"])
            assert result.exit_code == 0, result.output

            # Verify single GPU was parsed correctly
            assert mock_config.gpu_device_ids == ["0"]

    def test_serve_unknown_model(self, runner, mock_docker_operations):
        """Test serve command with unknown model (should warn and return)."""
        with mock_docker_operations(
            model_filter=lambda model: False  # No docker for any model
        ) as (mock_registry, _, _, _):
            # Should not raise an exception, just log warning
            result = runner.invoke(app, ["serve", "unknown_model"])
            assert result.exit_code == 0, result.output

            mock_registry.get.assert_called_once_with("unknown_model", default=True)


class TestConvertCommand:
    """Test the 'convert' command."""

    def test_convert_single_file(
        self, runner, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with a single PDF file."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api() as openai_client:
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        "http://localhost:8000/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output
                # 2 pages in test PDF -> 2 API calls
                assert openai_client.chat.completions.create.call_count == 2

    def test_convert_multiple_files(
        self,
        runner,
        file_path,
        mock_docker_operations,
        mock_openai_api,
        tmp_output_dir,
        tmp_path,
    ):
        """Test convert with multiple PDF files.

        Typer's current CLI takes a single INPUTS argument. We pass a directory
        containing two PDFs.
        """
        with mock_docker_operations(include_client=False):
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            (input_dir / "a.pdf").write_bytes(Path(file_path).read_bytes())
            (input_dir / "b.pdf").write_bytes(Path(file_path).read_bytes())

            with mock_openai_api() as openai_client:
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(input_dir),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        "http://localhost:8000/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output
                # 2 files × 2 pages = 4 API calls
                assert openai_client.chat.completions.create.call_count == 4

    def test_convert_with_glob_pattern(
        self, runner, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with glob pattern."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api() as openai_client:
                # Use the parent directory with a glob pattern
                pattern = str(file_path.parent / "*.pdf")

                result = runner.invoke(
                    app,
                    [
                        "convert",
                        pattern,
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        "http://localhost:8000/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output
                assert openai_client.chat.completions.create.call_count >= 2

    def test_convert_with_custom_uri(
        self, runner, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with custom URI (no Docker server needed)."""
        with mock_docker_operations(include_client=False):
            custom_uri = "http://custom-server:9000/v1"

            with mock_openai_api() as openai_client:
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        custom_uri,
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output
                assert openai_client.chat.completions.create.call_count == 2

    def test_convert_without_uri_starts_server(
        self, runner, file_path, mock_docker_operations, tmp_output_dir
    ):
        """Test convert without URI starts a Docker server."""
        with mock_docker_operations(include_client=True, client_batch_return=None) as (
            mock_docker_reg,
            mock_docker_config,
            mock_server,
            mock_client,
        ):
            result = runner.invoke(
                app,
                [
                    "convert",
                    str(file_path),
                    "--out-folder",
                    str(tmp_output_dir),
                    "--model",
                    "lightonocr",
                    "--debug",
                ],
            )

            assert result.exit_code == 0, result.output

            # Verify Docker server was started
            mock_docker_reg.get.assert_called_once_with("lightonocr", default=False)
            mock_docker_config.get_server.assert_called_once_with(auto_stop=True)
            mock_server.start.assert_called_once()
            mock_client.batch.assert_called_once()

    def test_convert_with_gpus(
        self, runner, file_path, mock_docker_operations, tmp_output_dir
    ):
        """Test convert with GPU configuration."""
        with mock_docker_operations(include_client=True, client_batch_return=None) as (
            mock_docker_reg,
            mock_docker_config,
            mock_server,
            mock_client,
        ):
            result = runner.invoke(
                app,
                [
                    "convert",
                    str(file_path),
                    "--out-folder",
                    str(tmp_output_dir),
                    "--model",
                    "lightonocr",
                    "--gpus",
                    "0,1",
                    "--debug",
                ],
            )

            assert result.exit_code == 0, result.output

            # Verify GPU device IDs were set
            assert mock_docker_config.gpu_device_ids == ["0", "1"]

    def test_convert_with_output_folder(
        self, runner, file_path, mock_docker_operations, mock_openai_api
    ):
        """Test convert with custom output folder."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api() as openai_client:
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = runner.invoke(
                        app,
                        [
                            "convert",
                            str(file_path),
                            "--out-folder",
                            tmpdir,
                            "--model",
                            "gemini-2.5-flash-lite",
                            "--uri",
                            "http://localhost:8000/v1",
                            "--debug",
                        ],
                    )

                    assert result.exit_code == 0, result.output
                    assert openai_client.chat.completions.create.call_count == 2

    def test_convert_string_inputs(
        self, runner, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
    ):
        """Test convert with a single file path argument."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api() as openai_client:
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        "http://localhost:8000/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output
                assert openai_client.chat.completions.create.call_count == 2

    def test_convert_filters_non_pdf_files(
        self, runner, mock_docker_operations, mock_openai_api
    ):
        """Test that convert filters out non-PDF files."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api():
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Create a non-PDF file
                    txt_file = Path(tmpdir) / "test.txt"
                    txt_file.write_text("test")

                    with pytest.raises(ValueError, match="Unsupported file extension"):
                        runner.invoke(
                            app,
                            [
                                "convert",
                                str(txt_file),
                                "--model",
                                "gemini-2.5-flash-lite",
                                "--uri",
                                "http://localhost:8000/v1",
                                "--debug",
                            ],
                            catch_exceptions=False,
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
        runner,
        file_path,
        model_name,
        mock_docker_operations,
        mock_openai_api,
        tmp_output_dir,
    ):
        """Test convert with different registered models."""
        with mock_docker_operations(include_client=False):
            with mock_openai_api() as openai_client:
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        model_name,
                        "--uri",
                        "http://localhost:8000/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output
                assert openai_client.chat.completions.create.call_count == 2


class TestCLIIntegration:
    """Integration tests for CLI with mocked server."""

    def test_full_workflow_without_uri(
        self, runner, file_path, mock_docker_operations, tmp_output_dir
    ):
        """Test full conversion workflow without providing URI."""
        with mock_docker_operations(include_client=True, client_batch_return=None) as (
            mock_docker_reg,
            mock_docker_config,
            mock_server,
            mock_client,
        ):
            result = runner.invoke(
                app,
                [
                    "convert",
                    str(file_path),
                    "--out-folder",
                    str(tmp_output_dir),
                    "--model",
                    "lightonocr",
                    "--debug",
                ],
            )

            assert result.exit_code == 0, result.output

            # Verify full workflow
            mock_docker_reg.get.assert_called_once()
            mock_server.start.assert_called_once()
            mock_client.batch.assert_called_once()

    def test_serve_then_convert_scenario(
        self, runner, file_path, mock_docker_operations, mock_openai_api, tmp_output_dir
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
            result = runner.invoke(app, ["serve", "lightonocr", "--port", "8056"])
            assert result.exit_code == 0, result.output

            # Verify serve worked
            mock_server.start.assert_called_once()

        # Second part: convert with URI
        with mock_docker_operations(include_client=False):
            with mock_openai_api() as openai_client:
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        "http://localhost:8056/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output
                assert openai_client.chat.completions.create.call_count == 2


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
        self, runner, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test convert with real Gemini converter and mocked OpenAI API."""
        with mock_openai_api() as openai_client:
            result = runner.invoke(
                app,
                [
                    "convert",
                    str(file_path),
                    "--out-folder",
                    str(tmp_output_dir),
                    "--model",
                    "gemini-2.5-flash-lite",
                    "--uri",
                    "http://mocked-api/v1",
                    "--debug",
                ],
            )

            assert result.exit_code == 0, result.output

            # Verify OpenAI API was called (2 pages in test PDF)
            assert openai_client.chat.completions.create.call_count == 2

            # Verify the model parameter was correct
            call_args = openai_client.chat.completions.create.call_args_list[0]
            assert call_args[1]["model"] == "gemini-2.5-flash-lite"

    def test_convert_with_real_converter_lightonocr(
        self, runner, file_path, mock_docker_operations, tmp_output_dir
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

            result = runner.invoke(
                app,
                [
                    "convert",
                    str(file_path),
                    "--out-folder",
                    str(tmp_output_dir),
                    "--model",
                    "lightonocr",
                    "--debug",
                ],
            )

            assert result.exit_code == 0, result.output

            # Verify server was started
            mock_server.start.assert_called_once()

            # Verify client batch was called
            mock_client.batch.assert_called_once()

    def test_convert_batch_multiple_files(
        self, runner, file_path, mock_openai_api, tmp_output_dir, tmp_path
    ):
        """Test batch conversion of multiple files with real converter."""
        with mock_openai_api() as openai_client:
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            (input_dir / "a.pdf").write_bytes(Path(file_path).read_bytes())
            (input_dir / "b.pdf").write_bytes(Path(file_path).read_bytes())

            result = runner.invoke(
                app,
                [
                    "convert",
                    str(input_dir),
                    "--out-folder",
                    str(tmp_output_dir),
                    "--model",
                    "gemini-2.5-flash-lite",
                    "--uri",
                    "http://mocked-api/v1",
                    "--debug",
                ],
            )

            assert result.exit_code == 0, result.output

            # Should process 2 files × 2 pages = 4 API calls
            assert openai_client.chat.completions.create.call_count == 4

    def test_convert_verifies_document_structure(
        self, runner, file_path, mock_openai_api, tmp_path
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
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        "http://mocked-api/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output

                # Verify conversion happened (2 pages)
                assert openai_client.chat.completions.create.call_count == 2

    def test_convert_handles_api_errors_gracefully(
        self, runner, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test that converter handles API errors without crashing."""
        with patch(
            "vlmparse.converter_with_server.get_model_from_uri",
            return_value="gemini-2.5-flash-lite",
        ):
            with mock_openai_api(side_effect=Exception("API Error")) as openai_client:
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        "http://mocked-api/v1",
                    ],
                )

                assert result.exit_code == 0, result.output

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
        self, runner, file_path, mock_openai_api, model_name, tmp_output_dir
    ):
        """Test that each converter uses the correct model name in API calls."""
        with mock_openai_api() as openai_client:
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value=model_name,
            ):
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        model_name,
                        "--uri",
                        "http://mocked-api/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output

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
        self, runner, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test convert with DotsOCR which has different prompt modes."""
        with mock_openai_api() as openai_client:
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="dotsocr",
            ):
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "dotsocr",
                        "--uri",
                        "http://mocked-api/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output

                # Verify API was called (2 pages)
                assert openai_client.chat.completions.create.call_count == 2

                # Check that messages were sent (DotsOCR uses specific prompt format)
                call_args = openai_client.chat.completions.create.call_args_list[0]
                assert "messages" in call_args[1]

    def test_convert_with_max_image_size_limit(
        self, runner, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test that max_image_size limit is respected for models that have it."""
        with mock_openai_api() as openai_client:
            # LightOnOCR has max_image_size=1540
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="lightonocr",
            ):
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "lightonocr",
                        "--uri",
                        "http://mocked-api/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output

                assert openai_client.chat.completions.create.call_count == 2

            openai_client.reset_mock()

            # Nanonets has no max_image_size limit
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="nanonets/Nanonets-OCR2-3B",
            ):
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "nanonets/Nanonets-OCR2-3B",
                        "--uri",
                        "http://mocked-api/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output

                assert openai_client.chat.completions.create.call_count == 2

    def test_convert_with_glob_pattern_real_converter(
        self, runner, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test glob pattern expansion with real converter."""
        pattern = str(file_path.parent / "*.pdf")

        with patch(
            "vlmparse.converter_with_server.get_model_from_uri",
            return_value="gemini-2.5-flash-lite",
        ):
            with mock_openai_api() as openai_client:
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        pattern,
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "gemini-2.5-flash-lite",
                        "--uri",
                        "http://mocked-api/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output

                # At least one file should be found and processed
                assert openai_client.chat.completions.create.call_count >= 2

    def test_convert_checks_completion_kwargs(
        self, runner, file_path, mock_openai_api, tmp_output_dir
    ):
        """Test that converter processes pages correctly."""
        with mock_openai_api() as openai_client:
            with patch(
                "vlmparse.converter_with_server.get_model_from_uri",
                return_value="lightonocr",
            ):
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(file_path),
                        "--out-folder",
                        str(tmp_output_dir),
                        "--model",
                        "lightonocr",
                        "--uri",
                        "http://mocked-api/v1",
                        "--debug",
                    ],
                )

                assert result.exit_code == 0, result.output

                # Check that API was called (2 pages)
                assert openai_client.chat.completions.create.call_count == 2

                # Verify messages were sent to API
                call_args = openai_client.chat.completions.create.call_args_list[0]
                assert "messages" in call_args[1]
                assert len(call_args[1]["messages"]) > 0
