import pytest

from vlmparse.converter_with_server import ConverterWithServer


class TestBatchParser:
    """Tests for ConverterWithServer (acting as BatchParser)."""

    def test_init_starts_docker_server(self, mock_docker_operations):
        """Test that initializing with a model requiring docker starts the server."""
        # Setup using unified mocking system
        with mock_docker_operations(include_client=True) as (
            mock_docker_registry,
            mock_config,
            mock_server,
            mock_client,
        ):
            # Initialize
            with ConverterWithServer(model="test_model", provider="hf") as parser:
                # Verify interactions
                # For provider="hf", we expect lookup in registry then fallback if not found
                # The mock setup seems to imply it finds it or defaults?
                # In start_server, if provider="hf" and not in registry, it makes a default config.
                # If mock_docker_registry.get returns something, it uses it.
                mock_docker_registry.get.assert_called_with("test_model")
                mock_config.get_server.assert_called_with(auto_stop=True)
                mock_server.start.assert_called_once()
                mock_config.get_client.assert_called_once()
                assert parser.client == mock_client

    def test_init_no_docker_fallback(self, mock_docker_operations, mock_openai_api):
        """Test fallback to standard converter when no docker config exists."""
        # Setup mocks - docker returns None, use real converter registry
        with mock_docker_operations(
            model_filter=lambda model: False  # No docker for any model
        ) as (mock_docker_reg, _, _, _):
            with mock_openai_api():
                # Initialize with a real model from registry
                with ConverterWithServer(model="gemini-2.5-flash-lite") as parser:
                    # Verify interactions
                    # Because provider="registry" (default), we check valid model
                    # If model is not in docker registry, it shouldn't try to get with default=False anymore
                    # logic:
                    # if provider="registry": if model in docker_config_registry.list_models() -> start_server
                    # else -> directly to converter_config_registry

                    # The mock_docker_operations mocks list_models via the model_filter probably?
                    # Let's check how mock_docker_operations is implemented in conftest
                    # Assuming it mocks list_models.

                    # Client should be initialized from real converter registry
                    assert parser.client is not None

    def test_parse_updates_client_config(
        self, mock_docker_operations, datadir, mock_openai_api, tmp_path
    ):
        """Test that parse method updates client configuration and calls batch."""
        # Use real test file
        test_file = datadir / "Fiche_Graines_A5.pdf"

        with mock_docker_operations(
            model_filter=lambda model: False  # No docker for any model
        ):
            with mock_openai_api():
                with ConverterWithServer(model="gemini-2.5-flash-lite") as parser:
                    # Call parse with real file
                    parser.client.return_documents_in_batch_mode = True
                    documents = parser.parse(
                        inputs=[str(test_file)],
                        out_folder=str(tmp_path),
                        mode="md",
                        dpi=300,
                        debug=True,
                    )

                    # Verify client config updates
                    assert parser.client.config.dpi == 300
                    assert parser.client.debug is True
                    assert parser.client.save_mode == "md"
                    # Concurrency should be 1 because debug=True
                    assert parser.client.num_concurrent_files == 1
                    assert parser.client.num_concurrent_pages == 1

                    # Verify result
                    assert documents is not None
                    assert len(documents) > 0

    def test_parse_retry_logic(
        self, mock_docker_operations, datadir, mock_openai_api, tmp_path
    ):
        """Test the retrylast logic filters already processed files."""
        # Create two copies of the test file
        test_file = datadir / "Fiche_Graines_A5.pdf"
        temp_dir = tmp_path / "input_files"
        temp_dir.mkdir()
        file1 = temp_dir / "file1.pdf"
        file2 = temp_dir / "file2.pdf"

        # Copy test file to simulate multiple inputs
        import shutil

        shutil.copy(test_file, file1)
        shutil.copy(test_file, file2)

        # Setup folder structure for retry
        run_folder = tmp_path / "output" / "run1"
        results_folder = run_folder / "results"
        results_folder.mkdir(parents=True)

        # Create a processed result for file1
        (results_folder / "file1.zip").touch()

        with mock_docker_operations(model_filter=lambda model: False):
            with mock_openai_api():
                with ConverterWithServer(model="gemini-2.5-flash-lite") as parser:
                    parser.client.return_documents_in_batch_mode = True
                    # Call parse with retrylast - should only process file2
                    documents = parser.parse(
                        inputs=[str(file1), str(file2)],
                        out_folder=str(tmp_path / "output"),
                        retrylast=True,
                    )

                    # Should only process file2 (file1 was already processed)
                    # Verify by checking that only 1 file was processed
                    assert documents is not None
                    assert len(documents) == 1

    def test_parse_retry_no_previous_runs(
        self, mock_docker_operations, datadir, mock_openai_api, tmp_path
    ):
        """Test that retrylast raises ValueError if no previous runs found."""
        test_file = datadir / "Fiche_Graines_A5.pdf"

        with mock_docker_operations(model_filter=lambda model: False):
            with mock_openai_api():
                with ConverterWithServer(model="gemini-2.5-flash-lite") as parser:
                    # tmp_path is empty, so os.listdir(tmp_path) will be empty
                    with pytest.raises(ValueError, match="No previous runs found"):
                        parser.parse(
                            inputs=[str(test_file)],
                            out_folder=str(tmp_path),
                            retrylast=True,
                        )
