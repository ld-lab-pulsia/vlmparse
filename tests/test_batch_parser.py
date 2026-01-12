from unittest.mock import MagicMock, patch

import pytest

from vlmparse.converter_with_server import ConverterWithServer
from vlmparse.data_model.document import Document


@pytest.fixture
def mock_docker_registry():
    with patch("vlmparse.registries.docker_config_registry") as mock:
        yield mock


@pytest.fixture
def mock_converter_registry():
    with patch("vlmparse.registries.converter_config_registry") as mock:
        yield mock


@pytest.fixture
def mock_get_file_paths():
    with patch("vlmparse.converter_with_server.get_file_paths") as mock:
        yield mock


class TestBatchParser:
    """Tests for ConverterWithServer (acting as BatchParser)."""

    def test_init_starts_docker_server(self, mock_docker_registry):
        """Test that initializing with a model requiring docker starts the server."""
        # Setup mock docker config
        mock_config = MagicMock()
        mock_server = MagicMock()
        mock_client = MagicMock()

        mock_config.get_server.return_value = mock_server
        mock_config.get_client.return_value = mock_client
        mock_docker_registry.get.return_value = mock_config

        # Initialize
        parser = ConverterWithServer(model="test_model", with_vllm_server=True)

        # Verify interactions
        mock_docker_registry.get.assert_called_with("test_model", default=True)
        mock_config.get_server.assert_called_with(auto_stop=True)
        mock_server.start.assert_called_once()
        mock_config.get_client.assert_called_once()
        assert parser.client == mock_client

    def test_init_no_docker_fallback(
        self, mock_docker_registry, mock_converter_registry
    ):
        """Test fallback to standard converter when no docker config exists."""
        # Setup mocks
        mock_docker_registry.get.return_value = None

        mock_converter_config = MagicMock()
        mock_client = MagicMock()
        mock_converter_config.get_client.return_value = mock_client
        mock_converter_registry.get.return_value = mock_converter_config

        # Initialize
        parser = ConverterWithServer(model="test_model")

        # Verify interactions
        mock_docker_registry.get.assert_called_with("test_model", default=False)
        mock_converter_registry.get.assert_called_with("test_model")
        mock_converter_config.get_client.assert_called_once()
        assert parser.client == mock_client

    def test_init_with_uri(self, mock_converter_registry):
        """Test initialization with explicit URI."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_converter_registry.get.return_value = mock_config

        parser = ConverterWithServer(model="test_model", uri="http://custom.uri")

        mock_converter_registry.get.assert_called_with(
            "test_model", uri="http://custom.uri"
        )
        mock_config.get_client.assert_called_once()
        assert parser.client == mock_client

    def test_parse_updates_client_config(
        self, mock_docker_registry, mock_get_file_paths, tmp_path
    ):
        """Test that parse method updates client configuration and calls batch."""
        # Setup mocks
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_docker_registry.get.return_value = mock_config

        mock_get_file_paths.return_value = ["file1.pdf", "file2.pdf"]

        # Mock batch return value
        mock_doc = MagicMock(spec=Document)
        mock_client.batch.return_value = [mock_doc, mock_doc]

        parser = ConverterWithServer(model="test_model")

        # Call parse
        documents = parser.parse(
            inputs=["dummy"], out_folder=str(tmp_path), mode="md", dpi=300, debug=True
        )

        # Verify client config updates
        assert mock_client.config.dpi == 300
        assert mock_client.debug is True
        assert mock_client.save_mode == "md"
        # Concurrency should be 1 because debug=True
        assert mock_client.num_concurrent_files == 1
        assert mock_client.num_concurrent_pages == 1

        # Verify batch call
        mock_client.batch.assert_called_once_with(["file1.pdf", "file2.pdf"])

        # Verify result
        assert len(documents) == 2
        assert documents[0] == mock_doc

    def test_parse_retry_logic(
        self, mock_docker_registry, mock_get_file_paths, tmp_path
    ):
        """Test the retrylast logic filters already processed files."""
        # Setup folder structure for retry
        run_folder = tmp_path / "run1"
        results_folder = run_folder / "results"
        results_folder.mkdir(parents=True)

        # Create a processed result
        (results_folder / "file1.zip").touch()

        # Setup mocks
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_docker_registry.get.return_value = mock_config

        # Input has file1 (processed) and file2 (new)
        mock_get_file_paths.return_value = ["path/to/file1.pdf", "path/to/file2.pdf"]

        parser = ConverterWithServer(model="test_model")

        # Call parse with retrylast
        parser.parse(inputs=["dummy"], out_folder=str(tmp_path), retrylast=True)

        # Verify only file2 was sent to batch
        # file1 should be filtered out because file1.zip exists
        call_args = mock_client.batch.call_args
        assert call_args is not None
        batch_files = call_args[0][0]
        assert len(batch_files) == 1
        assert "file2.pdf" in batch_files[0]
        assert "file1.pdf" not in batch_files[0]

    def test_parse_retry_no_previous_runs(
        self, mock_docker_registry, mock_get_file_paths, tmp_path
    ):
        """Test that retrylast raises ValueError if no previous runs found."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_docker_registry.get.return_value = mock_config

        parser = ConverterWithServer(model="test_model")

        # tmp_path is empty, so os.listdir(tmp_path) will be empty

        with pytest.raises(ValueError, match="No previous runs found"):
            parser.parse(inputs=["dummy"], out_folder=str(tmp_path), retrylast=True)
