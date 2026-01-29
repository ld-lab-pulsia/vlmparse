import os
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def datadir():
    return Path(__file__).parent / "data"


os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


@pytest.fixture(scope="session", autouse=True)
def file_path():
    return Path(__file__).parent / "data" / "Fiche_Graines_A5.pdf"


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# Unified Mocking System
# =============================================================================


class MockOpenAIResponse:
    """Configurable mock for OpenAI API responses."""

    def __init__(
        self,
        content: str = "# Test Document\n\nThis is a test page with some content.",
        prompt_tokens: int = 50,
        completion_tokens: int = 150,
        reasoning_tokens: int = 30,
        side_effect: Optional[Exception] = None,
    ):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.side_effect = side_effect

    def create_response(self) -> MagicMock:
        """Create a mock response object."""
        if self.side_effect:
            raise self.side_effect

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = self.content
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = self.prompt_tokens
        mock_response.usage.completion_tokens = self.completion_tokens
        mock_response.usage.reasoning_tokens = self.reasoning_tokens
        return mock_response


class MockDockerServer:
    """Configurable mock for Docker server operations."""

    def __init__(
        self,
        uri: str = "http://localhost:8056",
        container_id: str = "test_container_id",
        container_name: str = "test_container",
        start_error: Optional[Exception] = None,
    ):
        self.uri = uri
        self.container_id = container_id
        self.container_name = container_name
        self.start_error = start_error

    def create_mock(self) -> tuple[MagicMock, MagicMock, MagicMock]:
        """Create mock config, server, and container objects."""
        mock_config = MagicMock()
        mock_server = MagicMock()
        mock_container = MagicMock()

        mock_container.id = self.container_id
        mock_container.name = self.container_name

        if self.start_error:
            mock_server.start.side_effect = self.start_error
        else:
            mock_server.start.return_value = (self.uri, None)

        mock_config.get_server.return_value = mock_server

        return mock_config, mock_server, mock_container


@pytest.fixture
def mock_openai_api():
    """
    Unified fixture for mocking OpenAI API calls.

    The fixture returns a context manager that patches the OpenAI client and returns the mock instance.

    Usage:
        def test_example(mock_openai_api):
            with mock_openai_api() as client:
                # Use client in test

            # Or with custom content
            with mock_openai_api(content="Custom") as client:
                # client is configured

            # For streaming responses
            with mock_openai_api(content="Streamed", stream=True) as client:
                # client is configured for streaming
    """
    from contextlib import contextmanager

    @contextmanager
    def _create_mock(
        content: str = "# Test Document\n\nThis is a test page with some content.",
        prompt_tokens: int = 50,
        completion_tokens: int = 150,
        reasoning_tokens: int = 30,
        side_effect: Optional[Exception] = None,
        call_tracker: Optional[Callable] = None,
        stream: bool = False,
    ):
        """Create and configure a mock OpenAI client."""
        with patch("openai.AsyncOpenAI") as mock_client_class:
            response_config = MockOpenAIResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                side_effect=side_effect,
            )

            mock_instance = MagicMock()
            # Support for async close() method
            mock_instance.close = AsyncMock()

            if side_effect:
                mock_instance.chat.completions.create = AsyncMock(
                    side_effect=side_effect
                )
            elif call_tracker:
                mock_instance.chat.completions.create = AsyncMock(
                    side_effect=call_tracker
                )
            else:

                async def create_response(*args, **kwargs):
                    is_stream = kwargs.get("stream", False)
                    if is_stream or stream:
                        # Return an async iterator for streaming responses
                        return _create_stream_response(content)
                    return response_config.create_response()

                mock_instance.chat.completions.create = AsyncMock(
                    side_effect=create_response
                )

            mock_client_class.return_value = mock_instance
            yield mock_instance

    async def _create_stream_response(content: str):
        """Create an async iterator that yields streaming chunks."""
        # Split content into chunks to simulate streaming
        chunks = [content[i : i + 10] for i in range(0, len(content), 10)]
        for chunk_text in chunks:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = chunk_text
            yield chunk

    return _create_mock


@pytest.fixture
def mock_docker_operations():
    """
    Unified fixture for mocking Docker server operations.

    The fixture returns a context manager that patches the docker registry.

    Usage:
        def test_example(mock_docker_operations):
            with mock_docker_operations() as (mock_reg, mock_config, mock_server, mock_client):
                # Use mocks in test
    """
    from contextlib import contextmanager

    @contextmanager
    def _create_mock(
        uri: str = "http://localhost:8056",
        container_id: str = "test_container_id",
        container_name: str = "test_container",
        start_error: Optional[Exception] = None,
        model_filter: Optional[Callable[[str], bool]] = None,
        include_client: bool = True,
        client_batch_return: Optional[list] = None,
    ):
        """Create and configure mock Docker registry and related objects."""
        with patch(
            "vlmparse.registries.docker_config_registry"
        ) as mock_docker_registry:
            server_config = MockDockerServer(
                uri=uri,
                container_id=container_id,
                container_name=container_name,
                start_error=start_error,
            )

            mock_config, mock_server, mock_container = server_config.create_mock()

            # Configure client if needed
            if include_client:
                mock_client = MagicMock()
                if client_batch_return is not None:
                    mock_client.batch.return_value = client_batch_return
                mock_config.get_client.return_value = mock_client
            else:
                mock_client = None

            # Configure registry behavior
            def get_docker_config(model_name: str, default: bool = False):
                if model_filter and not model_filter(model_name):
                    return None
                return mock_config

            mock_docker_registry.get.side_effect = get_docker_config

            yield mock_docker_registry, mock_config, mock_server, mock_client

    return _create_mock
