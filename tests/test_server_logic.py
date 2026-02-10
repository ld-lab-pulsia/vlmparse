from unittest.mock import MagicMock, patch

import pytest

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.converter_with_server import ConverterWithServer, start_server
from vlmparse.registries import ConverterConfigRegistry
from vlmparse.servers.docker_server import DEFAULT_MODEL_NAME, VLLMDockerServerConfig
from vlmparse.servers.server_registry import DockerConfigRegistry

# --- Test Registries ---


def test_converter_config_registry_get_valid():
    registry = ConverterConfigRegistry()
    registry.register(
        "test_model", lambda uri=None: OpenAIConverterConfig(model_name="test_model")
    )

    config = registry.get("test_model")
    assert isinstance(config, OpenAIConverterConfig)
    assert config.model_name == "test_model"


def test_converter_config_registry_get_invalid():
    registry = ConverterConfigRegistry()
    with pytest.raises(ValueError, match="Model 'non_existent' not found in registry"):
        registry.get("non_existent")


def test_docker_config_registry_get_valid():
    registry = DockerConfigRegistry()
    registry.register(
        "test_model", lambda: VLLMDockerServerConfig(model_name="test_model")
    )

    config = registry.get("test_model")
    assert isinstance(config, VLLMDockerServerConfig)
    assert config.model_name == "test_model"


def test_docker_config_registry_get_invalid():
    registry = DockerConfigRegistry()
    config = registry.get("non_existent")
    assert config is None


# --- Test start_server ---


@patch("vlmparse.registries.docker_config_registry")
@patch("vlmparse.servers.docker_server.VLLMDockerServerConfig")
def test_start_server_registry_success(mock_vllm_config, mock_registry):
    mock_config = MagicMock()
    mock_config.get_server.return_value.start.return_value = (
        "http://localhost:8000",
        MagicMock(),
    )
    mock_registry.get.return_value = mock_config

    start_server("test_model", gpus="0", provider="registry")

    # Check calling with proper model name
    mock_registry.get.assert_called_with("test_model")
    mock_config.get_server.assert_called()


@patch("vlmparse.registries.docker_config_registry")
def test_start_server_registry_not_found(mock_registry):
    mock_registry.get.return_value = None

    with pytest.raises(ValueError, match="Model 'unknown_model' not found in registry"):
        start_server("unknown_model", gpus="0", provider="registry")


@patch("vlmparse.registries.docker_config_registry")
@patch("vlmparse.servers.docker_server.VLLMDockerServerConfig")
def test_start_server_hf_fallback(mock_vllm_config, mock_registry):
    mock_registry.get.return_value = None
    mock_server_instance = MagicMock()
    mock_server_instance.start.return_value = ("http://localhost:8000", MagicMock())

    # Mock the return value of VLLMDockerServerConfig(...) which is an object that has .get_server()
    mock_docker_config_instance = MagicMock()
    mock_docker_config_instance.get_server.return_value = mock_server_instance
    mock_vllm_config.return_value = mock_docker_config_instance

    _, _, _, config = start_server("hf_model", gpus="0", provider="hf")

    # Check that we tried to get from registry first (which returned None)
    mock_registry.get.assert_called_with("hf_model")
    # Check that we instantiated VLLMDockerServerConfig
    mock_vllm_config.assert_called_with(
        model_name="hf_model", default_model_name=DEFAULT_MODEL_NAME
    )
    assert config == mock_docker_config_instance


# --- Test ConverterWithServer ---


@patch("vlmparse.converter_with_server.start_server")
@patch("vlmparse.registries.converter_config_registry")
@patch("vlmparse.registries.docker_config_registry")
def test_converter_registry_unregistered(
    mock_docker_reg, mock_converter_reg, mock_start
):
    # Setup: Not in docker registry, Not in converter registry (raise ValueError)
    mock_docker_reg.list_models.return_value = []
    mock_converter_reg.get.side_effect = ValueError("Not found")

    # Case: provider="registry", uri provided (remote)
    converter = ConverterWithServer(
        model="unknown", uri="http://foo", provider="registry"
    )

    with pytest.raises(ValueError, match="Not found"):
        converter.start_server_and_client()

    mock_converter_reg.get.assert_called_with("unknown", uri="http://foo")


@patch("vlmparse.converter_with_server.start_server")
@patch("vlmparse.registries.converter_config_registry")
@patch("vlmparse.registries.docker_config_registry")
def test_converter_hf_unregistered(mock_docker_reg, mock_converter_reg, mock_start):
    # Setup: Not in docker registry
    mock_docker_reg.list_models.return_value = []

    # Case: provider="hf", uri provided (remote) -> Should use OpenAIConverterConfig generic
    converter = ConverterWithServer(model="unknown", uri="http://foo", provider="hf")

    converter.start_server_and_client()
    assert converter.client is not None
    assert isinstance(converter.client.config, OpenAIConverterConfig)
    assert converter.client.config.model_name == "unknown"
    mock_converter_reg.get.assert_not_called()


@patch("vlmparse.converter_with_server.start_server")
@patch("vlmparse.registries.converter_config_registry")
@patch("vlmparse.registries.docker_config_registry")
def test_converter_with_local_start(mock_docker_reg, mock_converter_reg, mock_start):
    # Setup: provider="hf" means force start local
    mock_docker_reg.list_models.return_value = []
    mock_docker_config = MagicMock()
    mock_start.return_value = ("url", "container", "server", mock_docker_config)

    converter = ConverterWithServer(model="unknown", provider="hf")
    converter.start_server_and_client()

    mock_start.assert_called()
    mock_docker_config.get_client.assert_called()
