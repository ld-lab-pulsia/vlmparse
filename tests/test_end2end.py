import os

import pytest

from vlmparse.registries import converter_config_registry, docker_config_registry


@pytest.mark.parametrize("model", ["gemini-2.5-flash-lite"])
def test_convert(file_path, model):
    config = converter_config_registry.get(model)
    client = config.get_client(return_documents_in_batch_mode=True)
    docs = client.batch([file_path])
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.pages) == 2
    assert doc.pages[0].text is not None
    assert doc.pages[1].text is not None


@pytest.mark.skipif(
    "RUN_DEPLOYMENT_VLLM" not in os.environ
    or os.environ["RUN_DEPLOYMENT_VLLM"] == "false"
    or "GPU_TEST_VLMPARSE" not in os.environ,
    reason="Skipping because RUN_DEPLOYMENT_VLLM is not set or is false or GPU_TEST is not set",
)
@pytest.mark.parametrize(
    "model",
    [
        "docling",
        "lightonocr",
        "dotsocr",
        "nanonets/Nanonets-OCR2-3B",
        "hunyuanocr",
        "olmocr-2-fp8",
        "paddleocrvl",
        "mineru25",
        "chandra",
        "deepseekocr",
        "granite-docling",
    ],
)
def test_convert_with_docker(file_path, model):
    """Test conversion with automatic Docker deployment (requires GPU due to vllm limitations)."""
    # Get docker config
    docker_config = docker_config_registry.get(model)
    assert docker_config is not None, f"No docker config found for {model}"

    # Start server with auto_stop=True for automatic cleanup
    server = docker_config.get_server(auto_stop=True)
    server.gpu_device_ids = os.environ["GPU_TEST_VLMPARSE"].split(",")
    server.start()

    # Get client from docker config
    client = docker_config.get_client(return_documents_in_batch_mode=True)

    # Convert document
    docs = client.batch([file_path])

    # Assertions
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.pages) == 2
    assert doc.pages[0].text is not None
    assert doc.pages[1].text is not None

    # Server will be automatically stopped due to auto_stop=True
