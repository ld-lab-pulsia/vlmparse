import os

import pytest

from vlmparse.registries import converter_config_registry


@pytest.mark.parametrize("model", ["gemini-2.5-flash-lite"])
def test_convert(file_path, model, tmp_output_dir):
    config = converter_config_registry.get(model)
    client = config.get_client(
        return_documents_in_batch_mode=True, debug=True, save_folder=str(tmp_output_dir)
    )
    docs = client.batch([file_path])
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.pages) == 2
    assert doc.pages[0].text is not None
    assert doc.pages[1].text is not None

    if model in ["gemini-2.5-flash-lite"]:
        assert doc.pages[0].completion_tokens > 0
        assert doc.pages[0].prompt_tokens > 0


@pytest.mark.skip(reason="Disabled to avoid excessive API calls")
@pytest.mark.parametrize("model", ["mistral-ocr"])
def test_convert_mistral_ocr(file_path, model, tmp_output_dir):
    config = converter_config_registry.get(model)
    client = config.get_client(
        return_documents_in_batch_mode=True, debug=True, save_folder=str(tmp_output_dir)
    )
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
        "Qwen/Qwen3-VL-8B-Instruct",
        "lightonocr2",
    ],
)
def test_converter_with_server_with_docker(file_path, model, tmp_output_dir):
    """Test conversion with automatic Docker deployment (requires GPU due to vllm limitations)."""

    from vlmparse.converter_with_server import ConverterWithServer

    prompt_modes_by_model = {
        "dotsocr": ["ocr", "ocr_layout"],
        "deepseekocr": ["ocr", "ocr_layout", "image_description"],
        "paddleocrvl": ["ocr", "table", "formula", "chart"],
        "chandra": ["ocr", "ocr_layout"],
    }

    with ConverterWithServer(
        model=model,
        uri=None,
        gpus=os.environ["GPU_TEST_VLMPARSE"],
        with_vllm_server=True,
        concurrency=10,
        port=8173,
    ) as converter_with_server:
        converter_with_server.client.return_documents_in_batch_mode = True

        docs = converter_with_server.parse(
            [str(file_path), str(file_path)], out_folder=str(tmp_output_dir), debug=True
        )

        # Assertions
        assert len(docs) == 2
        doc = docs[0]
        assert len(doc.pages) == 2
        assert doc.pages[0].text is not None and len(doc.pages[0].text) > 0
        assert doc.pages[1].text is not None and len(doc.pages[1].text) > 0

        prompt_modes = prompt_modes_by_model.get(model, [])
        for conversion_mode in prompt_modes:
            docs = converter_with_server.parse(
                [str(file_path)],
                out_folder=str(tmp_output_dir),
                debug=True,
                conversion_mode=conversion_mode,
            )

            assert len(docs) == 1
            doc = docs[0]
            assert len(doc.pages) == 2
            assert doc.pages[0].text is not None and len(doc.pages[0].text) > 0
            assert doc.pages[1].text is not None and len(doc.pages[1].text) > 0

            if conversion_mode == "ocr_layout" and model in {
                "dotsocr",
                "deepseekocr",
                "chandra",
            }:
                assert doc.pages[0].items is not None
                assert len(doc.pages[0].items) > 0
