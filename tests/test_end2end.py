from vlmparse.registries import converter_config_registry
import pytest


@pytest.mark.parametrize("model", ["gemini-2.5-flash-lite"])
def test_convert(file_path, model):
    config = converter_config_registry.get(model)
    client = config.get_client()
    docs = client.batch([file_path])
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.pages) == 2
    assert doc.pages[0].text is not None
    assert doc.pages[1].text is not None
