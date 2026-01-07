import os
from collections.abc import Callable

from vlmparse.clients.chandra import ChandraConverterConfig, ChandraDockerServerConfig
from vlmparse.clients.deepseekocr import (
    DeepSeekOCRConverterConfig,
    DeepSeekOCRDockerServerConfig,
)
from vlmparse.clients.docling import DoclingConverterConfig, DoclingDockerServerConfig
from vlmparse.clients.dotsocr import DotsOCRConverterConfig, DotsOCRDockerServerConfig
from vlmparse.clients.granite_docling import (
    GraniteDoclingConverterConfig,
    GraniteDoclingDockerServerConfig,
)
from vlmparse.clients.hunyuanocr import (
    HunyuanOCRConverterConfig,
    HunyuanOCRDockerServerConfig,
)
from vlmparse.clients.lightonocr import (
    LightOnOCRConverterConfig,
    LightOnOCRDockerServerConfig,
)
from vlmparse.clients.mineru import MinerUConverterConfig, MinerUDockerServerConfig
from vlmparse.clients.nanonetocr import (
    NanonetOCR2ConverterConfig,
    NanonetOCR2DockerServerConfig,
)
from vlmparse.clients.olmocr import OlmOCRConverterConfig, OlmOCRDockerServerConfig
from vlmparse.clients.openai_converter import LLMParams, OpenAIConverterConfig
from vlmparse.clients.paddleocrvl import (
    PaddleOCRVLConverterConfig,
    PaddleOCRVLDockerServerConfig,
)
from vlmparse.servers.docker_server import DEFAULT_MODEL_NAME, docker_config_registry


def get_default(cls, field_name):
    field_info = cls.model_fields.get(field_name)
    if field_info is None:
        return [] if field_name == "aliases" else None
    if field_info.default_factory:
        return field_info.default_factory()
    return field_info.default


for server_config_cls in [
    ChandraDockerServerConfig,
    LightOnOCRDockerServerConfig,
    DotsOCRDockerServerConfig,
    PaddleOCRVLDockerServerConfig,
    NanonetOCR2DockerServerConfig,
    HunyuanOCRDockerServerConfig,
    DoclingDockerServerConfig,
    OlmOCRDockerServerConfig,
    MinerUDockerServerConfig,
    DeepSeekOCRDockerServerConfig,
    GraniteDoclingDockerServerConfig,
]:
    aliases = get_default(server_config_cls, "aliases") or []
    model_name = get_default(server_config_cls, "model_name")
    names = [n for n in aliases + [model_name] if isinstance(n, str)]
    for name in names:
        docker_config_registry.register(name, lambda cls=server_config_cls: cls())


class ConverterConfigRegistry:
    """Registry for mapping model names to their Docker configurations."""

    def __init__(self):
        self._registry = dict()

    def register(
        self,
        model_name: str,
        config_factory: Callable[[str], OpenAIConverterConfig | None],
    ):
        """Register a config factory for a model name."""
        self._registry[model_name] = config_factory

    def get(
        self, model_name: str, uri: str | None = None
    ) -> OpenAIConverterConfig | None:
        """Get config for a model name. Returns default if not registered."""
        if model_name in self._registry:
            return self._registry[model_name](uri=uri)
        # Fallback to OpenAIConverterConfig for unregistered models
        if uri is not None:
            return OpenAIConverterConfig(
                llm_params=LLMParams(base_url=uri, model_name=model_name)
            )
        return OpenAIConverterConfig(llm_params=LLMParams(model_name=model_name))

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._registry.keys())


# Global registry instance
converter_config_registry = ConverterConfigRegistry()
GOOGLE_API_BASE_URL = (
    os.getenv("GOOGLE_API_BASE_URL")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
)


for gemini_model in [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
]:
    converter_config_registry.register(
        gemini_model,
        lambda uri=None, model=gemini_model: OpenAIConverterConfig(
            llm_params=LLMParams(
                model_name=model,
                base_url=GOOGLE_API_BASE_URL if uri is None else uri,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
        ),
    )
for openai_model in [
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]:
    converter_config_registry.register(
        openai_model,
        lambda uri=None, model=openai_model: OpenAIConverterConfig(
            llm_params=LLMParams(
                model_name=model,
                base_url=None,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        ),
    )

for converter_config_cls in [
    ChandraConverterConfig,
    LightOnOCRConverterConfig,
    DotsOCRConverterConfig,
    PaddleOCRVLConverterConfig,
    NanonetOCR2ConverterConfig,
    HunyuanOCRConverterConfig,
    DeepSeekOCRConverterConfig,
    GraniteDoclingConverterConfig,
    OlmOCRConverterConfig,
]:
    aliases = get_default(converter_config_cls, "aliases") or []
    model_name = get_default(converter_config_cls, "model_name")
    names = [n for n in aliases + [model_name] if isinstance(n, str)]
    for name in names:
        converter_config_registry.register(
            name,
            lambda uri, cls=converter_config_cls: cls(
                llm_params=LLMParams(
                    base_url=uri,
                    model_name=DEFAULT_MODEL_NAME,
                    api_key="",
                )
            ),
        )
for converter_config_cls in [MinerUConverterConfig, DoclingConverterConfig]:
    aliases = get_default(converter_config_cls, "aliases") or []
    model_name = get_default(converter_config_cls, "model_name")
    names = [n for n in aliases + [model_name] if isinstance(n, str)]
    for name in names:
        converter_config_registry.register(
            name,
            lambda uri, cls=converter_config_cls: cls(base_url=uri),
        )
