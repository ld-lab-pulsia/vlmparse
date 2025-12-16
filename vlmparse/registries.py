import os
from collections.abc import Callable

from vlmparse.clients.docling import DoclingConverterConfig, DoclingDockerServerConfig
from vlmparse.clients.dotsocr import DotsOCRConverterConfig, DotsOCRDockerServerConfig
from vlmparse.clients.hunyuanocr import (
    HunyuanOCRConverterConfig,
    HunyuanOCRDockerServerConfig,
)
from vlmparse.clients.lightonocr import (
    LightOnOCRConverterConfig,
    LightOnOCRDockerServerConfig,
)
from vlmparse.clients.nanonetocr import (
    NanonetOCR2ConverterConfig,
    NanonetOCR2DockerServerConfig,
)
from vlmparse.clients.openai_converter import LLMParams, OpenAIConverterConfig
from vlmparse.clients.paddleocrvl import (
    PaddleOCRVLConverterConfig,
    PaddleOCRVLDockerServerConfig,
)
from vlmparse.servers.docker_server import DEFAULT_MODEL_NAME, docker_config_registry

docker_config_registry.register("lightonocr", lambda: LightOnOCRDockerServerConfig())
docker_config_registry.register("dotsocr", lambda: DotsOCRDockerServerConfig())
docker_config_registry.register("paddleocrvl", lambda: PaddleOCRVLDockerServerConfig())
docker_config_registry.register(
    "nanonets/Nanonets-OCR2-3B", lambda: NanonetOCR2DockerServerConfig()
)
docker_config_registry.register(
    "tencent/HunyuanOCR", lambda: HunyuanOCRDockerServerConfig()
)
docker_config_registry.register("hunyuanocr", lambda: HunyuanOCRDockerServerConfig())
docker_config_registry.register("docling", lambda: DoclingDockerServerConfig())


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
converter_config_registry.register(
    "lightonocr",
    lambda uri=None: LightOnOCRConverterConfig(
        llm_params=LLMParams(
            base_url=uri or "http://localhost:8000/v1",
            model_name=DEFAULT_MODEL_NAME,
            api_key="",
        )
    ),
)
converter_config_registry.register(
    "dotsocr",
    lambda uri=None: DotsOCRConverterConfig(
        llm_params=LLMParams(
            base_url=uri or "http://localhost:8000/v1",
            model_name=DEFAULT_MODEL_NAME,
            api_key="",
        )
    ),
)
converter_config_registry.register(
    "paddleocrvl",
    lambda uri=None: PaddleOCRVLConverterConfig(
        llm_params=LLMParams(
            base_url=uri or "http://localhost:8000/v1",
            model_name=DEFAULT_MODEL_NAME,
            api_key="",
        )
    ),
)
converter_config_registry.register(
    "nanonets/Nanonets-OCR2-3B",
    lambda uri=None: NanonetOCR2ConverterConfig(
        llm_params=LLMParams(
            base_url=uri or "http://localhost:8000/v1",
            model_name=DEFAULT_MODEL_NAME,
            api_key="",
        )
    ),
)
converter_config_registry.register(
    "nanonets/Nanonets-OCR2-3B",
    lambda uri=None: NanonetOCR2ConverterConfig(
        llm_params=LLMParams(
            base_url=uri or "http://localhost:8000/v1",
            model_name=DEFAULT_MODEL_NAME,
            api_key="",
        )
    ),
)
converter_config_registry.register(
    "hunyuanocr",
    lambda uri=None: HunyuanOCRConverterConfig(
        llm_params=LLMParams(
            base_url=uri or "http://localhost:8000/v1",
            model_name=DEFAULT_MODEL_NAME,
            api_key="",
        )
    ),
)
converter_config_registry.register(
    "docling",
    lambda uri=None: DoclingConverterConfig(base_url=uri or "http://localhost:5001"),
)
