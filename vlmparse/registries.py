import os
from collections.abc import Callable

from vlmparse.clients.chandra import ChandraDockerServerConfig
from vlmparse.clients.deepseekocr import (
    DeepSeekOCR2DockerServerConfig,
    DeepSeekOCRDockerServerConfig,
)
from vlmparse.clients.docling import DoclingDockerServerConfig
from vlmparse.clients.dotsocr import DotsOCRDockerServerConfig
from vlmparse.clients.glmocr import GLMOCRDockerServerConfig
from vlmparse.clients.granite_docling import GraniteDoclingDockerServerConfig
from vlmparse.clients.hunyuanocr import HunyuanOCRDockerServerConfig
from vlmparse.clients.lightonocr import (
    LightonOCR21BServerConfig,
    LightOnOCRDockerServerConfig,
)
from vlmparse.clients.mineru import MinerUDockerServerConfig
from vlmparse.clients.mistral_converter import MistralOCRConverterConfig
from vlmparse.clients.nanonetocr import NanonetOCR2DockerServerConfig
from vlmparse.clients.olmocr import OlmOCRDockerServerConfig
from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.clients.paddleocrvl import PaddleOCRVLDockerServerConfig
from vlmparse.converter import ConverterConfig
from vlmparse.servers.docker_compose_server import DockerComposeServerConfig
from vlmparse.servers.docker_server import DockerServerConfig
from vlmparse.servers.server_registry import docker_config_registry


def get_default(cls, field_name):
    field_info = cls.model_fields.get(field_name)
    if field_info is None:
        return [] if field_name == "aliases" else None
    if field_info.default_factory:
        return field_info.default_factory()
    return field_info.default


# All server configs - single source of truth
SERVER_CONFIGS: list[type[DockerServerConfig | DockerComposeServerConfig]] = [
    ChandraDockerServerConfig,
    LightOnOCRDockerServerConfig,
    DotsOCRDockerServerConfig,
    PaddleOCRVLDockerServerConfig,
    GLMOCRDockerServerConfig,
    NanonetOCR2DockerServerConfig,
    HunyuanOCRDockerServerConfig,
    DoclingDockerServerConfig,
    OlmOCRDockerServerConfig,
    MinerUDockerServerConfig,
    DeepSeekOCRDockerServerConfig,
    DeepSeekOCR2DockerServerConfig,
    GraniteDoclingDockerServerConfig,
    LightonOCR21BServerConfig,
]

# Register docker server configs
for server_config_cls in SERVER_CONFIGS:
    aliases = get_default(server_config_cls, "aliases") or []
    model_name = get_default(server_config_cls, "model_name")
    names = [n for n in aliases + [model_name] if isinstance(n, str)]
    for name in names:
        docker_config_registry.register(name, lambda cls=server_config_cls: cls())


class ConverterConfigRegistry:
    """Registry for mapping model names to their converter configurations.

    Thread-safe registry that maps model names to their converter configuration factories.
    """

    def __init__(self):
        import threading

        self._registry: dict[str, Callable[[str | None], ConverterConfig]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        model_name: str,
        config_factory: Callable[[str | None], ConverterConfig],
    ):
        """Register a config factory for a model name (thread-safe)."""
        with self._lock:
            self._registry[model_name] = config_factory

    def register_from_server(
        self,
        server_config_cls: type[DockerServerConfig | DockerComposeServerConfig],
    ):
        """Register converter config derived from a server config class.

        This ensures model_name and default_model_name are consistently
        passed from server to client config via _create_client_kwargs.
        """
        aliases = get_default(server_config_cls, "aliases") or []
        model_name = get_default(server_config_cls, "model_name")
        names = [n for n in aliases + [model_name] if isinstance(n, str)]
        # Also register short name (after last /)
        if model_name and "/" in model_name:
            names.append(model_name.split("/")[-1])

        def factory(uri: str | None, cls=server_config_cls) -> ConverterConfig:
            server = cls()
            client_config = server.client_config
            # Override base_url if provided
            if uri is not None:
                client_config = client_config.model_copy(update={"base_url": uri})
            return client_config

        with self._lock:
            for name in names:
                self._registry[name] = factory

    def get(
        self,
        model_name: str,
        uri: str | None = None,
    ) -> ConverterConfig:
        """Get config for a model name (thread-safe). Raises ValueError if not registered."""
        with self._lock:
            factory = self._registry.get(model_name)

        if factory is not None:
            return factory(uri)

        raise ValueError(f"Model '{model_name}' not found in registry.")

    def list_models(self) -> list[str]:
        """List all registered model names (thread-safe)."""
        with self._lock:
            return list(self._registry.keys())


# Global registry instance
converter_config_registry = ConverterConfigRegistry()

# Register all server-backed converters through the server config
# This ensures model_name and default_model_name are consistently passed
for server_config_cls in SERVER_CONFIGS:
    converter_config_registry.register_from_server(server_config_cls)

# External API configs (no server config - these are cloud APIs)
GOOGLE_API_BASE_URL = (
    os.getenv("GOOGLE_API_BASE_URL")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
)


for gemini_model in [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]:
    converter_config_registry.register(
        gemini_model,
        lambda uri=None, model=gemini_model: OpenAIConverterConfig(
            model_name=model,
            base_url=GOOGLE_API_BASE_URL if uri is None else uri,
            api_key=os.getenv("GOOGLE_API_KEY"),
            default_model_name=model,
        ),
    )
for openai_model in [
    "gpt-5.2",
    "gpt-5",
    "gpt-5-mini",
]:
    converter_config_registry.register(
        openai_model,
        lambda uri=None, model=openai_model: OpenAIConverterConfig(
            model_name=model,
            base_url=None,
            api_key=os.getenv("OPENAI_API_KEY"),
            default_model_name=model,
        ),
    )

for mistral_model in ["mistral-ocr-latest", "mistral-ocr"]:
    converter_config_registry.register(
        mistral_model,
        lambda uri=None, model=mistral_model: MistralOCRConverterConfig(
            base_url="https://api.mistral.ai/v1" if uri is None else uri,
            api_key=os.getenv("MISTRAL_API_KEY"),
        ),
    )
