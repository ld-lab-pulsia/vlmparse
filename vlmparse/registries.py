import os
from collections.abc import Callable

from vlmparse.clients.chandra import (
    Chandra2DockerServerConfig,
    ChandraDockerServerConfig,
)
from vlmparse.clients.deepseekocr import (
    DeepSeekOCR2DockerServerConfig,
    DeepSeekOCRDockerServerConfig,
)
from vlmparse.clients.docling import DoclingDockerServerConfig
from vlmparse.clients.dotsmocr import DotsMOCRDockerServerConfig
from vlmparse.clients.dotsocr import (
    DotsOCR1p5DockerServerConfig,
    DotsOCRDockerServerConfig,
)
from vlmparse.clients.fireredocr import FireRedOCRDockerServerConfig
from vlmparse.clients.glmocr import GLMOCRDockerServerConfig
from vlmparse.clients.glmocr_vlmparse import GLMOCRVlmparseDockerServerConfig
from vlmparse.clients.granite_docling import GraniteDoclingDockerServerConfig
from vlmparse.clients.hunyuanocr import HunyuanOCRDockerServerConfig
from vlmparse.clients.lightonocr import (
    LightonOCR21BboxServerConfig,
    LightonOCR21BServerConfig,
    LightOnOCRDockerServerConfig,
)
from vlmparse.clients.mineru import MinerUDockerServerConfig
from vlmparse.clients.mistral_converter import MistralOCRConverterConfig
from vlmparse.clients.nanonetocr import NanonetOCR2DockerServerConfig
from vlmparse.clients.ocrverse import OCRVerseDockerServerConfig
from vlmparse.clients.olmocr import OlmOCRDockerServerConfig
from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.clients.paddleocrvl import PaddleOCRVLDockerServerConfig
from vlmparse.clients.qianfanocr import (
    QianfanOCRDockerServerConfig,
    QianfanOCRThinkingDockerServerConfig,
)
from vlmparse.converter import ConverterConfig
from vlmparse.model_endpoint_config import ModelEndpointConfig
from vlmparse.servers.container_group_server import ContainerGroupServerConfig
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
SERVER_CONFIGS: list[
    type[DockerServerConfig | DockerComposeServerConfig | ContainerGroupServerConfig]
] = [
    ChandraDockerServerConfig,
    Chandra2DockerServerConfig,
    LightOnOCRDockerServerConfig,
    DotsOCRDockerServerConfig,
    PaddleOCRVLDockerServerConfig,
    GLMOCRDockerServerConfig,
    GLMOCRVlmparseDockerServerConfig,
    NanonetOCR2DockerServerConfig,
    HunyuanOCRDockerServerConfig,
    DoclingDockerServerConfig,
    OlmOCRDockerServerConfig,
    MinerUDockerServerConfig,
    DeepSeekOCRDockerServerConfig,
    DeepSeekOCR2DockerServerConfig,
    GraniteDoclingDockerServerConfig,
    LightonOCR21BServerConfig,
    LightonOCR21BboxServerConfig,
    DotsOCR1p5DockerServerConfig,
    DotsMOCRDockerServerConfig,
    FireRedOCRDockerServerConfig,
    OCRVerseDockerServerConfig,
    QianfanOCRDockerServerConfig,
    QianfanOCRThinkingDockerServerConfig,
]

# Register docker server configs
for server_config_cls in SERVER_CONFIGS:
    aliases = get_default(server_config_cls, "aliases") or []
    model_name = get_default(server_config_cls, "model_name")
    names = [n for n in aliases + [model_name] if isinstance(n, str)]
    for name in names:
        docker_config_registry.register(
            name,
            lambda cls=server_config_cls: cls(),  # ty: ignore[missing-argument]
        )


class ConverterConfigRegistry:
    """Registry for mapping model names to their converter configurations.

    Thread-safe registry that maps (model_name, provider) pairs to their
    converter configuration factories. Supports multiple providers per model.
    """

    DEFAULT_PROVIDER = "registry"

    def __init__(self):
        import threading

        self._registry: dict[
            str, dict[str, Callable[[str | None], ConverterConfig]]
        ] = {}
        self._lock = threading.RLock()

    def register(
        self,
        model_name: str,
        config_factory: Callable[[str | None], ConverterConfig],
        provider: str = DEFAULT_PROVIDER,
    ):
        """Register a config factory for a model name and provider (thread-safe)."""
        with self._lock:
            self._registry.setdefault(model_name, {})[provider] = config_factory

    def register_from_server(
        self,
        server_config_cls: type[
            DockerServerConfig | DockerComposeServerConfig | ContainerGroupServerConfig
        ],
        provider: str = DEFAULT_PROVIDER,
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
            server = cls()  # ty: ignore
            if uri is not None:
                return server.client_config_for_uri(uri)
            return server.client_config

        with self._lock:
            for name in names:
                self._registry.setdefault(name, {})[provider] = factory

    def get(
        self,
        model_name: str,
        uri: str | None = None,
        provider: str | None = None,
    ) -> ConverterConfig:
        """Get config for a model name (thread-safe). Raises ValueError if not registered.

        If provider is None and only one provider exists, returns that one.
        If multiple providers exist and none is specified, raises ValueError.
        """
        with self._lock:
            providers = self._registry.get(model_name)

            if providers is None:
                raise ValueError(f"Model '{model_name}' not found in registry.")

            if provider is not None:
                factory = providers.get(provider)
                if factory is None:
                    raise ValueError(
                        f"Provider '{provider}' not found for model '{model_name}'. "
                        f"Available providers: {list(providers.keys())}"
                    )
            elif len(providers) == 1:
                factory = next(iter(providers.values()))
            elif self.DEFAULT_PROVIDER in providers:
                factory = providers[self.DEFAULT_PROVIDER]
            else:
                factory = next(iter(providers.values()))

        return factory(uri)

    def list_models(self) -> list[str]:
        """List all registered model names (thread-safe)."""
        with self._lock:
            return list(self._registry.keys())

    def list_providers(self, model_name: str) -> list[str]:
        """List all providers for a given model name (thread-safe)."""
        with self._lock:
            providers = self._registry.get(model_name)
            return list(providers.keys()) if providers else []


# Global registry instance
converter_config_registry = ConverterConfigRegistry()

# Register all server-backed converters through the server config
# This ensures model_name and default_model_name are consistently passed
for server_config_cls in SERVER_CONFIGS:
    converter_config_registry.register_from_server(server_config_cls)

# External API configs (no server config - these are cloud APIs)
GOOGLE_API_BASE_URL = os.getenv(
    "GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
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
            inline_image_description=True,
            endpoint=ModelEndpointConfig(
                base_url=GOOGLE_API_BASE_URL if uri is None else uri,
                api_key=os.getenv("GOOGLE_API_KEY", ""),
                model_name=model,
            ),
        ),
        provider="google",
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
            inline_image_description=True,
            endpoint=ModelEndpointConfig(
                base_url=None,
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model_name=model,
            ),
        ),
        provider="openai",
    )

for mistral_model in ["mistral-ocr-latest", "mistral-ocr"]:
    converter_config_registry.register(
        mistral_model,
        lambda uri=None, model=mistral_model: MistralOCRConverterConfig(
            base_url="https://api.mistral.ai/v1" if uri is None else uri,
            api_key=os.getenv("MISTRAL_API_KEY", ""),
        ),
        provider="mistral",
    )


# -- Generic provider factories (hf, google, openai, azure) --------------------
# These allow creating configs for *any* model name via a provider, even if not
# pre-registered, and are used by get_client_config() to eliminate the provider if/elif.


def _make_hf_factory(model: str, uri: str | None) -> ConverterConfig:
    return OpenAIConverterConfig(
        model_name=model,
        endpoint=ModelEndpointConfig(base_url=uri),
    )


def _make_google_factory(
    model: str,
    uri: str | None,
    api_key: str | None = None,
) -> ConverterConfig:
    api_key = api_key if api_key is not None else os.getenv("GOOGLE_API_KEY", "")
    return OpenAIConverterConfig(
        model_name=model,
        inline_image_description=True,
        endpoint=ModelEndpointConfig(
            base_url=GOOGLE_API_BASE_URL if uri is None else uri,
            api_key=api_key,
            model_name=model,
        ),
    )


def _make_openai_factory(
    model: str,
    uri: str | None,
    api_key: str | None = None,
) -> ConverterConfig:
    api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
    return OpenAIConverterConfig(
        model_name=model,
        inline_image_description=True,
        endpoint=ModelEndpointConfig(
            base_url=uri,
            api_key=api_key,
            model_name=model,
        ),
    )


def _make_azure_factory(
    model: str,
    uri: str | None,
    api_key: str | None = None,
    use_response_api: bool = False,
) -> ConverterConfig:
    api_key = api_key if api_key is not None else os.getenv("AZURE_OPENAI_API_KEY", "")
    return OpenAIConverterConfig(
        model_name=model,
        endpoint=ModelEndpointConfig(
            base_url=uri if uri is not None else os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key,
            model_name=model,
        ),
        is_azure=True,
        use_response_api=use_response_api,
    )
