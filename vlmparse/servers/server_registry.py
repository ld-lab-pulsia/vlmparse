from typing import Callable

from vlmparse.servers.base_server import BaseServerConfig

DEFAULT_PROVIDER = "registry"


class DockerConfigRegistry:
    """Registry for mapping model names to their Docker configurations.

    Thread-safe registry that maps model names to their Docker configuration factories.
    Supports multiple providers per model name.
    """

    def __init__(self):
        import threading

        self._registry: dict[str, dict[str, Callable[[], BaseServerConfig | None]]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        model_name: str,
        config_factory: Callable[[], BaseServerConfig | None],
        provider: str = DEFAULT_PROVIDER,
    ):
        """Register a config factory for a model name and provider (thread-safe)."""
        with self._lock:
            self._registry.setdefault(model_name, {})[provider] = config_factory

    def get(
        self,
        model_name: str,
        provider: str | None = None,
    ) -> BaseServerConfig | None:
        """Get config for a model name (thread-safe). Returns None if not registered.

        If provider is None and only one provider exists, returns that one.
        If multiple providers exist and none is specified, raises ValueError.
        """
        with self._lock:
            providers = self._registry.get(model_name)
            if providers is None:
                return None
            if provider is not None:
                factory = providers.get(provider)
                if factory is None:
                    return None
            elif len(providers) == 1:
                factory = next(iter(providers.values()))
            else:
                raise ValueError(
                    f"Multiple providers for model '{model_name}': "
                    f"{list(providers.keys())}. Specify a provider."
                )
        return factory()

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
docker_config_registry = DockerConfigRegistry()
