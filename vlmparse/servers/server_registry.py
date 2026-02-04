from typing import Callable

from vlmparse.servers.base_server import BaseServerConfig


class DockerConfigRegistry:
    """Registry for mapping model names to their Docker configurations.

    Thread-safe registry that maps model names to their Docker configuration factories.
    """

    def __init__(self):
        import threading

        self._registry: dict[str, Callable[[], BaseServerConfig | None]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        model_name: str,
        config_factory: Callable[[], BaseServerConfig | None],
    ):
        """Register a config factory for a model name (thread-safe)."""
        with self._lock:
            self._registry[model_name] = config_factory

    def get(self, model_name: str) -> BaseServerConfig | None:
        """Get config for a model name (thread-safe). Returns None if not registered."""
        with self._lock:
            if model_name not in self._registry:
                return None
            factory = self._registry[model_name]
        return factory()

    def list_models(self) -> list[str]:
        """List all registered model names (thread-safe)."""
        with self._lock:
            return list(self._registry.keys())


# Global registry instance
docker_config_registry = DockerConfigRegistry()
