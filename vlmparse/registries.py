from vlmparse.clients.nanonetocr import NanonetOCR2ConverterConfig, NanonetOCR2DockerServerConfig
from vlmparse.servers.docker_server import docker_config_registry
from vlmparse.clients.lightonocr import LightOnOCRConverterConfig, LightOnOCRDockerServerConfig
from vlmparse.clients.dotsocr import DotsOCRConverterConfig, DotsOCRDockerServerConfig
from vlmparse.clients.docling import DoclingConverterConfig, DoclingDockerServerConfig
from vlmparse.clients.openai_converter import LLMParams, OpenAIConverterConfig
import os
from collections.abc import Callable
import os




docker_config_registry.register("lightonocr", lambda: LightOnOCRDockerServerConfig())
docker_config_registry.register("dotsocr", lambda: DotsOCRDockerServerConfig())
docker_config_registry.register("nanonets/Nanonets-OCR2-3B", lambda: NanonetOCR2DockerServerConfig())
docker_config_registry.register("docling", lambda: DoclingDockerServerConfig())
docker_config_registry.register("gemini-2.5-flash-lite", lambda: None)
docker_config_registry.register("gemini-2.5-flash", lambda: None)
docker_config_registry.register("gemini-2.5-pro", lambda: None)




class ConverterConfigRegistry:
    """Registry for mapping model names to their Docker configurations."""
    
    def __init__(self):
        self._registry = dict()

    def register(self, model_name: str, config_factory: Callable[[str], OpenAIConverterConfig | None]):
        """Register a config factory for a model name."""
        self._registry[model_name] = config_factory
    
    def get(self, model_name: str, uri: str | None = None) -> OpenAIConverterConfig | None:
        """Get config for a model name. Returns default if not registered."""
        if uri is not None:
            return OpenAIConverterConfig(llm_params=LLMParams(base_url=uri, model_name=model_name))
        if model_name not in self._registry:
            return OpenAIConverterConfig(llm_params=LLMParams(model_name=model_name))
        return self._registry[model_name](uri=uri)


# Global registry instance
converter_config_registry = ConverterConfigRegistry()



converter_config_registry.register("gemini-2.5-flash-lite", lambda uri=None: OpenAIConverterConfig(llm_params=LLMParams(model_name="gemini-2.5-flash-lite", base_url=os.getenv("GOOGLE_API_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/")))
converter_config_registry.register("gemini-2.5-flash", lambda uri=None: OpenAIConverterConfig(llm_params=LLMParams(model_name="gemini-2.5-flash", base_url=os.getenv("GOOGLE_API_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/")))
converter_config_registry.register("gemini-2.5-pro", lambda uri=None: OpenAIConverterConfig(llm_params=LLMParams(model_name="gemini-2.5-pro", base_url=os.getenv("GOOGLE_API_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/")))
converter_config_registry.register("lightonocr", lambda uri=None: LightOnOCRConverterConfig(llm_params=LLMParams(base_url=uri or "http://localhost:8000/v1", model_name="lightonai/LightOnOCR-1B-1025", api_key="")))
converter_config_registry.register("dotsocr", lambda uri=None: DotsOCRConverterConfig(llm_params=LLMParams(base_url=uri or "http://localhost:8000/v1", model_name="dotsocr-model", api_key="")))
converter_config_registry.register("nanonets/Nanonets-OCR2-3B", lambda uri=None: NanonetOCR2ConverterConfig(llm_params=LLMParams(base_url=uri or "http://localhost:8000/v1", model_name="nanonets/Nanonets-OCR2-3B", api_key="")))
converter_config_registry.register("docling", lambda uri=None: DoclingConverterConfig(base_url=uri or "http://localhost:5001"))



