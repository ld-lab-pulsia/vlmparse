from loguru import logger
from .utils import vllm_server

import os

from pydantic import BaseModel, Field
from typing import Callable


class DockerServerConfig(BaseModel):
    """Configuration for deploying a Docker server."""
    
    model_name: str
    docker_image: str = "vllm/vllm-openai:latest"
    dockerfile_dir: str|None = None
    command_args: list[str] = Field(default_factory=list)
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: ["Application startup complete", "Uvicorn running"]
    )
    docker_port: int = 8056
    gpu_device_ids: list[str] | None = None
    default_model_name: str="vllm-model"
    hf_home_folder: str|None = os.getenv("HF_HOME", None)
    add_model_key_to_vllm_server: bool = True
    
    class Config:
        extra = "allow"

    @property
    def llm_params(self):
        from benchdocparser.clients.openai_converter import LLMParams
        return LLMParams(base_url=f"http://localhost:{self.docker_port}/v1", model_name=self.default_model_name)

    @property
    def client_config(self):
        from benchdocparser.clients.openai_converter import OpenAIConverterConfig
        return OpenAIConverterConfig(llm_params=self.llm_params)
        
    def get_client(self):
        return self.client_config.get_client()

    def get_server(self, auto_stop: bool = True):
        return ConverterServer(config=self, auto_stop=auto_stop)


class ConverterServer:
    """Manages VLLM server lifecycle with start/stop methods."""
    
    def __init__(self, config: DockerServerConfig, auto_stop: bool = True):
        self.config = config
        self.auto_stop = auto_stop
        self._server_context = None
        self._container = None
        self.base_url = None
        
    def start(self):
        """Start the VLLM server."""
        if self._server_context is not None:
            logger.warning("Server already started")
            return self.base_url
            
        self._server_context = vllm_server(config=self.config, cleanup=self.auto_stop)
        self.base_url, self._container = self._server_context.__enter__()
        logger.info(f"Server started at {self.base_url}")
        logger.info(f"Container ID: {self._container.id}")
        logger.info(f"Container name: {self._container.name}")
        return self.base_url, self._container
    
    def stop(self):
        """Stop the VLLM server."""
        if self._server_context is not None:
            self._server_context.__exit__(None, None, None)
            self._server_context = None
            self._container = None
            self.base_url = None
            logger.info("Server stopped")
    
    def __del__(self):
        """Automatically stop server when object is destroyed if auto_stop is True."""
        if self.auto_stop and self._server_context is not None:
            self.stop()

class DockerConfigRegistry:
    """Registry for mapping model names to their Docker configurations."""
    
    def __init__(self):
        self._registry = dict()

    def register(self, model_name: str, config_factory: Callable[[], DockerServerConfig | None]):
        """Register a config factory for a model name."""
        self._registry[model_name] = config_factory
    
    def get(self, model_name: str) -> DockerServerConfig | None:
        """Get config for a model name. Returns default if not registered."""
        if model_name not in self._registry:
            return DockerServerConfig(model_name=model_name)
        return self._registry[model_name]()


# Global registry instance
docker_config_registry = DockerConfigRegistry()
