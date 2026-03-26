"""GLM-OCR Vlmparse client: layout detection (LitServe) + vLLM recognition."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from vlmparse.clients.layout_vlm_converter import (
    LayoutVLMConverter,
    LayoutVLMConverterConfig,
)
from vlmparse.servers.container_group_server import (
    ContainerGroupServerConfig,
    ServiceDefinition,
)

DOCKER_PIPELINE_V2_DIR = (
    Path(__file__).parent.parent / "docker_pipelines" / "glmocr-vlmparse"
)

# Label → task type mapping from GLM-OCR config.yaml (label_task_mapping)
_LABEL_TASK_MAP: dict[str, str] = {
    # OCR as text
    "abstract": "text",
    "algorithm": "text",
    "content": "text",
    "doc_title": "text",
    "figure_title": "text",
    "paragraph_title": "text",
    "reference_content": "text",
    "text": "text",
    "vertical_text": "text",
    "vision_footnote": "text",
    "seal": "text",
    "formula_number": "text",
    "header": "text",
    "footer": "text",
    "number": "text",
    "footnote": "text",
    "aside_text": "text",
    "reference": "text",
    # OCR as table
    "table": "table",
    # OCR as formula
    "display_formula": "formula",
    "inline_formula": "formula",
    # Keep region, skip OCR
    "chart": "skip",
    "image": "skip",
    "footer_image": "skip",
    "header_image": "skip",
}

# Task-specific prompts from GLM-OCR config.yaml (task_prompt_mapping)
_TASK_PROMPTS: dict[str, str] = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


class GLMOCRVlmparseConverterConfig(LayoutVLMConverterConfig):
    model_name: str = "GLM-OCR-vlmparse"
    aliases: list[str] = Field(
        default_factory=lambda: ["glmocr-vlmparse", "glm-ocr-vlmparse"]
    )
    supported_modes: list[str] = Field(default_factory=lambda: ["ocr_layout"])
    max_tokens: int = 4096
    temperature: float = 0.8
    top_p: float | None = 0.9
    top_k: int | None = 50
    repetition_penalty: float | None = 1.1
    timeout: int = 600

    def get_client(self, **kwargs) -> "GLMOCRVlmparseConverter":
        return GLMOCRVlmparseConverter(config=self, **kwargs)


class GLMOCRVlmparseConverter(LayoutVLMConverter):
    config: GLMOCRVlmparseConverterConfig

    def _get_label_task(self, label: str) -> str:
        return _LABEL_TASK_MAP.get(label, "text")

    def _get_prompt(self, label: str) -> str | None:
        task = _LABEL_TASK_MAP.get(label, "text")
        return _TASK_PROMPTS.get(task)


class GLMOCRVlmparseDockerServerConfig(ContainerGroupServerConfig):
    model_name: str = "GLM-OCR-vlmparse"
    aliases: list[str] = Field(
        default_factory=lambda: ["glmocr-vlmparse", "glm-ocr-vlmparse"]
    )
    server_service: str = "layout-server"
    docker_port: int = 8090
    gpu_services: list[str] = Field(
        default_factory=lambda: ["vllm-server", "layout-server"]
    )
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: [
            "Your LitServe app is ready",
            "Uvicorn running",
            "Application startup complete",
        ]
    )
    services: dict[str, ServiceDefinition] = Field(default_factory=dict)

    def model_post_init(self, __context):
        hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        vlm_model_id = os.getenv("VLM_MODEL_ID", "zai-org/GLM-OCR")
        layout_model_dir = os.getenv(
            "LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
        )
        layout_threshold = os.getenv("LAYOUT_THRESHOLD", "0.3")
        cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")

        self.services = {
            "vllm-server": ServiceDefinition(
                image="vlmparse-glmocr-vllm-server:latest",
                dockerfile_dir=DOCKER_PIPELINE_V2_DIR / "vllm-server",
                internal_port=8080,
                ready_indicators=["Application startup complete."],
                command=[
                    vlm_model_id,
                    "--served-model-name",
                    "default",
                    "--allowed-local-media-path",
                    "/",
                    "--port",
                    "8080",
                    "--speculative-config",
                    '{"method": "mtp", "num_speculative_tokens": 1}',
                    "--gpu-memory-utilization",
                    "0.8",
                ],
                environment={
                    "CUDA_VISIBLE_DEVICES": cuda_devices,
                    "HF_HOME": "/root/.cache/huggingface",
                    "TORCH_USE_CUDA_DSA": "1",
                },
                volumes={hf_home: {"bind": "/root/.cache/huggingface", "mode": "rw"}},
                shm_size="16g",
            ),
            "layout-server": ServiceDefinition(
                image="vlmparse-glmocr-layout-server:latest",
                dockerfile_dir=DOCKER_PIPELINE_V2_DIR / "layout-server",
                internal_port=8090,
                command=["python", "/app/layout_server.py"],
                environment={
                    "LAYOUT_MODEL_DIR": layout_model_dir,
                    "LAYOUT_PORT": "8090",
                    "LAYOUT_THRESHOLD": layout_threshold,
                    "HF_HOME": "/root/.cache/huggingface",
                },
                volumes={
                    str(
                        DOCKER_PIPELINE_V2_DIR / "layout-server" / "layout_server.py"
                    ): {
                        "bind": "/app/layout_server.py",
                        "mode": "ro",
                    },
                    hf_home: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                },
                shm_size="8g",
            ),
        }

    @property
    def client_config(self) -> GLMOCRVlmparseConverterConfig:
        return self._client_config_from_service_urls(
            {
                name: f"http://localhost:{self.get_host_port(name)}"
                for name in self.services
            }
        )

    def _client_config_from_service_urls(
        self, service_urls: dict[str, str]
    ) -> GLMOCRVlmparseConverterConfig:
        return GLMOCRVlmparseConverterConfig(
            **self._create_client_kwargs(service_urls["layout-server"]),
            vlm_base_url=service_urls["vllm-server"],
        )
