from pydantic import Field

from vlmparse.clients.dotsocr import (
    DotsOCRConverter,
    DotsOCRConverterConfig,
    DotsOCRDockerServerConfig,
)
from vlmparse.servers.docker_server import DEFAULT_MODEL_NAME
from vlmparse.utils import to_base64


class DotsMOCRDockerServerConfig(DotsOCRDockerServerConfig):
    """Configuration for DotsMOCR model."""

    model_name: str = "rednote-hilab/dots.mocr"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--tensor-parallel-size",
            "1",
            "--gpu-memory-utilization",
            "0.95",
            "--chat-template-content-format",
            "string",
            "--served-model-name",
            DEFAULT_MODEL_NAME,
            "--trust-remote-code",
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--no-enable-prefix-caching",
            "--max-model-len",
            "32768",
        ]
    )
    aliases: list[str] = Field(default_factory=lambda: ["dotsmocr"])

    @property
    def client_config(self):
        return DotsMOCRConverterConfig(
            **self._create_client_kwargs(
                f"http://localhost:{self.docker_port}{self.get_base_url_suffix()}"
            )
        )


class DotsMOCRConverterConfig(DotsOCRConverterConfig):
    model_name: str = "rednote-hilab/dots.mocr"
    completion_kwargs: dict | None = {
        "temperature": 0.1,
        "top_p": 1.0,
        "max_completion_tokens": 32600,
    }
    aliases: list[str] = Field(default_factory=lambda: ["dotsmocr"])

    def get_client(self, **kwargs) -> "DotsMOCRConverter":
        return DotsMOCRConverter(config=self, **kwargs)


class DotsMOCRConverter(DotsOCRConverter):
    """DotsMOCR VLLM converter. Unlike DotsOCR, does not use special image tokens."""

    async def _async_inference_with_vllm(self, image, prompt):
        """Run async inference with VLLM (no special image token prefix)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{to_base64(image)}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        return await self._get_chat_completion(messages)
