from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig


class DeepSeekOCRDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for DeepSeekOCR model."""

    model_name: str = "deepseek-ai/DeepSeek-OCR"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--async-scheduling",
            "--logits_processors",
            "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb",
            "0",
        ]
    )
    aliases: list[str] = Field(default_factory=lambda: ["deepseekocr"])

    @property
    def client_config(self):
        return DeepSeekOCRConverterConfig(llm_params=self.llm_params)


class DeepSeekOCRConverterConfig(OpenAIConverterConfig):
    """DeepSeekOCR converter - backward compatibility alias."""

    model_name: str = "deepseek-ai/DeepSeek-OCR"
    aliases: list[str] = Field(default_factory=lambda: ["deepseekocr"])
    preprompt: str | None = None
    postprompt: str | None = "<|grounding|>Convert the document to markdown."
    completion_kwargs: dict | None = {
        "temperature": 0.0,
        "extra_body": {
            "skip_special_tokens": False,
            # args used to control custom logits processor
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                # whitelist: <td>, </td>
                "whitelist_token_ids": [128821, 128822],
            },
        },
    }
    max_image_size: int | None = 1540
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["deepseekocr"])
