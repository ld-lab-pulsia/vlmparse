import re

from loguru import logger
from PIL import Image
from pydantic import Field

from vlmparse.clients.openai_converter import (
    OpenAIConverterClient,
    OpenAIConverterConfig,
)
from vlmparse.data_model.box import BoundingBox
from vlmparse.data_model.document import Item, Page
from vlmparse.servers.docker_server import VLLMDockerServerConfig
from vlmparse.utils import to_base64

# ==============================================================================
# DeepSeek-OCR (v1)
# ==============================================================================


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
        return DeepSeekOCRConverterConfig(
            **self._create_client_kwargs(
                f"http://localhost:{self.docker_port}{self.get_base_url_suffix()}"
            )
        )


class DeepSeekOCRConverterConfig(OpenAIConverterConfig):
    """DeepSeekOCR converter - backward compatibility alias."""

    model_name: str = "deepseek-ai/DeepSeek-OCR"
    aliases: list[str] = Field(default_factory=lambda: ["deepseekocr"])
    postprompt: str | None = None
    prompts: dict[str, str] = {
        "layout": "<|grounding|>Convert the document to markdown.",
        "ocr": "Free OCR.",
        "image_description": "Describe this image in detail.",
    }
    prompt_mode_map: dict[str, str] = {
        "ocr_layout": "layout",
        "table": "layout",
    }

    completion_kwargs: dict | None = {
        "temperature": 0.0,
        "max_tokens": 8181,
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
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["deepseekocr"])

    def get_client(self, **kwargs) -> "DeepSeekOCRConverterClient":
        return DeepSeekOCRConverterClient(config=self, **kwargs)


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text):
    try:
        label_type = ref_text[1]
        matches = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", ref_text[2])
        cor_list = [[int(x) for x in m] for m in matches]
    except Exception as e:
        logger.warning(f"Error parsing coordinates: {e}")
        return None

    return (label_type, cor_list)


class DeepSeekOCRConverterClient(OpenAIConverterClient):
    """Client for DeepSeekOCR with specific post-processing."""

    def extract_items(self, image: Image.Image, matches: list) -> list[Item]:
        items = []
        width, height = image.size

        for match in matches:
            # match is tuple: (full_str, label, coords_str)
            result = extract_coordinates_and_label(match)
            if not result:
                continue

            category, coords = result
            if not coords:
                continue

            # Create boxes
            boxes = []
            for point in coords:
                if len(point) != 4:
                    continue
                x1, y1, x2, y2 = point
                # Scale to image size (0-999 -> pixel)
                x1 = (x1 / 999) * width
                y1 = (y1 / 999) * height
                x2 = (x2 / 999) * width
                y2 = (y2 / 999) * height

                boxes.append(
                    BoundingBox(
                        l=min(x1, x2), t=min(y1, y2), r=max(x1, x2), b=max(y1, y2)
                    )
                )

            if not boxes:
                continue

            # Merge if multiple boxes for one item
            try:
                final_box = (
                    BoundingBox.merge_boxes(boxes) if len(boxes) > 1 else boxes[0]
                )
            except Exception as e:
                logger.warning(f"Error merging boxes: {e}")
                continue

            items.append(Item(category=category, text=match[1], box=final_box))

        return items

    async def async_call_inside_page(self, page: Page) -> Page:
        # Prepare messages as in parent class
        assert page.image is not None, "Page image is required for processing"
        image = page.image

        prompt_key = self.get_prompt_key() or "ocr"

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
                    {"type": "text", "text": self.config.prompts[prompt_key]},
                ],
            },
        ]

        # Get raw response using parent's method
        response, usage = await self._get_chat_completion(messages)
        logger.info("Response length: " + str(len(response)))
        page.raw_response = response

        if prompt_key == "layout":
            # Post-processing
            matches, matches_image, matches_other = re_match(response)

            # Extract items (bounding boxes)
            page.items = self.extract_items(page.image, matches)

            # Clean text
            outputs = response

            # Replace image references with a placeholder
            for a_match_image in matches_image:
                outputs = outputs.replace(a_match_image, "![image]")

            # Replace other references (text grounding) and cleanup
            for a_match_other in matches_other:
                outputs = (
                    outputs.replace(a_match_other, "")
                    .replace("\\coloneqq", ":=")
                    .replace("\\eqqcolon", "=:")
                )
        else:
            outputs = response

        page.text = outputs.strip()
        logger.debug(page.text)
        page = self.add_usage(page, usage)
        return page


# ==============================================================================
# DeepSeek-OCR-2
# ==============================================================================


class DeepSeekOCR2DockerServerConfig(VLLMDockerServerConfig):
    """Configuration for DeepSeek-OCR-2 model.

    DeepSeek-OCR-2 uses a custom architecture that requires:
    - Custom model registration via hf_overrides
    - NoRepeatNGram logits processor with specific whitelist tokens
    - Custom image processor (DeepseekOCR2Processor)
    """

    docker_image: str = "vllm/vllm-openai:nightly"
    model_name: str = "deepseek-ai/DeepSeek-OCR-2"
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--hf-overrides",
            '{"architectures": ["DeepseekOCR2ForCausalLM"]}',
            "--block-size",
            "256",
            "--trust-remote-code",
            "--max-model-len",
            "8192",
            "--swap-space",
            "0",
            "--gpu-memory-utilization",
            "0.9",
            "--logits_processors",
            "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
        ]
    )
    aliases: list[str] = Field(
        default_factory=lambda: ["deepseekocr2", "DeepSeek-OCR-2"]
    )

    @property
    def client_config(self):
        return DeepSeekOCR2ConverterConfig(
            **self._create_client_kwargs(
                f"http://localhost:{self.docker_port}{self.get_base_url_suffix()}"
            )
        )


class DeepSeekOCR2ConverterConfig(OpenAIConverterConfig):
    """DeepSeek-OCR-2 converter configuration.

    Key differences from DeepSeek-OCR v1:
    - Uses DeepseekOCR2ForCausalLM architecture
    - Different logits processor parameters (ngram_size=20, window_size=50)
    - Supports cropping mode for image processing
    """

    model_name: str = "deepseek-ai/DeepSeek-OCR-2"
    aliases: list[str] = Field(
        default_factory=lambda: ["deepseekocr2", "DeepSeek-OCR-2"]
    )
    postprompt: str | None = None
    prompts: dict[str, str] = {
        "layout": "<|grounding|>Convert the document to markdown.",
        "ocr": "Free OCR.",
        "image_description": "Describe this image in detail.",
    }
    prompt_mode_map: dict[str, str] = {
        "ocr_layout": "layout",
        "table": "layout",
    }

    completion_kwargs: dict | None = {
        "temperature": 0.0,
        "max_tokens": 8180,
        "extra_body": {
            "skip_special_tokens": False,
            # args used to control custom logits processor
            "vllm_xargs": {
                "ngram_size": 20,
                "window_size": 50,
                # whitelist: <td>, </td>
                "whitelist_token_ids": [128821, 128822],
            },
        },
    }
    dpi: int = 144  # Default DPI used in reference implementation

    def get_client(self, **kwargs) -> "DeepSeekOCR2ConverterClient":
        return DeepSeekOCR2ConverterClient(config=self, **kwargs)


class DeepSeekOCR2ConverterClient(DeepSeekOCRConverterClient):
    """Client for DeepSeek-OCR-2 with specific post-processing.

    Inherits from DeepSeekOCRConverterClient as the post-processing logic
    for parsing grounding references and extracting items is the same.
    The main differences are in the model configuration and logits processor.
    """

    async def async_call_inside_page(self, page: Page) -> Page:
        # Prepare messages as in parent class
        assert page.image is not None, "Page image is required for processing"
        image = page.image

        prompt_key = self.get_prompt_key() or "ocr"

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
                    {"type": "text", "text": self.config.prompts[prompt_key]},
                ],
            },
        ]

        # Get raw response using parent's method
        response, usage = await self._get_chat_completion(messages)
        logger.info("Response length: " + str(len(response)))
        page.raw_response = response

        if prompt_key == "layout":
            # Post-processing
            matches, matches_image, matches_other = re_match(response)

            # Extract items (bounding boxes)
            page.items = self.extract_items(page.image, matches)

            # Clean text
            outputs = response

            # Check for sentence end marker (indicates successful completion)
            # If not present, it might be due to repetition detection
            if "<｜end▁of▁sentence｜>" in outputs:
                outputs = outputs.replace("<｜end▁of▁sentence｜>", "")

            # Replace image references with a placeholder
            for a_match_image in matches_image:
                outputs = outputs.replace(a_match_image, "![image]")

            # Replace other references (text grounding) and cleanup
            for a_match_other in matches_other:
                outputs = (
                    outputs.replace(a_match_other, "")
                    .replace("\\coloneqq", ":=")
                    .replace("\\eqqcolon", "=:")
                    .replace("\n\n\n\n", "\n\n")
                    .replace("\n\n\n", "\n\n")
                )
        else:
            outputs = response

        page.text = outputs.strip()
        logger.debug(page.text)
        page = self.add_usage(page, usage)
        return page
