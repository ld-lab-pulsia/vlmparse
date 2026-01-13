import re
from typing import ClassVar, Literal

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

    PROMPTS: ClassVar[dict] = {
        "layout": "<|grounding|>Convert the document to markdown.",
        "ocr": "Free OCR.",
        "image_description": "Describe this image in detail.",
    }

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
        image = page.image

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
                    {"type": "text", "text": self.PROMPTS[self.config.prompt_mode]},
                ],
            },
        ]

        # Get raw response using parent's method
        response = await self._get_chat_completion(messages)
        logger.info("Response length: " + str(len(response)))
        page.raw_response = response

        if self.config.prompt_mode == "layout":
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

        return page


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

    prompt_mode: Literal["layout", "ocr"] = "ocr"
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
