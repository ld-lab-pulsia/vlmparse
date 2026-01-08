import json
import math
from pathlib import Path
from typing import ClassVar, Literal

from loguru import logger
from PIL import Image
from pydantic import Field

from vlmparse.clients.openai_converter import (
    OpenAIConverterClient,
    OpenAIConverterConfig,
)
from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.data_model.document import BoundingBox, Item, Page
from vlmparse.servers.docker_server import DEFAULT_MODEL_NAME, DockerServerConfig
from vlmparse.utils import to_base64

DOCKERFILE_DIR = Path(__file__).parent.parent.parent / "docker_pipelines"


class DotsOCRDockerServerConfig(DockerServerConfig):
    """Configuration for DotsOCR model."""

    model_name: str = "rednote-hilab/dots.ocr"
    docker_image: str = "dotsocr:latest"
    dockerfile_dir: str = str(DOCKERFILE_DIR / "dotsocr")
    command_args: list[str] = Field(
        default_factory=lambda: [
            "--tensor-parallel-size",
            "1",
            "--gpu-memory-utilization",
            "0.8",
            "--chat-template-content-format",
            "string",
            "--served-model-name",
            DEFAULT_MODEL_NAME,
            "--trust-remote-code",
            # "--limit-mm-per-prompt",
            # '{"image": 1}',
            # "--no-enable-prefix-caching",
            # "--max-model-len",
            # "16384",
        ]
    )
    add_model_key_to_server: bool = False
    aliases: list[str] = Field(default_factory=lambda: ["dotsocr"])

    @property
    def client_config(self):
        return DotsOCRConverterConfig(llm_params=self.llm_params)


class DotsOCRConverterConfig(OpenAIConverterConfig):
    model_name: str = "rednote-hilab/dots.ocr"
    preprompt: str | None = ""
    postprompt: str | None = None
    completion_kwargs: dict | None = {
        "temperature": 0.1,
        "top_p": 1.0,
        "max_completion_tokens": 16384,
    }
    aliases: list[str] = Field(default_factory=lambda: ["dotsocr"])
    dpi: int = 200
    prompt_mode: Literal["prompt_layout_all_en", "prompt_ocr"] = "prompt_ocr"

    def get_client(self, **kwargs) -> "DotsOCRConverter":
        return DotsOCRConverter(config=self, **kwargs)


class DotsOCRConverter(OpenAIConverterClient):
    """DotsOCR VLLM converter."""

    # Constants
    MIN_PIXELS: ClassVar[int] = 3136
    MAX_PIXELS: ClassVar[int] = 11289600
    IMAGE_FACTOR: ClassVar[int] = 28

    # Prompts
    PROMPTS: ClassVar[dict] = {
        "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",
        "prompt_ocr": """Extract the text content from this image.""",
    }

    @staticmethod
    def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    @staticmethod
    def ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    @staticmethod
    def floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 3136,
        max_pixels: int = 11289600,
    ):
        """Rescales image dimensions to meet factor, pixel range, and aspect ratio constraints."""
        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, self.floor_by_factor(height / beta, factor))
            w_bar = max(factor, self.floor_by_factor(width / beta, factor))
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((h_bar * w_bar) / max_pixels)
                h_bar = max(factor, self.floor_by_factor(h_bar / beta, factor))
                w_bar = max(factor, self.floor_by_factor(w_bar / beta, factor))
        return h_bar, w_bar

    def fetch_image(
        self,
        image,
        min_pixels=None,
        max_pixels=None,
    ) -> Image.Image:
        """Fetch and resize image."""
        # Resize if needed
        if min_pixels or max_pixels:
            width, height = image.size
            if not min_pixels:
                min_pixels = self.MIN_PIXELS
            if not max_pixels:
                max_pixels = self.MAX_PIXELS
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=self.IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            assert resized_height > 0 and resized_width > 0
            image = image.resize((resized_width, resized_height))

        return image

    def post_process_cells(
        self,
        origin_image: Image.Image,
        cells: list,
        input_width: int,
        input_height: int,
    ) -> list:
        """Post-process cell bounding boxes to original image dimensions."""
        if not cells or not isinstance(cells, list):
            return cells

        original_width, original_height = origin_image.size

        scale_x = input_width / original_width
        scale_y = input_height / original_height

        cells_out = []
        for cell in cells:
            bbox = cell["bbox"]
            bbox_resized = [
                int(float(bbox[0]) / scale_x),
                int(float(bbox[1]) / scale_y),
                int(float(bbox[2]) / scale_x),
                int(float(bbox[3]) / scale_y),
            ]
            cell_copy = cell.copy()
            cell_copy["bbox"] = bbox_resized
            cells_out.append(cell_copy)

        return cells_out

    async def _async_inference_with_vllm(self, image, prompt):
        """Run async inference with VLLM."""
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
                    {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"},
                ],
            }
        ]

        return await self._get_chat_completion(messages)

    async def _parse_image_vllm(self, origin_image, prompt_mode="prompt_layout_all_en"):
        """Parse image using VLLM inference."""

        image = self.fetch_image(
            origin_image, min_pixels=self.MIN_PIXELS, max_pixels=self.MAX_PIXELS
        )
        prompt = self.PROMPTS[prompt_mode]

        response = await self._async_inference_with_vllm(image, prompt)

        if prompt_mode in ["prompt_layout_all_en"]:
            try:
                cells = json.loads(response)
                cells = self.post_process_cells(
                    origin_image,
                    cells,
                    image.width,
                    image.height,
                )
                return {}, cells, False
            except Exception as e:
                logger.warning(f"cells post process error: {e}, returning raw response")
                return {}, response, True
        else:
            return {}, response, None

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image

        _, response, _ = await self._parse_image_vllm(
            image, prompt_mode=self.config.prompt_mode
        )
        logger.info("Response: " + str(response))

        items = None
        if self.config.prompt_mode == "prompt_layout_all_en":
            text = "\n\n".join([item.get("text", "") for item in response])

            items = []
            for item in response:
                l, t, r, b = item["bbox"]
                items.append(
                    Item(
                        text=item.get("text", ""),
                        box=BoundingBox(l=l, t=t, r=r, b=b),
                        category=item["category"],
                    )
                )
            response = text
            page.items = items

        text = clean_response(response)
        text = html_to_md_keep_tables(text)
        page.text = text
        return page
