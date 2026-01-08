import math
import time

from loguru import logger
from PIL import Image
from pydantic import Field

from vlmparse.clients.openai_converter import (
    OpenAIConverterClient,
    OpenAIConverterConfig,
)
from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.data_model.document import Page
from vlmparse.servers.docker_server import VLLMDockerServerConfig
from vlmparse.utils import to_base64

ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
]
ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in [x0, y0, x1, y1] format.  Bboxes are normalized 0-{{bbox_scale}}. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure

{PROMPT_ENDING}
""".strip()

OCR_PROMPT = f"""
OCR this image to HTML.

{PROMPT_ENDING}
""".strip()

PROMPT_MAPPING = {
    "ocr_layout": OCR_LAYOUT_PROMPT,
    "ocr": OCR_PROMPT,
}


def scale_to_fit(
    img: Image.Image,
    max_size: tuple[int, int] = (3072, 2048),
    min_size: tuple[int, int] = (28, 28),
):
    resample_method = Image.Resampling.LANCZOS
    width, height = img.size
    if width == 0 or height == 0:
        return img
    max_width, max_height = max_size
    min_width, min_height = min_size
    current_pixels = width * height
    max_pixels = max_width * max_height
    min_pixels = min_width * min_height

    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5
        new_width = math.floor(width * scale_factor)
        new_height = math.floor(height * scale_factor)
    elif current_pixels < min_pixels:
        scale_factor = (min_pixels / current_pixels) ** 0.5
        new_width = math.ceil(width * scale_factor)
        new_height = math.ceil(height * scale_factor)
    else:
        return img

    return img.resize((new_width, new_height), resample=resample_method)


def detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
):
    try:
        # Use existing html_to_md_keep_tables from vlmparse
        predicted_tokens = html_to_md_keep_tables(predicted_tokens)
    except Exception as e:
        logger.error(f"Error parsing markdown: {e}")
        return True

    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]

    for seq_len in range(1, window_size // 2 + 1):
        # Extract the potential repeating sequence from the end
        candidate_seq = predicted_tokens[-seq_len:]

        # Inverse scaling: shorter sequences need more repeats
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))

        # Count how many times this sequence appears consecutively at the end
        repeat_count = 0
        pos = len(predicted_tokens) - seq_len
        if pos < 0:
            continue

        while pos >= 0:
            if predicted_tokens[pos : pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break

        if repeat_count > max_repeats:
            return True

    return False


class ChandraConverterConfig(OpenAIConverterConfig):
    """Chandra converter configuration."""

    model_name: str = "datalab-to/chandra"
    prompt_type: str = "ocr"  # Default prompt type
    bbox_scale: int = 1024
    max_retries: int = 6
    max_failure_retries: int = None
    completion_kwargs: dict = Field(
        default_factory=lambda: {
            "temperature": 0.0,
            "max_tokens": 12384,
            "top_p": 0.1,
        }
    )
    aliases: list[str] = Field(default_factory=lambda: ["chandra"])

    def get_client(self, **kwargs) -> "ChandraConverterClient":
        return ChandraConverterClient(config=self, **kwargs)


class ChandraConverterClient(OpenAIConverterClient):
    """Client for Chandra model."""

    config: ChandraConverterConfig

    async def async_call_inside_page(self, page: Page) -> Page:
        """Process a single page using Chandra logic."""

        prompt = PROMPT_MAPPING.get(self.config.prompt_type, OCR_PROMPT)
        prompt = prompt.replace("{bbox_scale}", str(self.config.bbox_scale))

        image = scale_to_fit(page.image)
        image_b64 = to_base64(image)  # vlmparse.utils.to_base64

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        retries = 0
        max_retries = self.config.max_retries
        max_failure_retries = self.config.max_failure_retries

        result_content = ""
        error_occurred = False

        while True:
            try:
                # Adjust temperature if retrying
                temperature = self.config.completion_kwargs.get("temperature", 0.0)
                if retries > 0:
                    temperature = 0.3  # As per vllm.py logic

                completion_kwargs = self.config.completion_kwargs.copy()
                completion_kwargs["temperature"] = temperature
                if retries > 0:
                    completion_kwargs["top_p"] = 0.95

                result_content = await self._get_chat_completion(
                    messages, completion_kwargs=completion_kwargs
                )
                error_occurred = False
            except Exception as e:
                logger.error(f"Error during VLLM generation: {e}")
                error_occurred = True
                result_content = ""

            should_retry = False

            # Check for repeat token
            if not error_occurred:
                has_repeat = detect_repeat_token(result_content) or (
                    len(result_content) > 50
                    and detect_repeat_token(result_content, cut_from_end=50)
                )
                if has_repeat and retries < max_retries:
                    logger.warning(
                        f"Detected repeat token, retrying generation (attempt {retries + 1})..."
                    )
                    should_retry = True

            # Check for error
            if error_occurred:
                if max_failure_retries is not None:
                    if retries < max_failure_retries:
                        logger.warning(
                            f"Detected vllm error, retrying generation (attempt {retries + 1})..."
                        )
                        should_retry = True
                elif (
                    retries < max_retries
                ):  # Fallback to max_retries if max_failure_retries not set (vllm.py logic varies slightly but this is safe)
                    logger.warning(
                        f"Detected vllm error, retrying generation (attempt {retries + 1})..."
                    )
                    should_retry = True

            if should_retry:
                time.sleep(2 * (retries + 1))
                retries += 1
                continue
            else:
                break

        logger.info("Response length: " + str(len(result_content)))
        page.raw_response = result_content
        text = clean_response(result_content)

        # Convert HTML to MD
        text = html_to_md_keep_tables(text)
        page.text = text

        return page


class ChandraDockerServerConfig(VLLMDockerServerConfig):
    """Configuration for Chandra Docker server."""

    model_name: str = "datalab-to/chandra"
    aliases: list[str] = Field(default_factory=lambda: ["chandra"])

    @property
    def client_config(self):
        return ChandraConverterConfig(llm_params=self.llm_params)
