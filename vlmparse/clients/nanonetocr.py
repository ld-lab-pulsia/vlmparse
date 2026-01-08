from pydantic import Field

from vlmparse.clients.openai_converter import OpenAIConverterConfig
from vlmparse.servers.docker_server import VLLMDockerServerConfig


class NanonetOCR2DockerServerConfig(VLLMDockerServerConfig):
    """Configuration for NanonetOCR2 model."""

    model_name: str = "nanonets/Nanonets-OCR2-3B"
    aliases: list[str] = Field(default_factory=lambda: ["nanonetsocr2"])

    @property
    def client_config(self):
        return NanonetOCR2ConverterConfig(llm_params=self.llm_params)


class NanonetOCR2ConverterConfig(OpenAIConverterConfig):
    """Configuration for NanonetOCR2 model."""

    model_name: str = "nanonets/Nanonets-OCR2-3B"
    preprompt: str | None = None
    postprompt: str | None = (
        "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
    )
    completion_kwargs: dict | None = {"temperature": 0.0, "max_tokens": 15000}
    max_image_size: int | None = None
    dpi: int = 200
    aliases: list[str] = Field(default_factory=lambda: ["nanonetsocr2"])
