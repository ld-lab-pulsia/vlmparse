import io
import os

from loguru import logger
from pydantic import Field

from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import Page
from vlmparse.servers.docker_server import DockerServerConfig


class MonkeyOCRDockerServerConfig(DockerServerConfig):
    """Configuration for MonkeyOCR Docker server."""

    model_name: str = "monkeyocr"
    docker_image: str = "monkeyocr"
    docker_port: int = 7861
    container_port: int = 7861

    @property
    def client_config(self):
        return MonkeyOCRConverterConfig(api_url=f"http://localhost:{self.docker_port}")


class MonkeyOCRConverterConfig(ConverterConfig):
    """Configuration for MonkeyOCR API converter."""

    api_url: str = Field(
        default_factory=lambda: os.getenv("MONKEYOCR_API_URL", "http://localhost:7861")
    )
    timeout: int = 600

    def get_client(self, **kwargs) -> "MonkeyOCRConverter":
        return MonkeyOCRConverter(config=self, **kwargs)


class MonkeyOCRConverter(BaseConverter):
    """MonkeyOCR HTTP API converter."""

    config: MonkeyOCRConverterConfig

    def __init__(self, config: MonkeyOCRConverterConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        from httpx import AsyncClient

        self.client = AsyncClient(base_url=config.api_url, timeout=config.timeout)

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # MonkeyOCR /ocr/text endpoint
        files = {"file": ("image.png", img_byte_arr, "image/png")}

        try:
            response = await self.client.post("/ocr/text", files=files)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                content = result.get("content", "")
                page.text = content
            else:
                logger.error(f"MonkeyOCR failed: {result.get('message')}")

        except Exception as e:
            logger.error(f"Error calling MonkeyOCR API: {e}")
            raise

        return page
