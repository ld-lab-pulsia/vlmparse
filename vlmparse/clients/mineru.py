import asyncio
import io
import os

import orjson
from loguru import logger
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import BoundingBox, Item, Page
from vlmparse.servers.docker_server import DockerServerConfig


class MinerUDockerServerConfig(DockerServerConfig):
    """Configuration for MinerU Docker server."""

    model_name: str = "mineru25"
    docker_image: str = "pulsia/mineru25apipulsia:latest"
    docker_port: int = 4299
    container_port: int = 8000

    @property
    def client_config(self):
        return MinerUConverterConfig(api_url=f"http://localhost:{self.docker_port}")


class MinerUConverterConfig(ConverterConfig):
    """Configuration for MinerU API converter."""

    base_url: str = Field(
        default_factory=lambda: os.getenv("MINERU_API_URL", "http://localhost:4299")
    )
    timeout: int = 600

    def get_client(self, **kwargs) -> "MinerUConverter":
        return MinerUConverter(config=self, **kwargs)


def to_bytes_io(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


class MinerUConverter(BaseConverter):
    """MinerU HTTP API converter."""

    config: MinerUConverterConfig

    def __init__(self, config: MinerUConverterConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        from httpx import AsyncClient

        self.client = AsyncClient(base_url=config.api_url, timeout=config.timeout)

    async def _async_inference_with_api(self, image) -> list:
        """Run async inference with MinerU API."""

        img_byte_arr = await asyncio.to_thread(to_bytes_io, image)

        response = await self.client.post(
            "process-image",
            files={"image": ("image.png", img_byte_arr, "image/png")},
        )

        response.raise_for_status()

        res = orjson.loads(response.content)

        return res

    async def _parse_image_with_api(self, origin_image):
        response = await self._async_inference_with_api(origin_image)

        original_width, original_height = origin_image.size

        for cell in response:
            bbox = cell["bbox"]
            bbox_resized = [
                bbox[0] * original_width,
                bbox[1] * original_height,
                bbox[2] * original_width,
                bbox[3] * original_height,
            ]

            cell["bbox"] = bbox_resized

        return response

    async def async_call_inside_page(self, page: Page) -> Page:
        image = page.image

        # Call MinerU API
        response = await self._parse_image_with_api(image)
        logger.info("Response: " + str(response))

        contents = [item.get("content", "") for item in response]
        text = "\n\n".join([content for content in contents if content is not None])
        items = []
        for item in response:
            l, t, r, b = item["bbox"]
            txt = item.get("content", "")

            items.append(
                Item(
                    text=txt if txt is not None else "",
                    box=BoundingBox(l=l, t=t, r=r, b=b),
                    category=item["type"],
                )
            )
            page.items = items

        text = clean_response(text)
        text = html_to_md_keep_tables(text)
        page.text = text
        return page
