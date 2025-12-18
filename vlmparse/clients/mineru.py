import os

from loguru import logger
from pydantic import Field

from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import BoundingBox, Item, Page
from vlmparse.utils import to_base64


class MinerUConverterConfig(ConverterConfig):
    """Configuration for MinerU API converter."""

    api_url: str = Field(
        default_factory=lambda: os.getenv("MINERU_API_URL", "http://localhost:4297")
    )
    timeout: int = 600

    def get_client(self, **kwargs) -> "MinerUConverter":
        return MinerUConverter(config=self, **kwargs)


class MinerUConverter(BaseConverter):
    """MinerU HTTP API converter."""

    config: MinerUConverterConfig

    def __init__(self, config: MinerUConverterConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        from httpx import AsyncClient

        self.client = AsyncClient(base_url=config.api_url, timeout=config.timeout)

    async def _async_inference_with_api(self, image) -> list:
        """Run async inference with MinerU API."""
        response = await self.client.post(
            "process-image",
            json={"image": to_base64(image)},
        )
        response.raise_for_status()
        return response.json()

    async def _parse_image_with_api(self, origin_image):
        response = await self._async_inference_with_api(origin_image)

        original_width, original_height = origin_image.size

        cells_out = []
        for cell in response:
            bbox = cell["bbox"]
            bbox_resized = [
                bbox[0] * original_width,
                bbox[1] * original_height,
                bbox[2] * original_width,
                bbox[3] * original_height,
            ]
            cell_copy = cell.copy()
            cell_copy["bbox"] = bbox_resized
            cells_out.append(cell_copy)
        return cells_out

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
