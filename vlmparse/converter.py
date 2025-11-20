import asyncio

import nest_asyncio
from loguru import logger
from pathlib import Path
import traceback

from .base_model import BenchDocParserBaseModel
from .data_model.document import Document, Page, ProcessingError

from .build_doc import convert_pdfium_to_images

nest_asyncio.apply()

class ConverterConfig(BenchDocParserBaseModel):
    dpi: int = 175
    max_image_size: int | None = None


    def get_client(self, **kwargs) -> 'BaseConverter':
        return BaseConverter(config=self, **kwargs)


class BaseConverter:
    def __init__(self, config: ConverterConfig,     num_concurrent_files: int = 10,
    num_concurrent_pages: int = 10,
    save_folder: str|None=None,
    debug: bool = False):
        self.config = config
        self.num_concurrent_files = num_concurrent_files
        self.num_concurrent_pages = num_concurrent_pages
        self.save_folder = save_folder
        self.debug = debug


    async def async_call_inside_page(self, page: Page) -> Page:
        raise NotImplementedError

    async def async_call(self, file_path: str|Path) -> Document:

        document = Document(file_path=str(file_path))
        try:
            images = convert_pdfium_to_images(file_path, dpi=self.config.dpi)

            new_images = []
            if self.config.max_image_size is not None:
                for image in images:
                    ratio = self.config.max_image_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size)
                    logger.info(f"Resized image to {new_size}")
                    new_images.append(image)
            else:
                new_images = images

            document.pages = [Page(buffer_image=image) for image in new_images]

            # Process pages concurrently with semaphore
            semaphore = asyncio.Semaphore(self.num_concurrent_pages)

            async def worker(page: Page):
                async with semaphore:
                    logger.info("Image size: " + str(page.image.size))
                    try:
                        await self.async_call_inside_page(page)
                    except Exception:
                        if self.debug:
                            raise
                        else:
                            logger.exception(traceback.format_exc())
                            page.error = ProcessingError.from_class(self)

            tasks = [asyncio.create_task(worker(page)) for page in document.pages]
            await asyncio.gather(*tasks)
        except Exception:
            if self.debug:
                raise
            else:
                logger.exception(traceback.format_exc())
                document.error = ProcessingError.from_class(self)
                return document
        
        if self.save_folder is not None:
            save_folder = Path(self.save_folder)
            save_folder.mkdir(parents=True, exist_ok=True)
            document.to_zip(save_folder / (Path(document.file_path).name + ".zip"))
        
        return document


    def __call__(self, file_path: str|Path):
        return asyncio.run(self.async_call(file_path))


    async def async_batch(self, file_paths: list[str|Path]) -> list[Document]:
        """Process multiple files concurrently with semaphore limit."""
        semaphore = asyncio.Semaphore(self.num_concurrent_files)

        async def worker(file_path: str|Path) -> Document:
            async with semaphore:
                return await self.async_call(file_path)

        tasks = [asyncio.create_task(worker(file_path)) for file_path in file_paths]
        documents = await asyncio.gather(*tasks)
        return documents

    def batch(self, file_paths: list[str|Path]) -> list[Document]:
        """Synchronous wrapper for async_batch."""
        return asyncio.run(self.async_batch(file_paths))
