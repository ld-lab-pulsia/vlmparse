import asyncio
import time
import traceback
from pathlib import Path
from typing import Literal

import nest_asyncio
from loguru import logger

from .base_model import VLMParseBaseModel
from .build_doc import convert_pdfium_to_images
from .data_model.document import Document, Page, ProcessingError

nest_asyncio.apply()


class ConverterConfig(VLMParseBaseModel):
    dpi: int = 175
    max_image_size: int | None = 4000

    def get_client(self, **kwargs) -> "BaseConverter":
        return BaseConverter(config=self, **kwargs)


class BaseConverter:
    def __init__(
        self,
        config: ConverterConfig,
        num_concurrent_files: int = 10,
        num_concurrent_pages: int = 10,
        save_folder: str | None = None,
        save_mode: Literal["document", "md", "md_page"] = "document",
        debug: bool = False,
        return_documents_in_batch_mode: bool = False,
    ):
        self.config = config
        self.num_concurrent_files = num_concurrent_files
        self.num_concurrent_pages = num_concurrent_pages
        self.save_folder = save_folder
        self.save_mode = save_mode
        self.debug = debug
        self.return_documents_in_batch_mode = return_documents_in_batch_mode

    async def async_call_inside_page(self, page: Page) -> Page:
        raise NotImplementedError

    async def async_call(self, file_path: str | Path) -> Document:
        tic = time.perf_counter()
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
                    logger.debug("Image size: " + str(page.image.size))
                    try:
                        tic = time.perf_counter()
                        await self.async_call_inside_page(page)
                        toc = time.perf_counter()
                        page.latency = toc - tic
                        logger.debug(f"Time taken: {page.latency} seconds")
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        if self.debug:
                            raise
                        else:
                            logger.exception(traceback.format_exc())
                            page.error = ProcessingError.from_class(self)

            tasks = [asyncio.create_task(worker(page)) for page in document.pages]
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            raise
        except Exception:
            if self.debug:
                raise
            else:
                logger.exception(traceback.format_exc())
                document.error = ProcessingError.from_class(self)
                return document
        toc = time.perf_counter()
        document.latency = toc - tic
        logger.debug(f"Time taken to process the document: {document.latency} seconds")
        if self.save_folder is not None:
            self._save_document(document)

        return document

    def _save_document(self, document: Document):
        """Save document according to save_mode."""
        if document.is_error:
            save_folder = Path(self.save_folder) / "errors"

        else:
            save_folder = Path(self.save_folder) / "results"

        save_folder.mkdir(parents=True, exist_ok=True)
        doc_name = Path(document.file_path).stem

        if self.save_mode == "document":
            zip_path = save_folder / f"{doc_name}.zip"
            document.to_zip(zip_path)
            logger.info(f"Saved document to {zip_path}")

        elif self.save_mode == "md":
            md_path = save_folder / f"{doc_name}.md"
            text_content = "\n\n".join([page.text or "" for page in document.pages])
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            logger.info(f"Saved markdown to {md_path}")

        elif self.save_mode == "md_page":
            doc_folder = save_folder / doc_name
            doc_folder.mkdir(parents=True, exist_ok=True)
            for i, page in enumerate(document.pages, start=1):
                page_text = page.text if page.text else ""
                page_path = doc_folder / f"page_{i:04d}.md"
                with open(page_path, "w", encoding="utf-8") as f:
                    f.write(page_text)
            logger.info(f"Saved {len(document.pages)} pages to {doc_folder}")

        else:
            logger.warning(f"Unknown save_mode: {self.save_mode}, skipping save")

    def __call__(self, file_path: str | Path):
        return asyncio.run(self.async_call(file_path))

    async def async_batch(self, file_paths: list[str | Path]) -> list[Document | None]:
        """Process multiple files concurrently with semaphore limit."""
        semaphore = asyncio.Semaphore(self.num_concurrent_files)

        async def worker(file_path: str | Path) -> Document:
            async with semaphore:
                if self.return_documents_in_batch_mode:
                    return await self.async_call(file_path)
                else:
                    await self.async_call(file_path)

        tasks = [asyncio.create_task(worker(file_path)) for file_path in file_paths]
        documents = await asyncio.gather(*tasks)
        if self.return_documents_in_batch_mode:
            return documents

    def batch(self, file_paths: list[str | Path]) -> list[Document | None]:
        """Synchronous wrapper for async_batch."""
        return asyncio.run(self.async_batch(file_paths))
