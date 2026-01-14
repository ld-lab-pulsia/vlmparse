import asyncio
import threading
import time
import traceback
from pathlib import Path
from typing import Literal

from loguru import logger
from PIL import Image
from pydantic import Field

from .base_model import VLMParseBaseModel
from .build_doc import convert_specific_page_to_image, get_page_count, resize_image
from .constants import IMAGE_EXTENSIONS, PDF_EXTENSION
from .data_model.document import Document, Page, ProcessingError

# Add a lock to ensure PDFium is accessed by only one thread/task at a time
PDFIUM_LOCK = threading.Lock()


class ConverterConfig(VLMParseBaseModel):
    aliases: list[str] = Field(default_factory=list)
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
        save_page_images: bool = False,
    ):
        self.config = config
        self.num_concurrent_files = num_concurrent_files
        self.num_concurrent_pages = num_concurrent_pages
        self.save_folder = save_folder
        self.save_mode = save_mode
        self.debug = debug
        self.return_documents_in_batch_mode = return_documents_in_batch_mode
        self.save_page_images = save_page_images

    async def async_call_inside_page(self, page: Page) -> Page:
        raise NotImplementedError

    def add_page_image(self, page: Page, file_path, page_idx):
        if Path(file_path).suffix.lower() in IMAGE_EXTENSIONS:
            image = Image.open(file_path)
            if image.mode != "RGB":
                image = image.convert("L").convert("RGB")

        elif Path(file_path).suffix.lower() == PDF_EXTENSION:
            with PDFIUM_LOCK:
                image = convert_specific_page_to_image(
                    file_path,
                    page_idx,
                    dpi=self.config.dpi,
                )

        else:
            raise ValueError(
                f"Unsupported file extension: {Path(file_path).suffix.lower()}"
            )

        image = resize_image(image, self.config.max_image_size)
        page.buffer_image = image
        return page

    async def async_call(self, file_path: str | Path) -> Document:
        tic = time.perf_counter()
        document = Document(file_path=str(file_path))
        try:
            num_pages = get_page_count(file_path)
            document.pages = [Page() for _ in range(num_pages)]

            semaphore = asyncio.Semaphore(self.num_concurrent_pages)

            async def worker(page_idx: int, page: Page):
                async with semaphore:
                    try:
                        page = await asyncio.to_thread(
                            self.add_page_image, page, file_path, page_idx
                        )

                        tic = time.perf_counter()
                        page = await self.async_call_inside_page(page)
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
                    if not self.save_page_images:
                        page.buffer_image = dict(
                            file_path=str(file_path),
                            page_idx=page_idx,
                            dpi=self.config.dpi,
                            max_image_size=self.config.max_image_size,
                        )

            tasks = [
                asyncio.create_task(worker(i, page))
                for i, page in enumerate(document.pages)
            ]
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

    async def async_batch(self, file_paths: list[str | Path]) -> list[Document] | None:
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

    def batch(self, file_paths: list[str | Path]) -> list[Document] | None:
        """Synchronous wrapper for async_batch."""
        return asyncio.run(self.async_batch(file_paths))
