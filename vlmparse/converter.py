import asyncio
import time
from pathlib import Path
from typing import Literal

from loguru import logger
from PIL import Image
from pydantic import Field

from vlmparse.servers.docker_server import DEFAULT_MODEL_NAME

from .base_model import VLMParseBaseModel
from .build_doc import convert_specific_page_to_image, get_page_count, resize_image
from .constants import IMAGE_EXTENSIONS, PDF_EXTENSION
from .data_model.document import Document, Page, ProcessingError


class ConverterConfig(VLMParseBaseModel):
    model_name: str
    aliases: list[str] = Field(default_factory=list)
    dpi: int = Field(default=175, ge=30, le=600)
    max_image_size: int | None = Field(default=4000, ge=50)
    base_url: str | None = None
    default_model_name: str = DEFAULT_MODEL_NAME
    conversion_mode: Literal[
        "ocr",
        "ocr_layout",
        "table",
        "image_description",
        "formula",
        "chart",
    ] = "ocr"
    add_native_text: bool = False
    add_uri_to_items: bool = False
    """When True, overlapping native TextCells with hyperlink URIs are used to
    annotate matching words in each Item's text as markdown links.
    Automatically triggers native text-cell extraction (like add_native_text)."""

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
        self.num_concurrent_pages = (
            num_concurrent_pages  # triggers setter → creates page_semaphore
        )
        self.save_folder = save_folder
        self.save_mode = save_mode
        self.debug = debug
        self.return_documents_in_batch_mode = return_documents_in_batch_mode
        self.save_page_images = save_page_images

    @property
    def num_concurrent_pages(self) -> int:
        return self._num_concurrent_pages

    @num_concurrent_pages.setter
    def num_concurrent_pages(self, value: int) -> None:
        self._num_concurrent_pages = value
        self.page_semaphore = asyncio.Semaphore(value)

    async def async_call_inside_page(self, page: Page) -> Page:
        raise NotImplementedError

    async def async_call_inside_page_with_rendering(
        self, page: Page, file_path: str | Path, page_idx: int
    ) -> Page:
        if self.config.add_native_text or self.config.add_uri_to_items:
            from .data_model.box import BoundingBox
            from .docling_extractor import extract_page_text_cells

            page, (cells, pdf_w, pdf_h) = await asyncio.gather(
                asyncio.to_thread(self.add_page_image, page, file_path, page_idx),
                asyncio.to_thread(extract_page_text_cells, file_path, page_idx),
            )
            if cells is not None and pdf_w and pdf_h:
                img = page.image
                if img is not None:
                    scale_x = img.width / pdf_w
                    scale_y = img.height / pdf_h
                    for cell in cells:
                        b = cell.box
                        cell.box = BoundingBox(
                            l=b.l * scale_x,
                            t=b.t * scale_y,
                            r=b.r * scale_x,
                            b=b.b * scale_y,
                        )
                page.text_cells = cells
        return await self.async_call_inside_page(page)

    def add_page_image(self, page: Page, file_path, page_idx):
        if Path(file_path).suffix.lower() in IMAGE_EXTENSIONS:
            image = Image.open(file_path)
            if image.mode != "RGB":
                image = image.convert("L").convert("RGB")

        elif Path(file_path).suffix.lower() == PDF_EXTENSION:
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

            async def worker(page_idx: int, page: Page):
                async with self.page_semaphore:
                    try:
                        tic = time.perf_counter()
                        page = await self.async_call_inside_page_with_rendering(
                            page, file_path, page_idx
                        )
                        if self.config.add_uri_to_items:
                            from .uri_annotator import annotate_page_items_with_uris

                            annotate_page_items_with_uris(page)
                        toc = time.perf_counter()
                        page.latency = toc - tic
                        logger.debug(
                            "Page {page_idx} processed in {latency:.2f}s",
                            page_idx=page_idx,
                            latency=page.latency,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        if self.debug:
                            raise
                        else:
                            logger.opt(exception=True).error(
                                "Error processing page {page_idx} of {file_path}",
                                page_idx=page_idx,
                                file_path=str(file_path),
                            )
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
                logger.opt(exception=True).error(
                    "Error processing document {file_path}",
                    file_path=str(file_path),
                )
                document.error = ProcessingError.from_class(self)
                return document
        toc = time.perf_counter()
        document.latency = toc - tic
        logger.debug(
            "Document {file_path} processed in {latency:.2f}s",
            file_path=str(file_path),
            latency=document.latency,
        )
        if self.save_folder is not None:
            self._save_document(document)

        return document

    def _save_document(self, document: Document):
        """Save document according to save_mode."""
        if self.save_folder is not None:
            if document.is_error:
                save_folder = Path(self.save_folder) / "errors"

            else:
                save_folder = Path(self.save_folder) / "results"

        save_folder.mkdir(parents=True, exist_ok=True)
        doc_name = Path(document.file_path).stem

        if self.save_mode == "document":
            zip_path = save_folder / f"{doc_name}.zip"
            document.to_zip(zip_path)
            logger.debug(f"Saved document to {zip_path}")

        elif self.save_mode == "md":
            md_path = save_folder / f"{doc_name}.md"
            text_content = "\n\n".join([page.text or "" for page in document.pages])
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            logger.debug(f"Saved markdown to {md_path}")

        elif self.save_mode == "md_page":
            doc_folder = save_folder / doc_name
            doc_folder.mkdir(parents=True, exist_ok=True)
            for i, page in enumerate(document.pages, start=1):
                page_text = page.text if page.text else ""
                page_path = doc_folder / f"page_{i:04d}.md"
                with open(page_path, "w", encoding="utf-8") as f:
                    f.write(page_text)
            logger.debug(f"Saved {len(document.pages)} pages to {doc_folder}")

        else:
            logger.warning(f"Unknown save_mode: {self.save_mode}, skipping save")

    async def _async_call_with_cleanup(self, file_path: str | Path):
        """Call async_call and ensure cleanup."""
        try:
            return await self.async_call(file_path)
        finally:
            if hasattr(self, "aclose"):
                await self.aclose()

    def __call__(self, file_path: str | Path):
        return asyncio.run(self._async_call_with_cleanup(file_path))

    async def async_batch(self, file_paths: list[str | Path]) -> list[Document] | None:
        """Process multiple files concurrently with semaphore limit."""
        semaphore = asyncio.Semaphore(self.num_concurrent_files)

        async def worker(file_path: str | Path) -> Document | None:
            async with semaphore:
                if self.return_documents_in_batch_mode:
                    return await self.async_call(file_path)
                else:
                    await self.async_call(file_path)

        tasks = [asyncio.create_task(worker(file_path)) for file_path in file_paths]
        try:
            documents = await asyncio.gather(*tasks)
            if self.return_documents_in_batch_mode:
                documents = [doc for doc in documents if doc is not None]
                return documents
        finally:
            # Close async resources before the event loop ends
            if hasattr(self, "aclose"):
                await self.aclose()

    def batch(self, file_paths: list[str | Path]) -> list[Document] | None:
        """Synchronous wrapper for async_batch."""
        return asyncio.run(self.async_batch(file_paths))

    async def aclose(self):
        """Override if any async cleanup is needed."""
        pass
