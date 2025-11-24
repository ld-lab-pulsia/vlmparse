from pydantic import Field
from typing import Literal
import httpx

from vlmparse.servers.docker_server import DockerServerConfig
from vlmparse.converter import BaseConverter, ConverterConfig
from vlmparse.data_model.document import Page
from vlmparse.clients.pipe_utils.html_to_md_conversion import html_to_md_keep_tables
from vlmparse.clients.pipe_utils.utils import clean_response
from vlmparse.utils import to_base64
from loguru import logger


class DoclingDockerServerConfig(DockerServerConfig):
    """Configuration for Docling Serve using official image."""
    
    model_name: str = "docling"
    docker_image: str = "quay.io/docling-project/docling-serve:latest"
    command_args: list[str] = Field(default_factory=list)
    server_ready_indicators: list[str] = Field(
        default_factory=lambda: ["Application startup complete", "Uvicorn running"]
    )
    docker_port: int = 5001
    container_port: int = 5001
    environment: dict[str, str] = Field(
        default_factory=lambda: {
            "DOCLING_SERVE_HOST": "0.0.0.0",
            "DOCLING_SERVE_PORT": "5001"
        }
    )

    @property
    def client_config(self):
        return DoclingConverterConfig(base_url=f"http://localhost:{self.docker_port}")


class DoclingConverterConfig(ConverterConfig):
    """Configuration for Docling converter client."""
    base_url: str = "http://localhost:5001"
    timeout: int = 300
    output_format: Literal["markdown", "json", "text"] = "markdown"

    def get_client(self, **kwargs) -> 'DoclingConverter':
        return DoclingConverter(config=self, **kwargs)


class DoclingConverter(BaseConverter):
    """Client for Docling Serve API using httpx."""
    
    def __init__(
        self,
        config: DoclingConverterConfig,
        num_concurrent_files: int = 10,
        num_concurrent_pages: int = 10,
        save_folder: str | None = None,
        save_mode: Literal["document", "md", "md_page"] = "document",
        debug: bool = False
    ):
        super().__init__(
            config=config,
            num_concurrent_files=num_concurrent_files,
            num_concurrent_pages=num_concurrent_pages,
            save_folder=save_folder,
            save_mode=save_mode,
            debug=debug
        )
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
    
    async def async_call_inside_page(self, page: Page) -> Page:
        """Process a single page using Docling Serve API."""
        image = page.image
        
        # Convert image to base64
        image_base64 = to_base64(image)
        
        # Prepare the request according to Docling Serve API
        # Using the data URL format for image input
        request_data = {
            "sources": [
                {
                    "kind": "data_url",
                    "data_url": f"data:image/png;base64,{image_base64}"
                }
            ],
            "options": {
                "output_format": self.config.output_format
            }
        }
        
        # Make the API call
        url = f"{self.config.base_url}/v1/convert/source"
        
        try:
            response = await self.client.post(
                url,
                json=request_data,
                headers={"Content-Type": "application/json", "Accept": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Docling API response status: {response.status_code}")
            
            # Extract text from the response
            # The response structure depends on the output format
            if self.config.output_format == "markdown":
                # Extract markdown content from the response
                if "results" in result and len(result["results"]) > 0:
                    text = result["results"][0].get("markdown", "")
                elif "markdown" in result:
                    text = result["markdown"]
                else:
                    text = str(result)
            elif self.config.output_format == "text":
                if "results" in result and len(result["results"]) > 0:
                    text = result["results"][0].get("text", "")
                elif "text" in result:
                    text = result["text"]
                else:
                    text = str(result)
            else:  # json or other formats
                text = str(result)
            
            logger.info(f"Extracted text length: {len(text)}")
            
            # Clean and convert the response
            text = clean_response(text)
            text = html_to_md_keep_tables(text)
            page.text = text
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            page.text = f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error processing page with Docling: {e}")
            page.text = f"Error: {str(e)}"
        
        return page
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup when the converter is destroyed."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.client.aclose())
            else:
                asyncio.run(self.client.aclose())
        except Exception:
            pass

