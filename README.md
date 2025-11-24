# vlmparse

A unified wrapper for Vision Language Models (VLM) and OCR solutions to parse PDF documents into Markdown.

Features:

- ⚡ Async/concurrent processing for high throughput
- 🐳 Automatic Docker server management for local models
- 🔄 Unified interface across all VLM/OCR providers
- 📊 Built-in result visualization with Streamlit

Supported Converters:

- **Open Source Small VLMs**: `lightonocr`, `dotsocr`, `nanonets/Nanonets-OCR2-3B`
- **Pipelines**: `docling`
- **Proprietary LLMs**: `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gpt-5.1`

## Installation

```bash
uv sync
```

With optional dependencies:

```bash
uv sync --all-extras
```

## CLI Usage

### Convert PDFs

With a general VLM (requires setting your api key as an environment variable):

```bash
vlmparse convert --input "*.pdf" --out_folder ./output --model gemini-2.5-flash-lite
```

Convert with auto deployment of a small vlm (or any huggingface VLM model, requires a gpu + docker installation):

```bash
vlmparse convert --input "*.pdf" --out_folder ./output --model nanonets/Nanonets-OCR2-3B
```

### Deploy a local model server

Deployment (requires a gpu + docker installation):

```bash
vlmparse serve --model lightonocr --port 8000
```

then convert:

```bash
vlmparse convert --input "*.pdf" --out_folder ./output --model lightonocr --uri http://localhost:8000/v1
```

You can also list all running servers:

```bash
vlmparse list
```

### View results with Streamlit

```bash
vlmparse view ./output
```

## Configuration

Set API keys as environment variables:

```bash
export GOOGLE_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## Python API

Client interface:

```python
from vlmparse.registries import converter_config_registry

# Get a converter configuration
config = converter_config_registry.get("gemini-2.5-flash-lite")
client = config.get_client()

# Convert a single PDF
document = client("path/to/document.pdf")
print(document.to_markdown())

# Batch convert multiple PDFs
documents = client.batch(["file1.pdf", "file2.pdf"])
```

Docker server interface:

```python
from vlmparse.registries import docker_config_registry

config = docker_config_registry.get("lightonocr")
server = config.get_server()
server.start()

# Client calls...

server.stop()
```
