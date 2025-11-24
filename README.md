# vlmparse

A unified wrapper for Vision Language Models (VLM) and OCR solutions to parse PDF documents into Markdown.

Features:

- ⚡ Async/concurrent processing for high throughput
- 🐳 Automatic Docker server management for local models
- 🔄 Unified interface across all VLM/OCR providers
- 📊 Built-in result visualization with Streamlit

## Installation

```bash
uv sync
```

With optional dependencies:

```bash
uv sync --extra test --extra st_app
```

## Quick Start

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

## Supported Converters

- **Open Source OCR**: `lightonocr`, `dotsocr`, `nanonets/Nanonets-OCR2-3B`
- **Google Gemini**: `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`

## CLI Usage

### Convert PDFs

With a general VLM (requires setting your api key as an environment variable):

```bash
vlmparse convert --folders "*.pdf" --out_folder ./output --model gemini-2.5-flash-lite
```

Convert with auto deployment of a small vlm (or any huggingface VLM model, requires a gpu + docker installation):

```bash
vlmparse convert --folders "*.pdf" --out_folder ./output --model nanonets/Nanonets-OCR2-3B
```

### Deploy a local model server

Deployment (requires a gpu + docker installation):

```bash
vlmparse serve --model lightonocr --port 8000
```

then convert:

```bash
vlmparse convert --folders "*.pdf" --out_folder ./output --model lightonocr --uri http://localhost:8000/v1
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
