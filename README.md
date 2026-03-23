# vlmparse

<div align="center">

[\[[📜 arXiv](https://arxiv.org/abs/2602.11960)\]] | [[Dataset (🤗Hugging Face)]](https://huggingface.co/datasets/pulsia/fr-bench-pdf2md) | [[pypi]](https://pypi.org/project/vlmparse/) | [[vlmparse]](https://github.com/ld-lab-pulsia/vlmparse) | [[Benchmark]](https://github.com/ld-lab-pulsia/benchpdf2md) | [[Leaderboard]](https://huggingface.co/spaces/pulsia/fr-bench-pdf2md)

</div>

A unified wrapper for Vision Language Models (VLM) and OCR solutions to parse PDF documents into Markdown.

Features:

- ⚡ Async/concurrent processing for high throughput
- 🐳 Automatic Docker server management for local models
- 🔄 Unified interface across all VLM/OCR providers
- 📄 Selective page processing (convert only the pages you need)
- 📊 Built-in result visualization with Streamlit

Supported Converters:

- **Open Source Small VLMs**: `lightonocr2`, `mineru2.5`, `hunyuanocr`, `paddleocrvl-1.5`, `granite-docling`, `olmocr-2-fp8`, `dotsocr`, `chandra`, `deepseekocr2`, `nanonets/Nanonets-OCR2-3B`
- **Open Source Generalist VLMs**: such as the Qwen family.
- **Pipelines**: `docling`
- **Proprietary LLMs**: `gemini`, `gpt`, `claude`

## Installation

Simplest solution with only the cli: 

```bash
uv tool install vlmparse
```

If you want to run the granite-docling model or use the streamlit viewing app:

```bash
uv tool install vlmparse[docling_core,st_app]
```

If you prefer cloning the repository and using the local version:
```bash
uv sync
```

With optional dependencies:

```bash
uv sync --all-extras
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

## CLI Usage

Note that you can bypass the previous installation step and just add uvx before each of the commands below.

### Convert PDFs

With a general VLM (requires setting your api key as an environment variable):

```bash
vlmparse convert "*.pdf" -o ./output --model gemini-2.5-flash-lite
```

Convert only specific pages (1-indexed, supports ranges: `"3"`, `"1,3,5"`, `"2-5"`, `"1,3-5,8"`):

```bash
vlmparse convert doc.pdf -o ./output --model gemini-2.5-flash-lite --pages "1,3-5,8"
```

Convert with auto deployment of a small vlm (or any huggingface VLM model, requires a gpu + docker installation):

```bash
vlmparse convert "*.pdf" -o ./output --model nanonets/Nanonets-OCR2-3B
```

### Deploy a local model server

Deployment (requires a gpu + docker installation):
- You need a gpu dedicated for this.
- Check that the port is not used by another service.

```bash
vlmparse serve lightonocr2 --port 8000 --gpu 1
```

then convert:

```bash
vlmparse convert "*.pdf" -o ./output --uri http://localhost:8000/v1
```

You can also list all running servers:

```bash
vlmparse list
```

Show logs of a server (if only one server is running, the container name is not needed):
```bash
vlmparse log <container_name>
```

Stop a server (if only one server is running, the container name is not needed):
```bash
vlmparse stop <container_name>
```

### View conversion results with Streamlit

```bash
vlmparse view ./output
```

## Configuration

Set API keys as environment variables:

```bash
export GOOGLE_API_KEY="your-key"       # Gemini models
export OPENAI_API_KEY="your-key"       # GPT models
export ANTHROPIC_API_KEY="your-key"    # Claude models
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

Process only specific pages (0-indexed list):

```python
client.pages = [0, 2, 4]  # pages 1, 3 and 5 of the PDF
document = client("path/to/document.pdf")
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


Converter with automatic server management:

```python
from vlmparse.converter_with_server import ConverterWithServer

with ConverterWithServer(model="mineru25") as converter_with_server:
    documents = converter_with_server.parse(inputs=["file1.pdf", "file2.pdf"], out_folder="./output")
```

Note that if you pass an uri of a vllm server to `ConverterWithServer`, the model name is inferred automatically and no server is started.

## Credits

This work was realised by members of [Probayes](https://www.probayes.com/) and [OpenValue](https://openvalue.co/), two subsidiaries of La Poste.
