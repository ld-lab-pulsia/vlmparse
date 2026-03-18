# Dockerfile for GLM-OCR API service
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Clone GLM-OCR repository and install in editable mode
# Note: The pyproject.toml has packages=["glmocr"] which excludes submodules,
# so we need to install in editable mode to preserve the directory structure
RUN git clone https://github.com/zai-org/GLM-OCR.git /app/GLM-OCR && \
    cd /app/GLM-OCR && \
    pip install --no-cache-dir -e .[layout,server] && \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers.git

# Copy configuration
COPY config.yaml /app/GLM-OCR/glmocr/config.yaml

# Set working directory to the cloned repo
WORKDIR /app/GLM-OCR

# Expose the API port
EXPOSE 5002

# Start the Flask service
CMD ["sh", "-c", "python -m glmocr.server --log-level ${LOG_LEVEL:-INFO}"]
