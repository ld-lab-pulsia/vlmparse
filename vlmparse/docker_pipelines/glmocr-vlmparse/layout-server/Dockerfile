FROM python:3.11-slim

RUN apt-get update -qq && \
    apt-get install -y -qq libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir litserve \
        transformers pillow opencv-python-headless

WORKDIR /app
COPY layout_server.py /app/layout_server.py
