FROM python:3.11-slim

RUN apt-get update -qq && \
    apt-get install -y -qq libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir litserve torch torchvision \
        transformers pillow opencv-python-headless

WORKDIR /app
