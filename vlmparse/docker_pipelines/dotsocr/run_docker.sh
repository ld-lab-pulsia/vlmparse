#!/bin/bash

# Example script to manually run DotsOCR Docker container
# Note: The VLLMModelConfig handles deployment automatically when using get_converter()

docker run --gpus all \
    -p 8056:8000 \
    --rm \
    dotsocr:latest \
    /workspace/weights/DotsOCR \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --chat-template-content-format string \
    --served-model-name dotsocr-model \
    --trust-remote-code

