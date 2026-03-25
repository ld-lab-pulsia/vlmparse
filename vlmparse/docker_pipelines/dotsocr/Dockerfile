FROM vllm/vllm-openai:v0.9.1

# Install dependencies
RUN pip3 install --no-cache-dir flash_attn==2.8.0.post2
RUN pip3 install --no-cache-dir transformers==4.51.3
RUN pip3 install --no-cache-dir huggingface_hub

# Set up working directory
WORKDIR /workspace

# Download model from Hugging Face
RUN mkdir -p /workspace/weights && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='rednote-hilab/dots.ocr', local_dir='/workspace/weights/DotsOCR', local_dir_use_symlinks=False)"

# Set environment variables
ENV PYTHONPATH=/workspace/weights:$PYTHONPATH

# Expose port
EXPOSE 8000

# Copy and set entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

