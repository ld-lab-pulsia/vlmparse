FROM vllm/vllm-openai:nightly

RUN apt-get update -qq && apt-get install -y -qq git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers.git
