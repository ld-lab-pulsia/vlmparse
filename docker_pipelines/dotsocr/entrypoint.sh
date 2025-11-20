#!/bin/bash
set -ex

echo "--- Starting setup and server ---"
echo "Modifying vllm entrypoint..."

# Patch the vllm entrypoint to import custom modeling code
sed -i "/^from vllm\.entrypoints\.cli\.main import main/a from DotsOCR import modeling_dots_ocr_vllm" $(which vllm)

echo "vllm script after patch:"
grep -A 1 "from vllm.entrypoints.cli.main import main" $(which vllm)

echo "Starting server..."
exec vllm serve "$@"

