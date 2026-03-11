"""LitServe layout detection server using PP-DocLayoutV3 (transformers).

Exposes a single POST /predict endpoint:
    Request:  {"image": "<base64-encoded JPEG/PNG>"}
    Response: {"regions": [{"label": str, "score": float, "bbox": [x1, y1, x2, y2]}, ...]}

Environment variables:
    LAYOUT_MODEL_DIR    HuggingFace model id or local path
                        (default: PaddlePaddle/PP-DocLayoutV3_safetensors)
    LAYOUT_THRESHOLD    Confidence threshold (default: 0.3)
    LAYOUT_BATCH_SIZE   Max images per model forward pass (default: 8)
    LAYOUT_PORT         Server port (default: 8090)
    LAYOUT_COMPILE      Use torch.compile(dynamic=True) to eliminate
                        per-batch-size CUDA recompilation (default: 1)
"""

import base64
import logging
import os
import time

import cv2
import litserve as ls
import numpy as np
import torch
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessorFast,
)

ls.configure_logging(use_rich=True)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LAYOUT_MODEL_DIR = os.environ.get(
    "LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
)
THRESHOLD = float(os.environ.get("LAYOUT_THRESHOLD", "0.3"))
LAYOUT_COMPILE = os.environ.get("LAYOUT_COMPILE", "0") not in ("0", "false", "False")
MEASURE_TIMES = False  # Set to False to disable detailed timing logs


class LayoutDetectorAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(
            LAYOUT_MODEL_DIR
        )
        self.model = PPDocLayoutV3ForObjectDetection.from_pretrained(LAYOUT_MODEL_DIR)
        self.model.eval()
        self.model = self.model.to(device)
        self.precision = torch.float16
        if device != "cpu":
            self.model = self.model.to(self.precision)
        self.id2label = self.model.config.id2label
        if LAYOUT_COMPILE:
            # dynamic=True: one compiled kernel handles all batch sizes,
            # eliminating per-shape CUDA JIT recompilation (~1100–1400 ms spikes).
            # Use cudagraphs if g++ is unavailable (no C++ compiler needed);
            # install g++ in the Docker image to use the faster inductor backend.
            import shutil

            backend = "inductor" if shutil.which("g++") else "cudagraphs"
            self.model = torch.compile(self.model, dynamic=True, backend=backend)
            logger.info("torch.compile(dynamic=True, backend=%s) enabled.", backend)

        # Warmup: trigger kernel compilation (inductor: ~2 min first time) during
        # setup so it never blocks a live request. Subsequent startups use the cache.
        logger.info("Running warmup pass (may take a few minutes on first run)...")
        # CHW uint8 dummy tensor – mirrors what decode_request now produces.
        dummy_tensor = torch.zeros((3, 800, 800), dtype=torch.uint8)
        for batch_size in range(1, 9):
            all_tensors = [
                dummy_tensor.to(device=device, non_blocking=True)
            ] * batch_size
            dummy_inputs = self.processor(images=all_tensors, return_tensors="pt")
            dummy_inputs = {
                k: v.to(device=self.device, dtype=self.precision, non_blocking=True)
                for k, v in dummy_inputs.items()
            }
            with torch.inference_mode():
                self.model(**dummy_inputs)

        logger.info("Warmup complete.")

    def decode_request(self, request):
        """Decode a single request: base64 image → raw CHW uint8 tensor (CPU).

        Preprocessing (resize + normalize) is deferred to predict() so it runs
        on the GPU for the whole batch at once.
        """
        if MEASURE_TIMES:
            t0 = time.perf_counter()
        image_data = base64.b64decode(request["image"])
        arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        # HWC numpy uint8 → CHW torch uint8; preprocessing deferred to predict()
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        if MEASURE_TIMES:
            logger.debug("decode_request: %.1f ms", (time.perf_counter() - t0) * 1000)
        return {"image": image_tensor, "size": original_size}

    def predict(self, batch):
        """Run preprocessor on GPU, then model forward pass and postprocess."""
        if not isinstance(batch, list):
            batch = [batch]
        batch_size = len(batch)
        if MEASURE_TIMES:
            t0 = time.perf_counter()
        # Move raw CHW uint8 images to GPU and run the preprocessor there so
        # resize + normalize execute on CUDA rather than on CPU.
        images_on_device = [
            item["image"].to(device=self.device, non_blocking=True) for item in batch
        ]
        inputs = self.processor(images=images_on_device, return_tensors="pt")
        stacked = {
            k: v.to(device=self.device, dtype=self.precision, non_blocking=True)
            for k, v in inputs.items()
        }
        target_sizes = torch.tensor(
            [[img_h, img_w] for img_w, img_h in [item["size"] for item in batch]]
        ).to(self.device, non_blocking=True)
        if MEASURE_TIMES:
            t1 = time.perf_counter()
            logger.debug(
                "stack+to_device (batch=%d): %.1f ms", batch_size, (t1 - t0) * 1000
            )
        for k, v in stacked.items():
            logger.debug("  input %s: %s", k, list(v.shape))

        with torch.inference_mode():
            outputs = self.model(**stacked)
        if MEASURE_TIMES:
            if self.device != "cpu":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            logger.debug("forward (batch=%d): %.1f ms", batch_size, (t2 - t1) * 1000)

        raw_results = self.processor.post_process_object_detection(
            outputs, threshold=THRESHOLD, target_sizes=target_sizes
        )
        if self.device != "cpu":
            torch.cuda.synchronize()
        if MEASURE_TIMES:
            t3 = time.perf_counter()
            logger.debug(
                "postprocess (batch=%d): %.1f ms", batch_size, (t3 - t2) * 1000
            )
            logger.info(
                "predict total without decode and encode (batch=%d): %.1f ms  [stack=%.1f  fwd=%.1f  post=%.1f]",
                batch_size,
                (t3 - t0) * 1000,
                (t1 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t2) * 1000,
            )

        # Return list of (per-image result, original image size) so encode_response
        # receives one entry per input image.
        return list(zip(raw_results, [item["size"] for item in batch], strict=False))

    def encode_response(self, output):
        """Serialize a single image's detection result to JSON."""
        if MEASURE_TIMES:
            t0 = time.perf_counter()
        result, (_img_w, _img_h) = output
        scores = result["scores"].cpu().tolist()
        labels = result["labels"].cpu().tolist()
        boxes = result["boxes"].cpu().tolist()
        regions = []
        for score, label_id, box in zip(scores, labels, boxes, strict=False):
            x1, y1, x2, y2 = box
            regions.append(
                {
                    "label": self.id2label[int(label_id)],
                    "score": score,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        if MEASURE_TIMES:
            logger.debug("encode_response: %.1f ms", (time.perf_counter() - t0) * 1000)
        return {"regions": regions}


if __name__ == "__main__":
    api = LayoutDetectorAPI()
    server = ls.LitServer(
        api,
        api_path="/predict",
        max_batch_size=int(os.environ.get("LAYOUT_BATCH_SIZE", "4")),
        batch_timeout=0.01,
        workers_per_device=2,
    )
    server.run(
        port=int(os.environ.get("LAYOUT_PORT", "8090")),
        num_api_servers=1,
    )
