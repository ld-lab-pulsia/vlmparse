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
"""

import base64
import os
from io import BytesIO

import litserve as ls
import torch
from PIL import Image
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessorFast,
)

ls.configure_logging(use_rich=True)

LAYOUT_MODEL_DIR = os.environ.get(
    "LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
)
THRESHOLD = float(os.environ.get("LAYOUT_THRESHOLD", "0.3"))


class LayoutDetectorAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(
            LAYOUT_MODEL_DIR
        )
        self.model = PPDocLayoutV3ForObjectDetection.from_pretrained(LAYOUT_MODEL_DIR)
        self.model.eval()
        self.model = self.model.to(device)
        self.id2label = self.model.config.id2label

    def decode_request(self, request):
        """Decode a single request: base64 image → PIL Image."""
        image_data = base64.b64decode(request["image"])
        return Image.open(BytesIO(image_data)).convert("RGB")

    def predict(self, images):
        """Run detection on a batch (or single image) and return per-image results."""
        if not isinstance(images, list):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        target_sizes = torch.tensor(
            [[img.height, img.width] for img in images], device=self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        raw_results = self.processor.post_process_object_detection(
            outputs, threshold=THRESHOLD, target_sizes=target_sizes
        )

        # Return list of (per-image result, original image size) so encode_response
        # receives one entry per input image.
        return list(zip(raw_results, [img.size for img in images], strict=False))

    def encode_response(self, output):
        """Serialize a single image's detection result to JSON."""
        result, (_img_w, _img_h) = output
        regions = []
        for score, label_id, box in zip(
            result["scores"].cpu().tolist(),
            result["labels"].cpu().tolist(),
            result["boxes"].cpu().tolist(),
            strict=False,
        ):
            x1, y1, x2, y2 = box
            regions.append(
                {
                    "label": self.id2label[int(label_id)],
                    "score": score,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return {"regions": regions}


if __name__ == "__main__":
    api = LayoutDetectorAPI()
    server = ls.LitServer(
        api,
        api_path="/predict",
        max_batch_size=int(os.environ.get("LAYOUT_BATCH_SIZE", "8")),
        batch_timeout=0.05,
    )
    server.run(
        port=int(os.environ.get("LAYOUT_PORT", "8090")),
    )
