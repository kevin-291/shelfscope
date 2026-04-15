from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

from src.config import (
    CONFIDENCE_THRESHOLD,
    HF_CACHE_DIR,
    MAX_DETECTIONS,
    MODEL_ID,
)
from src.utils import ensure_dir, get_torch_device


@dataclass
class DetectorResult:
    detections: list[dict[str, Any]]
    image_size: tuple[int, int]


class DetectorService:
    def __init__(self) -> None:
        self.device = get_torch_device()
        ensure_dir(HF_CACHE_DIR)
        model_source = self._resolve_model_source()
        self.processor = DetrImageProcessor.from_pretrained(
            model_source, cache_dir=str(HF_CACHE_DIR)
        )
        config = self._load_config(model_source)
        self.model = DetrForObjectDetection.from_pretrained(
            model_source, config=config, cache_dir=str(HF_CACHE_DIR)
        ).to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.half()

    def _resolve_model_source(self) -> str:
        repo_cache_dir = HF_CACHE_DIR / "models--is36e--detr-resnet-50-sku110k"
        ref_file = repo_cache_dir / "refs" / "main"
        if ref_file.exists():
            snapshot = ref_file.read_text(encoding="utf-8").strip()
            snapshot_dir = repo_cache_dir / "snapshots" / snapshot
            if snapshot_dir.exists():
                return str(snapshot_dir)
        return MODEL_ID

    def _load_config(self, model_source: str) -> DetrConfig:
        config_path = Path(model_source) / "config.json"
        if config_path.exists():
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            raw_config = DetrConfig.get_config_dict(
                model_source, cache_dir=str(HF_CACHE_DIR)
            )[0]

        if raw_config.get("backbone_kwargs") is None:
            raw_config["backbone_kwargs"] = {}
        return DetrConfig.from_dict(raw_config)

    @torch.inference_mode()
    def detect(
        self, image: Image.Image, threshold: float = CONFIDENCE_THRESHOLD
    ) -> DetectorResult:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[height, width]], device=self.device)
        processed = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        detections: list[dict[str, Any]] = []
        boxes = processed["boxes"].detach().cpu().tolist()
        scores = processed["scores"].detach().cpu().tolist()

        for index, (box, score) in enumerate(zip(boxes, scores), start=1):
            x_min, y_min, x_max, y_max = [int(round(value)) for value in box]
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(x_min + 1, min(x_max, width))
            y_max = max(y_min + 1, min(y_max, height))

            detections.append(
                {
                    "detection_id": f"det_{index:04d}",
                    "bbox": [x_min, y_min, x_max, y_max],
                    "score": round(float(score), 4),
                    "class_name": "product",
                }
            )

            if len(detections) >= MAX_DETECTIONS:
                break

        return DetectorResult(detections=detections, image_size=(width, height))
