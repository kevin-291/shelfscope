from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils import hex_to_bgr


class VisualizationService:
    def render(
        self, image: Image.Image, detections: list[dict], output_path: Path
    ) -> None:
        canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        for detection in detections:
            x_min, y_min, x_max, y_max = detection["bbox"]
            color = hex_to_bgr(detection["group_color"])
            label = f"{detection['group_id']} {detection['score']:.2f}"

            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_top = max(0, y_min - text_h - 10)
            cv2.rectangle(
                canvas,
                (x_min, label_top),
                (x_min + text_w + 10, label_top + text_h + 8),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (x_min + 5, label_top + text_h + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(output_path), canvas)
