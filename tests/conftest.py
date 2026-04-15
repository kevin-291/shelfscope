from __future__ import annotations
from src import create_app

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeDetectorService:
    def __init__(self, detections: list[dict] | None = None) -> None:
        self.calls = 0
        self.device = "test-device"
        self._detections = detections or [
            {
                "detection_id": "det_0001",
                "bbox": [10, 10, 60, 90],
                "score": 0.95,
                "class_name": "product",
            },
            {
                "detection_id": "det_0002",
                "bbox": [70, 15, 125, 95],
                "score": 0.91,
                "class_name": "product",
            },
        ]

    def detect(self, image: Image.Image):
        self.calls += 1
        return type(
            "DetectorResult",
            (),
            {
                "detections": [dict(item) for item in self._detections],
                "image_size": image.size,
            },
        )()


class FakeGroupingService:
    def __init__(self) -> None:
        self.calls = 0
        self.device = "test-device"

    def assign_groups(self, image: Image.Image, detections: list[dict]):
        self.calls += 1
        colors = ["#FF6B6B", "#4ECDC4"]
        for index, detection in enumerate(detections):
            detection["group_id"] = f"group_{index + 1:03d}"
            detection["group_color"] = colors[index % len(colors)]
        return detections


class FakeVisualizationService:
    def __init__(self) -> None:
        self.calls = 0

    def render(
        self, image: Image.Image, detections: list[dict], output_path: Path
    ) -> None:
        self.calls += 1
        canvas = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        cv2.imwrite(str(output_path), canvas)


@pytest.fixture
def fake_services():
    return {
        "detector": FakeDetectorService(),
        "grouping": FakeGroupingService(),
        "visualization": FakeVisualizationService(),
    }


@pytest.fixture
def app_fixture(tmp_path: Path, fake_services):
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    app = create_app(
        detector_service=fake_services["detector"],
        grouping_service=fake_services["grouping"],
        visualization_service=fake_services["visualization"],
        upload_dir=upload_dir,
        output_dir=output_dir,
    )
    app.state.testing = True
    return app


@pytest.fixture
def client(app_fixture):
    return TestClient(app_fixture)


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (160, 120), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def uploaded_image(sample_image_bytes: bytes):
    return ("sample.png", io.BytesIO(sample_image_bytes), "image/png")


@pytest.fixture
def sample_images_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "sample_images (1) (1) (1) (1)"
        / "sample_images"
    )


@pytest.fixture
def real_sample_image_path(sample_images_dir: Path) -> Path:
    matches = sorted(sample_images_dir.glob("*.jpg"))
    if not matches:
        pytest.skip("No sample images found for integration testing.")
    return matches[0]
