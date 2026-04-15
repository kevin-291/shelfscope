from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api import get_detector_service
from src import create_app


def test_index_route_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Retail Shelf" in response.text


def test_infer_rejects_missing_image(client):
    response = client.post("/infer", files={})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "missing"


def test_infer_rejects_unsupported_file_type(client):
    response = client.post(
        "/infer",
        files={"image": ("bad.txt", io.BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_infer_rejects_invalid_image_bytes(client):
    response = client.post(
        "/infer",
        files={"image": ("broken.png", io.BytesIO(b"not-an-image"), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."


def test_infer_success_saves_outputs_and_returns_contract(
    client, app_fixture, uploaded_image, fake_services
):
    response = client.post("/infer", files={"image": uploaded_image})

    assert response.status_code == 200
    payload = response.json()
    request_id = payload["request_id"]
    output_dir = app_fixture.state.output_dir / request_id
    upload_dir = app_fixture.state.upload_dir / request_id

    assert payload["image"]["filename"] == "sample.png"
    assert len(payload["detections"]) == 2
    assert payload["detections"][0]["group_id"] == "group_001"
    assert payload["visualization_url"] == f"/outputs/{request_id}/annotated.png"
    assert payload["json_url"] == f"/outputs/{request_id}/result.json"

    assert (upload_dir / "input.png").exists()
    assert (output_dir / "annotated.png").exists()
    assert (output_dir / "result.json").exists()

    assert fake_services["detector"].calls == 1
    assert fake_services["grouping"].calls == 1
    assert fake_services["visualization"].calls == 1


def test_saved_json_matches_api_response(client, app_fixture, uploaded_image):
    response = client.post("/infer", files={"image": uploaded_image})
    payload = response.json()
    request_id = payload["request_id"]
    saved_json_path = app_fixture.state.output_dir / request_id / "result.json"

    saved_payload = json.loads(saved_json_path.read_text(encoding="utf-8"))

    assert saved_payload == payload


def test_outputs_route_serves_saved_artifacts(client, uploaded_image):
    response = client.post("/infer", files={"image": uploaded_image})
    payload = response.json()

    image_response = client.get(payload["visualization_url"])
    json_response = client.get(payload["json_url"])

    assert image_response.status_code == 200
    assert json_response.status_code == 200
    assert json_response.json()["request_id"] == payload["request_id"]


def test_pipeline_failure_returns_500(tmp_path: Path, sample_image_bytes: bytes):
    class FailingGroupingService:
        def assign_groups(self, image, detections):
            raise RuntimeError("grouping failed")

    detector_stub = type(
        "DetectorStub",
        (),
        {
            "detect": lambda self, image: type(
                "DetectorResult",
                (),
                {"detections": [], "image_size": image.size},
            )()
        },
    )()

    app = create_app(
        detector_service=detector_stub,
        grouping_service=FailingGroupingService(),
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
    )
    client = TestClient(app)

    response = client.post(
        "/infer",
        files={"image": ("sample.png", io.BytesIO(sample_image_bytes), "image/png")},
    )

    assert response.status_code == 500
    assert "Pipeline failed" in response.json()["detail"]


def test_detector_service_is_cached_per_app(tmp_path: Path):
    app = create_app(upload_dir=tmp_path / "uploads", output_dir=tmp_path / "outputs")

    class StubDetector:
        pass

    app.state.detector_service = StubDetector()
    first = get_detector_service(app)
    second = get_detector_service(app)

    assert first is second


def test_infer_with_real_models_and_sample_image(
    tmp_path: Path, real_sample_image_path: Path
):
    app = create_app(upload_dir=tmp_path / "uploads", output_dir=tmp_path / "outputs")
    client = TestClient(app)

    with real_sample_image_path.open("rb") as image_file:
        response = client.post(
            "/infer",
            files={
                "image": (
                    real_sample_image_path.name,
                    io.BytesIO(image_file.read()),
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()
    request_id = payload["request_id"]
    output_dir = app.state.output_dir / request_id

    assert payload["image"]["filename"] == real_sample_image_path.name
    assert len(payload["detections"]) > 0
    assert output_dir.joinpath("annotated.png").exists()
    assert output_dir.joinpath("result.json").exists()
    assert all("group_id" in detection for detection in payload["detections"])
