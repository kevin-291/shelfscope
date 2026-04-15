from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.config import ALLOWED_EXTENSIONS
from src.services.detector import DetectorService
from src.services.grouping import GroupingService
from src.services.visualization import VisualizationService
from src.utils import ensure_dir, make_request_id


templates = Jinja2Templates(directory="templates")
router = APIRouter()


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def save_upload_bytes(upload_path: Path, content: bytes) -> None:
    upload_path.write_bytes(content)


def write_response_json(output_dir: Path, response: dict[str, Any]) -> None:
    json_path = output_dir / "result.json"
    json_path.write_text(json.dumps(response, indent=2), encoding="utf-8")


def get_detector_service(app: FastAPI) -> DetectorService:
    detector_service = getattr(app.state, "detector_service", None)
    if detector_service is None:
        detector_service = DetectorService()
        app.state.detector_service = detector_service
    return detector_service


def get_grouping_service(app: FastAPI) -> GroupingService:
    return app.state.grouping_service


def get_visualization_service(app: FastAPI) -> VisualizationService:
    return app.state.visualization_service


async def run_inference_pipeline(
    *,
    image: Image.Image,
    uploaded_filename: str,
    request_id: str,
    output_dir: Path,
    detector_service: DetectorService,
    grouping_service: GroupingService,
    visualization_service: VisualizationService,
) -> dict[str, Any]:
    detector_result = await run_in_threadpool(detector_service.detect, image)
    detections = await run_in_threadpool(
        grouping_service.assign_groups, image, detector_result.detections
    )

    response = {
        "request_id": request_id,
        "image": {
            "filename": uploaded_filename,
            "width": detector_result.image_size[0],
            "height": detector_result.image_size[1],
        },
        "visualization_url": f"/outputs/{request_id}/annotated.png",
        "json_url": f"/outputs/{request_id}/result.json",
        "detections": detections,
    }

    annotated_path = output_dir / "annotated.png"
    await asyncio.gather(
        run_in_threadpool(
            visualization_service.render, image, detections, annotated_path
        ),
        run_in_threadpool(write_response_json, output_dir, response),
    )
    return response


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@router.get("/outputs/{request_id}/{filename:path}")
async def get_output_file(request_id: str, filename: str, request: Request):
    target = request.app.state.output_dir / request_id / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(target)


@router.post("/infer")
async def infer(request: Request, image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image upload is required.")

    suffix = Path(image.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}"
        )

    request_id = make_request_id()
    upload_dir_for_request = ensure_dir(request.app.state.upload_dir / request_id)
    output_dir_for_request = ensure_dir(request.app.state.output_dir / request_id)
    image_path = upload_dir_for_request / f"input{suffix}"

    uploaded_content = await image.read()
    await run_in_threadpool(save_upload_bytes, image_path, uploaded_content)

    try:
        loaded_image = await run_in_threadpool(load_image, image_path)
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        ) from exc

    try:
        response = await run_inference_pipeline(
            image=loaded_image,
            uploaded_filename=image.filename,
            request_id=request_id,
            output_dir=output_dir_for_request,
            detector_service=get_detector_service(request.app),
            grouping_service=get_grouping_service(request.app),
            visualization_service=get_visualization_service(request.app),
        )
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc
