from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from api import router
from src.config import OUTPUTS_DIR, UPLOADS_DIR
from src.services.detector import DetectorService
from src.services.grouping import GroupingService
from src.services.visualization import VisualizationService
from src.utils import ensure_dir


def create_app(
    *,
    detector_service: DetectorService | None = None,
    grouping_service: GroupingService | None = None,
    visualization_service: VisualizationService | None = None,
    upload_dir=None,
    output_dir=None,
) -> FastAPI:
    app = FastAPI(title="Infilect Shelf Grouping Demo")
    app.state.upload_dir = upload_dir or UPLOADS_DIR
    app.state.output_dir = output_dir or OUTPUTS_DIR
    app.state.detector_service = detector_service
    app.state.grouping_service = grouping_service or GroupingService()
    app.state.visualization_service = visualization_service or VisualizationService()

    ensure_dir(app.state.upload_dir)
    ensure_dir(app.state.output_dir)

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=False)
