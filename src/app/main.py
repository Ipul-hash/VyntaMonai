from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.app.config import settings
from src.app.api.v1.endpoints import router as v1_router
from src.app.api.v1.viewer import router as viewer_router
from src.app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Starting up VyntaMonai Service on CPU...")
    # Ensure required output directories exist
    settings.OUTPUT_SEG_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_SR_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down VyntaMonai Service...")


app = FastAPI(
    title="VyntaMonai AI Microservice & Medical Viewer",
    description="Local CPU-optimized Medical AI pipeline for 3D CT Segmentation and DICOM-SEG generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for local web viewer integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api")
app.include_router(viewer_router, prefix="/api/v1")

STATIC_DIR = Path(__file__).resolve().parent.parent / "viewer" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
@app.get("/viewer")
async def serve_viewer():
    """Serve custom HTML DICOM Web Viewer."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Viewer HTML not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
