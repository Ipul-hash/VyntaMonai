from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from src.app.config import settings
from src.app.core.logger import logger
from src.app.services.pipeline import MedicalAIPipeline

router = APIRouter(prefix="/v1", tags=["Medical AI Inference"])

# Lazy-loaded pipeline singleton
_pipeline: Optional[MedicalAIPipeline] = None


def get_pipeline() -> MedicalAIPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MedicalAIPipeline()
    return _pipeline


class HealthResponse(BaseModel):
    status: str
    engine: str
    device: str
    threads: int
    target_spacing: tuple[float, float, float]
    target_organ: str


class DirectoryInferenceRequest(BaseModel):
    directory_path: str = Field(
        ...,
        description="Absolute or relative path to the directory containing CT DICOM slices",
        examples=["data/samples/sample_ct_series"],
    )
    output_filename: Optional[str] = Field(
        default="spleen_segmentation.dcm",
        description="Output DICOM-SEG filename",
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check returning AI engine state and CPU constraints."""
    return HealthResponse(
        status="healthy",
        engine="MONAI-FastAPI-CPU",
        device=settings.DEVICE,
        threads=settings.NUM_CPU_THREADS,
        target_spacing=settings.TARGET_SPACING,
        target_organ=settings.ORGAN_NAME,
    )


@router.post("/infer/local-directory")
async def infer_local_directory(payload: DirectoryInferenceRequest) -> Dict[str, Any]:
    """
    Run 3D Spleen Segmentation pipeline on a local DICOM directory.
    Outputs standard DICOM-SEG file to data/output_seg/.
    """
    series_dir = Path(payload.directory_path)
    if not series_dir.exists() or not series_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Directory '{payload.directory_path}' does not exist or is not a directory.",
        )

    try:
        pipeline = get_pipeline()
        result = pipeline.run_on_directory(
            series_dir=series_dir,
            output_filename=payload.output_filename or "spleen_segmentation.dcm",
        )
        return result
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))
