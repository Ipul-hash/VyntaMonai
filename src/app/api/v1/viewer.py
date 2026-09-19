import io
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import pydicom

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import HTMLResponse

from src.app.config import settings
from src.app.core.logger import logger
from src.app.services.dicom_reader import DICOMSeriesReader
from src.app.services.pipeline import MedicalAIPipeline

router = APIRouter(prefix="/viewer", tags=["HTML Viewer Backend"])

# Volume cache to ensure ultra-fast slice rendering (< 5ms per request)
_volume_cache: Dict[str, Dict[str, Any]] = {}
_pipeline_singleton: Optional[MedicalAIPipeline] = None


def get_cached_study(study_name: str) -> Dict[str, Any]:
    """Retrieve or load 3D CT volume into RAM cache."""
    if study_name in _volume_cache:
        return _volume_cache[study_name]

    study_dir = settings.SAMPLES_DIR / study_name
    if not study_dir.exists() or not study_dir.is_dir():
        # Fallback to direct /mnt/PACS-DATA storage path
        pacs_data_path = Path("/mnt/PACS-DATA") / study_name
        if pacs_data_path.exists() and pacs_data_path.is_dir():
            study_dir = pacs_data_path
        else:
            raise HTTPException(status_code=404, detail=f"Study '{study_name}' not found.")

    logger.info(f"Caching volume for viewer: {study_name}...")
    volume_hu, sorted_datasets, meta = DICOMSeriesReader.load_series_from_directory(study_dir)
    
    # Check if DICOM-SEG exists for this study
    seg_file_candidates = [
        settings.OUTPUT_SEG_DIR / f"{study_name}_lung_seg.dcm" if study_name == "pasien_01" else None,
        settings.OUTPUT_SEG_DIR / f"{study_name}_seg.dcm",
        settings.OUTPUT_SEG_DIR / "spleen_segmentation.dcm" if study_name == "sample_ct_abdomen" else None,
    ]
    seg_mask = None
    organ_label = "Spleen"
    organ_volume_ml = 0.0

    for cand in seg_file_candidates:
        if cand and cand.exists():
            try:
                seg_ds = pydicom.dcmread(str(cand))
                raw_pixels = seg_ds.pixel_array
                if raw_pixels.ndim == 4:
                    raw_pixels = raw_pixels[0]
                
                # Check frame count
                if raw_pixels.shape[0] == volume_hu.shape[0]:
                    seg_mask = (raw_pixels > 0).astype(np.uint8)
                elif hasattr(seg_ds, "PerFrameFunctionalGroupsSequence"):
                    full_mask = np.zeros(volume_hu.shape, dtype=np.uint8)
                    sop_to_idx = {ds.SOPInstanceUID: i for i, ds in enumerate(sorted_datasets)}
                    for f_idx, fg in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
                        try:
                            ref = fg.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
                            if ref in sop_to_idx:
                                full_mask[sop_to_idx[ref]] = (raw_pixels[f_idx] > 0).astype(np.uint8)
                        except Exception:
                            pass
                    seg_mask = full_mask

                if hasattr(seg_ds, "SegmentSequence") and len(seg_ds.SegmentSequence) > 0:
                    organ_label = getattr(seg_ds.SegmentSequence[0], "SegmentLabel", "Organ")
                elif "lung" in cand.name:
                    organ_label = "Bilateral Lungs"

                dx, dy, dz = meta["voxel_spacing"]
                if seg_mask is not None:
                    organ_volume_ml = round(float(np.sum(seg_mask > 0)) * dx * dy * dz / 1000.0, 2)

                logger.info(f"Loaded DICOM-SEG overlay for {study_name} from {cand.name} ({organ_label}: {organ_volume_ml} mL)")
                break
            except Exception as e:
                logger.warning(f"Could not load SEG overlay: {e}")

    cached_data = {
        "volume_hu": volume_hu,
        "datasets": sorted_datasets,
        "meta": meta,
        "seg_mask": seg_mask,
        "organ_label": organ_label,
        "organ_volume_ml": organ_volume_ml,
    }
    _volume_cache[study_name] = cached_data
    return cached_data


@router.get("/studies")
async def list_available_studies() -> List[str]:
    """List available DICOM study directories in data/samples/."""
    studies = []
    if settings.SAMPLES_DIR.exists():
        for d in settings.SAMPLES_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                # verify it has dicom files
                if any(d.iterdir()):
                    studies.append(d.name)
    return sorted(studies)


@router.get("/series/{study_name}")
async def get_series_info(study_name: str) -> Dict[str, Any]:
    """Get spatial metadata and slice parameters for the web viewer."""
    cached = get_cached_study(study_name)
    datasets = cached["datasets"]
    meta = cached["meta"]
    
    z_positions = [float(ds.ImagePositionPatient[2]) for ds in datasets]

    return {
        "study_name": study_name,
        "patient_id": meta["patient_id"],
        "patient_name": meta["patient_name"],
        "modality": "CT",
        "num_slices": len(datasets),
        "rows": datasets[0].Rows,
        "columns": datasets[0].Columns,
        "voxel_spacing": meta["voxel_spacing"],
        "slice_thickness": getattr(datasets[0], "SliceThickness", meta["voxel_spacing"][2]),
        "z_positions": z_positions,
        "has_seg": cached["seg_mask"] is not None,
        "organ_name": cached["organ_label"],
        "organ_volume_ml": cached["organ_volume_ml"],
    }


@router.get("/slice/{study_name}/{slice_idx}")
async def render_ct_slice(
    study_name: str,
    slice_idx: int,
    wl: float = Query(40.0, description="Window Level (Center) in HU"),
    ww: float = Query(350.0, description="Window Width in HU"),
):
    """Render a single CT slice as PNG with real-time Window/Level adjustments."""
    cached = get_cached_study(study_name)
    volume_hu = cached["volume_hu"]
    num_slices = volume_hu.shape[0]

    if slice_idx < 0 or slice_idx >= num_slices:
        raise HTTPException(status_code=400, detail=f"Invalid slice index {slice_idx}. Total slices: {num_slices}")

    slice_hu = volume_hu[slice_idx]

    # Standard DICOM Window/Level linear transformation:
    # min_hu = wl - ww/2, max_hu = wl + ww/2
    min_hu = wl - (ww / 2.0)
    max_hu = wl + (ww / 2.0)

    # Normalize to 0-255 uint8 grayscale
    clamped = np.clip(slice_hu, min_hu, max_hu)
    if max_hu > min_hu:
        normalized = ((clamped - min_hu) / (max_hu - min_hu) * 255.0).astype(np.uint8)
    else:
        normalized = np.zeros_like(clamped, dtype=np.uint8)

    img = Image.fromarray(normalized, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


@router.get("/mask/{study_name}/{slice_idx}")
async def render_mask_slice(
    study_name: str,
    slice_idx: int,
    color: str = Query("red", description="Overlay color: red, green, cyan, yellow"),
):
    """Render the AI segmentation mask as a transparent RGBA PNG."""
    cached = get_cached_study(study_name)
    seg_mask = cached["seg_mask"]
    num_slices = cached["volume_hu"].shape[0]

    if slice_idx < 0 or slice_idx >= num_slices:
        raise HTTPException(status_code=400, detail=f"Invalid slice index {slice_idx}")

    h, w = cached["volume_hu"].shape[1], cached["volume_hu"].shape[2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if seg_mask is not None:
        mask_slice = seg_mask[slice_idx]
        pos = mask_slice > 0
        if color == "green":
            rgba[pos] = [30, 255, 100, 255]
        elif color == "cyan":
            rgba[pos] = [0, 220, 255, 255]
        elif color == "yellow":
            rgba[pos] = [255, 215, 0, 255]
        else:  # Default Red
            rgba[pos] = [255, 45, 85, 255]

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


@router.post("/run-ai/{study_name}")
async def trigger_ai_pipeline(study_name: str) -> Dict[str, Any]:
    """Trigger AI inference on a study and update cached DICOM-SEG overlay."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = MedicalAIPipeline()

    study_dir = settings.SAMPLES_DIR / study_name
    if not study_dir.exists():
        raise HTTPException(status_code=404, detail=f"Study '{study_name}' not found.")

    output_filename = f"{study_name}_seg.dcm"
    result = _pipeline_singleton.run_on_directory(study_dir, output_filename=output_filename)

    # Invalidate cache so viewer picks up new mask
    if study_name in _volume_cache:
        del _volume_cache[study_name]
    get_cached_study(study_name)

    return result


# Detection Findings Cache
_detection_cache: Dict[str, List[Dict[str, Any]]] = {}


@router.get("/detections/{study_name}")
async def get_study_detections(study_name: str) -> Dict[str, Any]:
    """Retrieve detected pulmonary nodules for the study."""
    if study_name in _detection_cache:
        return {
            "study_name": study_name,
            "total_nodules": len(_detection_cache[study_name]),
            "findings": _detection_cache[study_name],
        }

    # Automatically run detection if not cached
    return await run_study_detection(study_name)


@router.post("/detect/{study_name}")
async def run_study_detection(study_name: str) -> Dict[str, Any]:
    """Run Level 2 3D Pulmonary Nodule Detection (CADe) and save DICOM-SR."""
    from src.app.services.detection_engine import PulmonaryNoduleDetector
    from src.app.services.sr_builder import DICOMSRBuilder

    cached = get_cached_study(study_name)
    volume_hu = cached["volume_hu"]
    datasets = cached["datasets"]
    meta = cached["meta"]
    lung_mask = cached["seg_mask"]

    detector = PulmonaryNoduleDetector(
        min_diameter_mm=settings.CAD_MIN_DIAMETER_MM,
        min_confidence=settings.CAD_MIN_CONFIDENCE,
    )
    candidates = detector.detect_nodules(
        volume_hu=volume_hu,
        sorted_datasets=datasets,
        spatial_meta=meta,
        lung_mask=lung_mask,
    )

    findings_json = [c.to_dict() for c in candidates]
    _detection_cache[study_name] = findings_json

    # Save to DICOM-SR
    sr_dir = settings.OUTPUT_SR_DIR
    sr_dir.mkdir(parents=True, exist_ok=True)
    sr_path = sr_dir / f"{study_name}_sr.dcm"
    try:
        DICOMSRBuilder.create_dicom_sr(candidates, datasets, sr_path)
    except Exception as e:
        logger.warning(f"Failed to export DICOM-SR: {e}")

    return {
        "study_name": study_name,
        "total_nodules": len(findings_json),
        "findings": findings_json,
        "dicom_sr_file": str(sr_path) if sr_path.exists() else None,
    }

