import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from src.app.core.logger import logger
from src.app.config import settings
from src.app.services.dicom_reader import DICOMSeriesReader
from src.app.services.preprocessor import MedicalPreprocessor
from src.app.services.inference import CPUInferenceEngine
from src.app.services.seg_builder import DICOMSEGBuilder


class MedicalAIPipeline:
    """
    End-to-End Orchestrator for 3D DICOM AI Processing and Standard DICOM-SEG Generation.
    Decoupled and fully executable in local file mode or custom PACS integration mode.
    """

    def __init__(
        self,
        model_path: Optional[Path] = settings.SPLEEN_MODEL_PATH,
        output_dir: Path = settings.OUTPUT_SEG_DIR,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing Medical AI Pipeline modules...")
        self.preprocessor = MedicalPreprocessor()
        self.inference_engine = CPUInferenceEngine(model_path=model_path)
        logger.info("Pipeline ready.")

    def run_on_directory(
        self,
        series_dir: Path,
        output_filename: str = "spleen_segmentation.dcm",
    ) -> Dict[str, Any]:
        """
        Execute pipeline on a local DICOM series directory.

        Args:
            series_dir: Directory containing CT DICOM slices (.dcm)
            output_filename: Name of the generated DICOM-SEG file

        Returns:
            Dict containing execution metrics, clinical measurements, and output path.
        """
        start_time = time.time()
        logger.info(f"=== Starting Medical AI Pipeline for directory: {series_dir} ===")

        # Step 1: Read and validate DICOM series
        t0 = time.time()
        volume_hu, sorted_datasets, spatial_meta = DICOMSeriesReader.load_series_from_directory(series_dir)
        read_time = time.time() - t0
        logger.info(f"[Stage 1/5] Ingested & sorted {len(sorted_datasets)} slices in {read_time:.2f}s")

        # Step 2: Preprocess & resample to isotropic/target spacing
        t0 = time.time()
        tensor_resampled, pre_meta = self.preprocessor.preprocess(
            volume_hu=volume_hu,
            voxel_spacing=spatial_meta["voxel_spacing"],
        )
        preprocess_time = time.time() - t0
        logger.info(f"[Stage 2/5] Preprocessed and resampled in {preprocess_time:.2f}s")

        # Step 3: CPU Model Inference
        t0 = time.time()
        pred_mask_resampled = self.inference_engine.infer(tensor_resampled)
        infer_time = time.time() - t0
        logger.info(f"[Stage 3/5] CPU Inference finished in {infer_time:.2f}s")

        # Step 4: Post-process & restore mask to original DICOM slice grid
        t0 = time.time()
        restored_mask = self.preprocessor.postprocess(pred_mask_resampled, pre_meta)
        postprocess_time = time.time() - t0
        logger.info(f"[Stage 4/5] Restored mask to original coordinate grid in {postprocess_time:.2f}s")

        # Step 5: Encapsulate into DICOM-SEG via highdicom
        t0 = time.time()
        output_seg_path = self.output_dir / output_filename
        DICOMSEGBuilder.create_dicom_seg(
            mask=restored_mask,
            source_datasets=sorted_datasets,
            output_path=output_seg_path,
        )
        export_time = time.time() - t0
        logger.info(f"[Stage 5/5] Encapsulated DICOM-SEG in {export_time:.2f}s")

        total_duration = time.time() - start_time

        # Calculate clinical metrics (Spleen Volume in cm³ / mL)
        dx, dy, dz = spatial_meta["voxel_spacing"]
        voxel_vol_cm3 = (dx * dy * dz) / 1000.0
        total_positive_voxels = int(np.sum(restored_mask))
        organ_volume_ml = round(total_positive_voxels * voxel_vol_cm3, 2)

        results = {
            "status": "success",
            "study_instance_uid": spatial_meta["study_instance_uid"],
            "series_instance_uid": spatial_meta["series_instance_uid"],
            "patient_id": spatial_meta["patient_id"],
            "num_slices": len(sorted_datasets),
            "output_seg_file": str(output_seg_path),
            "measurements": {
                "segmented_organ": settings.ORGAN_NAME,
                "positive_voxels": total_positive_voxels,
                "calculated_volume_ml": organ_volume_ml,
            },
            "timings_sec": {
                "dicom_read": round(read_time, 2),
                "preprocessing": round(preprocess_time, 2),
                "inference_cpu": round(infer_time, 2),
                "postprocessing": round(postprocess_time, 2),
                "dicom_seg_export": round(export_time, 2),
                "total_pipeline": round(total_duration, 2),
            },
        }

        logger.info(f"=== Pipeline Completed Successfully in {total_duration:.2f}s ===")
        logger.info(f"Estimated {settings.ORGAN_NAME} Volume: {organ_volume_ml} mL (cm³)")
        return results
