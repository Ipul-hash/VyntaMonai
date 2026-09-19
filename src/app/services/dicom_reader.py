from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pydicom
from pydicom.dataset import Dataset

from src.app.core.logger import logger
from src.app.core.exceptions import (
    DICOMValidationError,
    SpatialInconsistencyError,
    MissingRequiredTagError,
)


class DICOMSeriesReader:
    """
    Standard-compliant Medical Imaging Reader for 3D DICOM CT Series.
    Enforces clinical spatial alignment, tag validation, and Hounsfield Unit (HU) conversion.
    """

    MANDATORY_TAGS = [
        "SOPInstanceUID",
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "PixelSpacing",
        "RescaleSlope",
        "RescaleIntercept",
    ]

    @classmethod
    def validate_dataset(cls, ds: Dataset, filepath: Path) -> None:
        """Validate mandatory DICOM tags for spatial reconstruction."""
        for tag in cls.MANDATORY_TAGS:
            if tag not in ds:
                raise MissingRequiredTagError(
                    f"File {filepath.name} missing mandatory DICOM tag: {tag}"
                )

    @classmethod
    def load_series_from_directory(
        cls, directory: Path
    ) -> Tuple[np.ndarray, List[Dataset], Dict[str, Any]]:
        """
        Load, validate, and spatially sort a CT DICOM series from a directory.

        Returns:
            volume_hu: 3D numpy array (Z, Y, X) in Hounsfield Units (HU)
            sorted_datasets: List of pydicom Dataset objects sorted spatially
            spatial_metadata: Dictionary containing voxel spacing, origin, and orientation
        """
        # 1. Recursively discover all DICOM candidates in directory & subdirectories (PACS format)
        dicom_files = [
            f for f in directory.rglob("*")
            if f.is_file() and not f.name.startswith(".") and not f.name.startswith("__")
        ]
        if not dicom_files:
            raise DICOMValidationError(f"No DICOM files found in directory {directory}")

        datasets_by_series: Dict[str, List[Dataset]] = {}
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(str(file_path), force=False)
                # Ensure dataset has PixelData and basic spatial attributes
                if hasattr(ds, "PixelData") and hasattr(ds, "ImagePositionPatient"):
                    cls.validate_dataset(ds, file_path)
                    s_uid = str(ds.SeriesInstanceUID)
                    if s_uid not in datasets_by_series:
                        datasets_by_series[s_uid] = []
                    datasets_by_series[s_uid].append(ds)
            except Exception as e:
                # Silently skip non-DICOM or non-image assets
                continue

        if not datasets_by_series:
            raise DICOMValidationError(f"No valid tomographic CT image datasets found in {directory}")

        # 2. Select optimal 3D series for pulmonary analysis:
        # Priority A: Series Description containing keywords ('lung', 'paru', 'thorax', 'chest', 'axial') with >= 10 slices
        # Priority B: Series with maximum slice count (most granular 3D volume, filters out 2D scouts)
        def score_series(item: Tuple[str, List[Dataset]]) -> Tuple[int, int]:
            _, d_list = item
            count = len(d_list)
            desc = str(getattr(d_list[0], "SeriesDescription", "")).lower()
            is_lung_targeted = any(kw in desc for kw in ["lung", "paru", "thorax", "chest", "axial", "pulmo"])
            target_score = 1 if (is_lung_targeted and count >= 10) else 0
            return (target_score, count)

        best_series_uid, datasets = max(datasets_by_series.items(), key=score_series)
        series_desc = getattr(datasets[0], "SeriesDescription", "CT Series")
        logger.info(
            f"Selected Series UID: {best_series_uid} ('{series_desc}') with {len(datasets)} slices (out of {len(datasets_by_series)} series)."
        )

        if len(datasets) < 2:
            raise DICOMValidationError(
                f"Insufficient valid 3D CT slices found in selected series ({len(datasets)}). Need >= 2 slices."
            )

        # 3. Extract and check ImageOrientationPatient (IOP)
        ref_iop = [float(x) for x in datasets[0].ImageOrientationPatient]
        r_vec = np.array(ref_iop[:3])  # Row direction cosine
        c_vec = np.array(ref_iop[3:])  # Column direction cosine
        n_vec = np.cross(r_vec, c_vec)  # Slice normal direction vector

        # Sort slices by projection along slice normal vector
        slice_positions = []
        for ds in datasets:
            iop = [float(x) for x in ds.ImageOrientationPatient]
            if not np.allclose(iop, ref_iop, atol=1e-2):
                continue  # Skip divergent scout/topogram slices
            ipp = np.array([float(x) for x in ds.ImagePositionPatient])
            proj = np.dot(ipp, n_vec)
            slice_positions.append((proj, ds))

        if len(slice_positions) < 2:
            raise SpatialInconsistencyError("Insufficient slices with consistent 3D spatial orientation.")

        # Deduplicate identical slice positions (if any) and sort ascending along slice normal
        unique_positions: Dict[float, Tuple[float, Dataset]] = {}
        for proj, ds in slice_positions:
            proj_key = round(proj, 2)
            if proj_key not in unique_positions:
                unique_positions[proj_key] = (proj, ds)

        sorted_items = sorted(unique_positions.values(), key=lambda x: x[0])
        sorted_datasets = [item[1] for item in sorted_items]
        projections = [item[0] for item in sorted_items]

        # 4. Calculate spatial voxel spacing
        pixel_spacing = [float(x) for x in sorted_datasets[0].PixelSpacing]
        dx, dy = pixel_spacing[1], pixel_spacing[0]  # in mm (column, row)
        
        diffs = np.diff(projections)
        positive_diffs = diffs[diffs > 0.01]
        if len(positive_diffs) > 0:
            slice_spacing = float(np.median(positive_diffs))
        else:
            slice_spacing = float(getattr(sorted_datasets[0], "SliceThickness", 1.0))

        dz = slice_spacing
        voxel_spacing = (dx, dy, dz)  # (X, Y, Z) in mm

        logger.info(
            f"Loaded {len(sorted_datasets)} sorted slices. Spacing (X, Y, Z): {voxel_spacing[0]:.2f}x{voxel_spacing[1]:.2f}x{voxel_spacing[2]:.2f} mm"
        )

        # 5. Construct 3D Volume and apply RescaleSlope & RescaleIntercept to HU
        slices_hu = []
        for ds in sorted_datasets:
            pixel_array = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu = pixel_array * slope + intercept
            slices_hu.append(hu)

        volume_hu = np.stack(slices_hu, axis=0)  # Shape: (Z, Y, X)

        spatial_metadata = {
            "voxel_spacing": voxel_spacing,  # (dx, dy, dz)
            "image_orientation_patient": ref_iop,
            "origin_ipp": [float(x) for x in sorted_datasets[0].ImagePositionPatient],
            "frame_of_reference_uid": getattr(sorted_datasets[0], "FrameOfReferenceUID", ""),
            "study_instance_uid": sorted_datasets[0].StudyInstanceUID,
            "series_instance_uid": best_series_uid,
            "patient_id": getattr(sorted_datasets[0], "PatientID", "UNKNOWN"),
            "patient_name": str(getattr(sorted_datasets[0], "PatientName", "Anonymous")),
        }

        return volume_hu, sorted_datasets, spatial_metadata
