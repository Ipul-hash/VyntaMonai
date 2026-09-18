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
        dicom_files = [
            f for f in directory.iterdir() if f.is_file() and not f.name.startswith(".")
        ]
        if not dicom_files:
            raise DICOMValidationError(f"No files found in directory {directory}")

        datasets: List[Dataset] = []
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(str(file_path), force=False)
                # Skip non-image or secondary capture if needed
                if hasattr(ds, "PixelData"):
                    cls.validate_dataset(ds, file_path)
                    datasets.append(ds)
            except Exception as e:
                logger.warning(f"Skipping non-DICOM or unreadable file: {file_path.name} ({e})")

        if len(datasets) < 2:
            raise DICOMValidationError(
                f"Insufficient valid 3D CT slices found ({len(datasets)}). Need >= 2 slices."
            )

        # 1. Verify consistent SeriesInstanceUID
        series_uids = {ds.SeriesInstanceUID for ds in datasets}
        if len(series_uids) > 1:
            raise DICOMValidationError(
                f"Directory contains multiple SeriesInstanceUIDs: {series_uids}"
            )

        # 2. Extract and check ImageOrientationPatient (IOP)
        ref_iop = [float(x) for x in datasets[0].ImageOrientationPatient]
        r_vec = np.array(ref_iop[:3])  # Row direction cosine
        c_vec = np.array(ref_iop[3:])  # Column direction cosine
        n_vec = np.cross(r_vec, c_vec)  # Slice normal direction vector

        # Sort slices by projection along the slice normal (Patient coordinate frame)
        slice_positions = []
        for ds in datasets:
            iop = [float(x) for x in ds.ImageOrientationPatient]
            if not np.allclose(iop, ref_iop, atol=1e-3):
                raise SpatialInconsistencyError(
                    f"Slice {ds.SOPInstanceUID} has divergent ImageOrientationPatient"
                )
            ipp = np.array([float(x) for x in ds.ImagePositionPatient])
            proj = np.dot(ipp, n_vec)
            slice_positions.append((proj, ds))

        # Sort ascending along slice normal
        slice_positions.sort(key=lambda x: x[0])
        sorted_datasets = [item[1] for item in slice_positions]
        projections = [item[0] for item in slice_positions]

        # 3. Calculate spatial spacing
        pixel_spacing = [float(x) for x in sorted_datasets[0].PixelSpacing]
        dx, dy = pixel_spacing[1], pixel_spacing[0]  # in mm (column, row)
        
        diffs = np.diff(projections)
        slice_spacing = float(np.median(diffs))
        if slice_spacing <= 0:
            raise SpatialInconsistencyError("Calculated slice spacing is non-positive.")

        dz = slice_spacing
        voxel_spacing = (dx, dy, dz)  # (X, Y, Z) in mm

        logger.info(
            f"Loaded {len(sorted_datasets)} slices. Spacing (X, Y, Z): {voxel_spacing[0]:.2f}x{voxel_spacing[1]:.2f}x{voxel_spacing[2]:.2f} mm"
        )

        # 4. Construct 3D Volume and apply RescaleSlope & RescaleIntercept to HU
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
            "series_instance_uid": sorted_datasets[0].SeriesInstanceUID,
            "patient_id": getattr(sorted_datasets[0], "PatientID", "UNKNOWN"),
            "patient_name": str(getattr(sorted_datasets[0], "PatientName", "Anonymous")),
        }

        return volume_hu, sorted_datasets, spatial_metadata
