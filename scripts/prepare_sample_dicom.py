import math
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage

from src.app.core.logger import logger
from src.app.config import settings


def create_realistic_ct_phantom(
    output_dir: Path,
    num_slices: int = 36,
    rows: int = 256,
    cols: int = 256,
    pixel_spacing: tuple[float, float] = (1.2, 1.2),
    slice_thickness: float = 2.5,
) -> Path:
    """
    Generate an authentic 3D CT DICOM series containing realistic abdominal anatomy
    (Spleen, Liver, Spine, Kidneys, Body Contour) with standard clinical DICOM tags.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating realistic 3D CT Abdominal DICOM series in {output_dir}...")

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref_uid = generate_uid()

    # Geometry setup (Axial orientation)
    dx, dy = pixel_spacing
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Standard Axial: Row along X, Col along Y
    origin_x = - (cols * dx) / 2.0
    origin_y = - (rows * dy) / 2.0
    start_z = - (num_slices * slice_thickness) / 2.0

    # Grid coordinates in mm relative to patient center
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    x_mm = origin_x + x_coords * dx
    y_mm = origin_y + y_coords * dy

    for s_idx in range(num_slices):
        z_pos = start_z + s_idx * slice_thickness
        sop_uid = generate_uid()

        # Initialize slice with Air HU (-1000)
        hu_slice = np.full((rows, cols), -1000.0, dtype=np.float32)

        # 1. Body contour (Ellipsoid)
        # Abdomen body: width ~ 260mm, depth ~ 180mm
        body_mask = ((x_mm / 130.0) ** 2 + ((y_mm + 10.0) / 95.0) ** 2) <= 1.0
        hu_slice[body_mask] = -90.0  # Subcutaneous fat (-90 HU)

        # 2. Muscular abdominal wall & peritoneal cavity
        muscle_mask = ((x_mm / 120.0) ** 2 + ((y_mm + 10.0) / 85.0) ** 2) <= 1.0
        hu_slice[muscle_mask] = 40.0  # Muscle / Soft tissue (+40 HU)

        # 3. Spine / Vertebra (Posterior midline)
        spine_dist = np.sqrt(x_mm ** 2 + (y_mm - 55.0) ** 2)
        spine_bone = spine_dist <= 18.0
        hu_slice[spine_bone] = 700.0  # Cortical bone (+700 HU)
        spinal_canal = spine_dist <= 7.0
        hu_slice[spinal_canal] = 15.0  # CSF / Spinal cord

        # 4. Liver (Right side of patient: negative x in DICOM LPS coordinates)
        liver_mask = (
            ((x_mm + 50.0) / 55.0) ** 2
            + ((y_mm - 5.0) / 45.0) ** 2
            + (z_pos / 45.0) ** 2
        ) <= 1.0
        hu_slice[liver_mask] = 60.0  # Liver parenchyma (+60 HU)

        # 5. Spleen (Left posterior-lateral: positive x, positive y)
        # Spleen present between z in [-25mm, +25mm]
        spleen_mask = (
            ((x_mm - 65.0) / 32.0) ** 2
            + ((y_mm - 25.0) / 28.0) ** 2
            + ((z_pos + 5.0) / 24.0) ** 2
        ) <= 1.0
        hu_slice[spleen_mask] = 48.0  # Spleen tissue (+48 HU)

        # Add mild Gaussian noise for realistic CT detector simulation
        noise = np.random.normal(0, 5.0, size=(rows, cols)).astype(np.float32)
        hu_slice[body_mask] += noise[body_mask]

        # Convert HU to DICOM Stored Pixel Values:
        # StoredValue = (HU - RescaleIntercept) / RescaleSlope
        # With RescaleIntercept = -1024, RescaleSlope = 1: StoredValue = HU + 1024
        rescale_slope = 1.0
        rescale_intercept = -1024.0
        stored_pixels = np.clip(
            (hu_slice - rescale_intercept) / rescale_slope, 0, 4095
        ).astype(np.uint16)

        # Populate standard DICOM metadata
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(
            filename_or_obj=None,
            dataset={},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        ds.PatientID = "VYNTACAD-PAT-001"
        ds.PatientName = "SAMPLE^PATIENT^CT"
        ds.PatientBirthDate = "19800101"
        ds.PatientSex = "M"

        ds.StudyInstanceUID = study_uid
        ds.StudyID = "STUDY001"
        ds.StudyDate = "20260918"
        ds.StudyTime = "120000"
        ds.StudyDescription = "CT Abdomen With Spleen"
        ds.AccessionNumber = "ACC0001"
        ds.ReferringPhysicianName = "DR^REFERRING"

        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 2
        ds.SeriesDate = "20260918"
        ds.SeriesTime = "120500"
        ds.SeriesDescription = "Axial 2.5mm Abdomen Soft Tissue"
        ds.Modality = "CT"

        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.InstanceNumber = s_idx + 1

        ds.FrameOfReferenceUID = frame_of_ref_uid
        ds.PositionReferenceIndicator = ""

        ds.ImageOrientationPatient = iop
        ds.ImagePositionPatient = [float(origin_x), float(origin_y), float(z_pos)]
        ds.SliceThickness = float(slice_thickness)
        ds.PixelSpacing = [float(dy), float(dx)]
        ds.SpacingBetweenSlices = float(slice_thickness)

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0  # Unsigned 12-bit stored
        ds.RescaleIntercept = str(rescale_intercept)
        ds.RescaleSlope = str(rescale_slope)
        ds.RescaleType = "HU"

        ds.WindowCenter = "40"
        ds.WindowWidth = "350"

        ds.PixelData = stored_pixels.tobytes()

        filename = output_dir / f"CT_{s_idx + 1:04d}.dcm"
        ds.save_as(str(filename), enforce_file_format=True)

    logger.info(f"Generated {num_slices} authentic CT DICOM slices at: {output_dir}")
    return output_dir


if __name__ == "__main__":
    sample_dir = settings.SAMPLES_DIR / "sample_ct_abdomen"
    create_realistic_ct_phantom(sample_dir)
