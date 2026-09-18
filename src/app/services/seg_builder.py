from pathlib import Path
from typing import List
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import generate_uid

import highdicom as hd
from highdicom.seg.sop import Segmentation
from highdicom.seg.content import SegmentDescription
from highdicom.seg.enum import SegmentationTypeValues, SegmentAlgorithmTypeValues
from highdicom.content import AlgorithmIdentificationSequence

from src.app.core.logger import logger
from src.app.core.exceptions import SegmentationExportError
from src.app.config import settings


class DICOMSEGBuilder:
    """
    Encapsulates 3D binary segmentation mask into standard DICOM-SEG SOP Instance
    using highdicom. Preserves spatial frame-of-reference and links source slice UIDs.
    """

    @classmethod
    def create_dicom_seg(
        cls,
        mask: np.ndarray,
        source_datasets: List[Dataset],
        output_path: Path,
        segment_label: str = settings.ORGAN_NAME,
        sct_code: str = settings.ORGAN_SCT_CODE,
        sct_meaning: str = settings.ORGAN_SCT_MEANING,
    ) -> Path:
        """
        Build and save a multi-frame DICOM-SEG object.

        Args:
            mask: 3D uint8 binary array (Z, Y, X) matching source_datasets spatial order
            source_datasets: List of original CT pydicom Datasets in same Z-order
            output_path: Target .dcm file path to save the DICOM-SEG
            segment_label: Label name of the segmented structure
            sct_code: SNOMED-CT code for segmented anatomy
            sct_meaning: SNOMED-CT meaning

        Returns:
            output_path: Path to saved DICOM-SEG
        """
        try:
            if mask.ndim != 3:
                raise SegmentationExportError(f"Expected 3D mask (Z, Y, X), got shape {mask.shape}")

            num_slices, rows, cols = mask.shape
            if num_slices != len(source_datasets):
                raise SegmentationExportError(
                    f"Mask slice count ({num_slices}) does not match source DICOM count ({len(source_datasets)})"
                )

            # Ensure standard DICOM Type 2 attributes exist across source slices
            for ds in source_datasets:
                if not hasattr(ds, "AccessionNumber"):
                    ds.AccessionNumber = ""
                if not hasattr(ds, "ReferringPhysicianName"):
                    ds.ReferringPhysicianName = ""
                if not hasattr(ds, "PatientBirthDate"):
                    ds.PatientBirthDate = ""
                if not hasattr(ds, "PatientSex"):
                    ds.PatientSex = ""
                if not hasattr(ds, "StudyID"):
                    ds.StudyID = "1"

            # 1. Standard SNOMED CT terminology definition
            # Category: Anatomical Structure (SCT 49755003 or CID 7150)
            category_code = Code(
                value="49755003",
                scheme_designator="SCT",
                meaning="Morphologically Abnormal Structure",
            ) if segment_label.lower() in ["tumor", "lesion"] else Code(
                value="123037004",
                scheme_designator="SCT",
                meaning="Anatomical Structure",
            )

            # Type: Target Organ Code (e.g. Spleen: 78961009)
            type_code = Code(
                value=sct_code,
                scheme_designator="SCT",
                meaning=sct_meaning,
            )

            # Algorithm Identification
            algo_family = Code(
                value="113690",
                scheme_designator="DCM",
                meaning="Artificial Intelligence",
            )
            algo_id = AlgorithmIdentificationSequence(
                name="VyntaMonai-CPU-UNet",
                version="1.0.0",
                family=algo_family,
                source="MONAI / VyntaCAD",
            )

            # Segment Description
            segment_desc = SegmentDescription(
                segment_number=1,
                segment_label=segment_label,
                segmented_property_category=category_code,
                segmented_property_type=type_code,
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algo_id,
            )

            # Ensure mask is uint8 boolean (0 or 1)
            pixel_array = (mask > 0).astype(np.uint8)

            # 2. Build highdicom Segmentation object
            # Note: highdicom expects pixel_array shape (num_slices, rows, cols) for single segment
            seg_dataset = Segmentation(
                source_images=source_datasets,
                pixel_array=pixel_array,
                segmentation_type=SegmentationTypeValues.BINARY,
                segment_descriptions=[segment_desc],
                series_instance_uid=generate_uid(),
                series_number=5000,
                sop_instance_uid=generate_uid(),
                instance_number=1,
                manufacturer="VyntaCAD",
                manufacturer_model_name="VyntaMonai-Engine",
                software_versions="1.0.0",
                device_serial_number="CPU-LOC-01",
                series_description=settings.SERIES_DESCRIPTION,
                omit_empty_frames=False,
            )

            # 3. Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            seg_dataset.save_as(str(output_path))
            logger.info(f"Successfully saved DICOM-SEG to: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate DICOM-SEG: {str(e)}")
            raise SegmentationExportError(f"DICOM-SEG generation error: {str(e)}") from e
