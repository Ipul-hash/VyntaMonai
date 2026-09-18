from pathlib import Path
from typing import List
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.sr.coding import Code

from src.app.core.logger import logger
from src.app.services.detection_engine import NoduleCandidate


class DICOMSRBuilder:
    """
    Standard DICOM Structured Report (DICOM-SR) Builder.
    Encapsulates Level 2 CADe detection findings into standard Enhanced SR format (TID 1500 compatible).
    """

    ENHANCED_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.22"

    @classmethod
    def create_dicom_sr(
        cls,
        findings: List[NoduleCandidate],
        source_datasets: List[Dataset],
        output_path: Path,
    ) -> Path:
        """
        Create and save standard DICOM-SR file.

        Args:
            findings: List of detected NoduleCandidate objects
            source_datasets: List of original CT slices
            output_path: Path to save the .dcm SR file

        Returns:
            output_path: Path of the saved DICOM-SR
        """
        logger.info(f"Generating DICOM-SR with {len(findings)} nodule findings...")
        ref_ds = source_datasets[0]

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = cls.ENHANCED_SR_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(
            filename_or_obj=None,
            dataset={},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        # Patient & Study Level Attributes
        ds.PatientID = getattr(ref_ds, "PatientID", "ANONYMOUS")
        ds.PatientName = getattr(ref_ds, "PatientName", "Anonymous")
        ds.PatientBirthDate = getattr(ref_ds, "PatientBirthDate", "")
        ds.PatientSex = getattr(ref_ds, "PatientSex", "")
        ds.StudyInstanceUID = ref_ds.StudyInstanceUID
        ds.StudyID = getattr(ref_ds, "StudyID", "1")
        ds.StudyDate = getattr(ref_ds, "StudyDate", "20260918")
        ds.StudyTime = getattr(ref_ds, "StudyTime", "120000")
        ds.AccessionNumber = getattr(ref_ds, "AccessionNumber", "")
        ds.ReferringPhysicianName = getattr(ref_ds, "ReferringPhysicianName", "")

        # Series & Instance Identification
        ds.Modality = "SR"
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 6000
        ds.SeriesDescription = "VyntaMonai Level 2 CADe Pulmonary Nodule Report"
        ds.SOPClassUID = cls.ENHANCED_SR_SOP_CLASS
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = 1

        # Document Header
        ds.ValueType = "CONTAINER"
        ds.ContinuityOfContent = "SEPARATE"

        # Concept Name: Diagnostic Imaging Report (LN: 18748-4)
        concept_ds = Dataset()
        concept_ds.CodeValue = "18748-4"
        concept_ds.CodingSchemeDesignator = "LN"
        concept_ds.CodeMeaning = "Diagnostic Imaging Report"
        ds.ConceptNameCodeSequence = Sequence([concept_ds])

        # Content Sequence (Tree of Findings)
        content_items = []

        # 1. Summary Narrative Text
        summary_item = Dataset()
        summary_item.RelationshipType = "CONTAINS"
        summary_item.ValueType = "TEXT"
        summary_concept = Dataset()
        summary_concept.CodeValue = "121071"
        summary_concept.CodingSchemeDesignator = "DCM"
        summary_concept.CodeMeaning = "Finding"
        summary_item.ConceptNameCodeSequence = Sequence([summary_concept])
        summary_item.TextValue = f"VyntaMonai CADe detected {len(findings)} pulmonary nodule candidate(s)."
        content_items.append(summary_item)

        # 2. Detailed Measurement Items for Each Nodule
        for finding in findings:
            item = Dataset()
            item.RelationshipType = "CONTAINS"
            item.ValueType = "CONTAINER"
            item.ContinuityOfContent = "SEPARATE"

            item_concept = Dataset()
            item_concept.CodeValue = "125007"
            item_concept.CodingSchemeDesignator = "DCM"
            item_concept.CodeMeaning = "Measurement Group"
            item.ConceptNameCodeSequence = Sequence([item_concept])

            # Child measurements: Diameter & Volume
            child_items = []

            # Longest Diameter
            diam_item = Dataset()
            diam_item.RelationshipType = "CONTAINS"
            diam_item.ValueType = "NUM"
            diam_concept = Dataset()
            diam_concept.CodeValue = "M-02550"
            diam_concept.CodingSchemeDesignator = "SRT"
            diam_concept.CodeMeaning = "Diameter"
            diam_item.ConceptNameCodeSequence = Sequence([diam_concept])

            val_ds = Dataset()
            val_ds.NumericValue = str(finding.diameter_mm)
            unit_ds = Dataset()
            unit_ds.CodeValue = "mm"
            unit_ds.CodingSchemeDesignator = "UCUM"
            unit_ds.CodeMeaning = "millimeter"
            val_ds.MeasurementUnitsCodeSequence = Sequence([unit_ds])
            diam_item.MeasuredValueSequence = Sequence([val_ds])
            child_items.append(diam_item)

            # Confidence Score Text
            conf_item = Dataset()
            conf_item.RelationshipType = "CONTAINS"
            conf_item.ValueType = "TEXT"
            conf_concept = Dataset()
            conf_concept.CodeValue = "111001"
            conf_concept.CodingSchemeDesignator = "DCM"
            conf_concept.CodeMeaning = "Algorithm Result"
            conf_item.ConceptNameCodeSequence = Sequence([conf_concept])

            cadx_info = ""
            if finding.cadx:
                cadx_info = (
                    f" | Lung-RADS: {finding.cadx.lung_rads_category} | "
                    f"Malignancy Risk: {finding.cadx.malignancy_percent}% ({finding.cadx.risk_level}) | "
                    f"Action: {finding.cadx.clinical_recommendation}"
                )

            conf_item.TextValue = (
                f"Nodule #{finding.nodule_id} ({finding.density_type}) | "
                f"Confidence: {int(finding.confidence * 100)}% | "
                f"Slice: {finding.slice_index + 1} | "
                f"Fleischner: {finding._get_fleischner_category()}"
                f"{cadx_info}"
            )
            child_items.append(conf_item)

            # If CADx is available, add explicit Malignancy Probability NUM item
            if finding.cadx:
                mal_item = Dataset()
                mal_item.RelationshipType = "CONTAINS"
                mal_item.ValueType = "NUM"
                mal_concept = Dataset()
                mal_concept.CodeValue = "111024"
                mal_concept.CodingSchemeDesignator = "DCM"
                mal_concept.CodeMeaning = "Probability of Malignancy"
                mal_item.ConceptNameCodeSequence = Sequence([mal_concept])

                m_val = Dataset()
                m_val.NumericValue = str(finding.cadx.malignancy_percent)
                m_unit = Dataset()
                m_unit.CodeValue = "%"
                m_unit.CodingSchemeDesignator = "UCUM"
                m_unit.CodeMeaning = "percent"
                m_val.MeasurementUnitsCodeSequence = Sequence([m_unit])
                mal_item.MeasuredValueSequence = Sequence([m_val])
                child_items.append(mal_item)

            item.ContentSequence = Sequence(child_items)
            content_items.append(item)

        ds.ContentSequence = Sequence(content_items)

        # Save to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(output_path), enforce_file_format=True)
        logger.info(f"Successfully saved DICOM-SR report to: {output_path}")

        return output_path
