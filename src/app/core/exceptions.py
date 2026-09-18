class MedicalImagingError(Exception):
    """Base exception class for medical imaging pipeline."""
    pass


class DICOMValidationError(MedicalImagingError):
    """Raised when a DICOM dataset fails clinical or spatial validation."""
    pass


class SpatialInconsistencyError(DICOMValidationError):
    """Raised when slice spacing, orientation, or positioning is inconsistent across a series."""
    pass


class MissingRequiredTagError(DICOMValidationError):
    """Raised when mandatory DICOM tags are missing from an instance."""
    pass


class ModelInferenceError(MedicalImagingError):
    """Raised during neural network execution failure."""
    pass


class SegmentationExportError(MedicalImagingError):
    """Raised when highdicom fails to encapsulate mask to DICOM-SEG."""
    pass
