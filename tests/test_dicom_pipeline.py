import unittest
from pathlib import Path
import tempfile
import numpy as np
import pydicom

from scripts.prepare_sample_dicom import create_realistic_ct_phantom
from src.app.services.dicom_reader import DICOMSeriesReader
from src.app.services.preprocessor import MedicalPreprocessor
from src.app.services.seg_builder import DICOMSEGBuilder


class TestDICOMPipeline(unittest.TestCase):
    """Automated unit & integration tests for local medical imaging pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.sample_ct_dir = cls.temp_dir / "sample_ct"
        cls.output_seg_path = cls.temp_dir / "test_spleen_seg.dcm"
        
        # Generate 12 test slices
        create_realistic_ct_phantom(cls.sample_ct_dir, num_slices=12)

    def test_01_dicom_series_reader(self):
        """Test reading, spatial sorting, and HU conversion."""
        volume_hu, sorted_datasets, meta = DICOMSeriesReader.load_series_from_directory(
            self.sample_ct_dir
        )
        self.assertEqual(len(sorted_datasets), 12)
        self.assertEqual(volume_hu.shape[0], 12)
        self.assertIn("voxel_spacing", meta)
        
        # Verify Z ordering is strictly ascending
        z_positions = [float(ds.ImagePositionPatient[2]) for ds in sorted_datasets]
        self.assertEqual(z_positions, sorted(z_positions))

    def test_02_preprocessor_shape_reversibility(self):
        """Test that preprocessing resampling and postprocessing restoration preserve matrix grid."""
        volume_hu, _, meta = DICOMSeriesReader.load_series_from_directory(self.sample_ct_dir)
        preprocessor = MedicalPreprocessor()
        
        tensor, pre_meta = preprocessor.preprocess(volume_hu, meta["voxel_spacing"])
        self.assertEqual(tensor.ndim, 5)
        
        # Simulated dummy mask
        dummy_pred = (tensor.squeeze(0).squeeze(0) > 0.3).float()
        restored = preprocessor.postprocess(dummy_pred, pre_meta)
        
        # Must strictly match original CT volume dimensions
        self.assertEqual(restored.shape, volume_hu.shape)
        self.assertEqual(restored.dtype, np.uint8)

    def test_03_dicom_seg_generation(self):
        """Test highdicom encapsulation and conformance of generated DICOM-SEG."""
        volume_hu, sorted_datasets, _ = DICOMSeriesReader.load_series_from_directory(
            self.sample_ct_dir
        )
        # Create a simple synthetic mask (e.g., center box)
        mask = np.zeros(volume_hu.shape, dtype=np.uint8)
        mask[4:8, 50:100, 50:100] = 1

        seg_path = DICOMSEGBuilder.create_dicom_seg(
            mask=mask,
            source_datasets=sorted_datasets,
            output_path=self.output_seg_path,
        )
        self.assertTrue(seg_path.exists())

        # Verify output DICOM-SEG header
        seg_ds = pydicom.dcmread(str(seg_path))
        self.assertEqual(seg_ds.SOPClassUID, "1.2.840.10008.5.1.4.1.1.66.4")  # Segmentation Storage
        self.assertEqual(seg_ds.Modality, "SEG")
        self.assertEqual(seg_ds.PatientID, sorted_datasets[0].PatientID)
        self.assertEqual(seg_ds.StudyInstanceUID, sorted_datasets[0].StudyInstanceUID)


if __name__ == "__main__":
    unittest.main()
