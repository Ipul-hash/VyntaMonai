import unittest
from pathlib import Path
import tempfile
import numpy as np

from scripts.prepare_sample_dicom import create_realistic_ct_phantom
from src.app.services.dicom_reader import DICOMSeriesReader
from src.app.services.detection_engine import PulmonaryNoduleDetector, NoduleCandidate
from src.app.services.sr_builder import DICOMSRBuilder


class TestDetectionPipeline(unittest.TestCase):
    """Test suite for Level 2 CADe Pulmonary Nodule Detection and DICOM-SR."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.sample_ct_dir = cls.temp_dir / "sample_ct"
        create_realistic_ct_phantom(cls.sample_ct_dir, num_slices=16)

    def test_01_candidate_creation_and_metrics(self):
        """Test NoduleCandidate data structure and Fleischner categorization."""
        candidate = NoduleCandidate(
            nodule_id=1,
            voxel_box=(100, 100, 5, 115, 115, 8),
            centroid_voxel=(107.5, 107.5, 6.5),
            world_coords_mm=(-25.0, 40.0, -12.0),
            diameter_mm=7.2,
            volume_mm3=195.4,
            mean_hu=-150.0,
            confidence=0.88,
            density_type="Part-Solid (Subsolid)",
            slice_index=6,
            sop_instance_uid="1.2.3.4.5",
        )
        d = candidate.to_dict()
        self.assertEqual(d["nodule_id"], 1)
        self.assertEqual(d["diameter_mm"], 7.2)
        self.assertIn("Intermediate Risk", d["fleischner_category"])

    def test_02_synthetic_nodule_detection(self):
        """Test detection engine on volume with injected synthetic nodule."""
        vol, ds_list, meta = DICOMSeriesReader.load_series_from_directory(self.sample_ct_dir)
        
        # Inject synthetic spherical nodule at slice 8
        z, y, x = 8, 120, 120
        radius = 4  # ~8mm diameter
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask_2d = x_grid**2 + y_grid**2 <= radius**2
        vol[z, y-radius:y+radius+1, x-radius:x+radius+1][mask_2d] = 50.0  # soft tissue HU

        detector = PulmonaryNoduleDetector(min_diameter_mm=2.0, min_confidence=0.30)
        # Create dummy lung mask covering injection point
        lung_mask = np.zeros_like(vol, dtype=bool)
        lung_mask[z, y-10:y+10, x-10:x+10] = True

        candidates = detector.detect_nodules(
            volume_hu=vol,
            sorted_datasets=ds_list,
            spatial_meta=meta,
            lung_mask=lung_mask,
        )
        self.assertGreaterEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].centroid_voxel[0], x, delta=2.0)
        self.assertAlmostEqual(candidates[0].centroid_voxel[1], y, delta=2.0)

    def test_03_dicom_sr_generation(self):
        """Test encapsulation of findings into DICOM-SR file."""
        _, ds_list, _ = DICOMSeriesReader.load_series_from_directory(self.sample_ct_dir)
        candidate = NoduleCandidate(
            nodule_id=1,
            voxel_box=(100, 100, 5, 115, 115, 8),
            centroid_voxel=(107.5, 107.5, 6.5),
            world_coords_mm=(-25.0, 40.0, -12.0),
            diameter_mm=6.5,
            volume_mm3=143.7,
            mean_hu=-50.0,
            confidence=0.92,
            density_type="Solid Nodule",
            slice_index=6,
            sop_instance_uid=ds_list[6].SOPInstanceUID,
        )
        sr_path = self.temp_dir / "test_report.dcm"
        saved_path = DICOMSRBuilder.create_dicom_sr([candidate], ds_list, sr_path)
        self.assertTrue(saved_path.exists())


if __name__ == "__main__":
    unittest.main()
