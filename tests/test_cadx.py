import unittest
import numpy as np

from src.app.services.cadx_engine import CADxDiagnosticEngine, CADxAssessment
from src.app.services.detection_engine import NoduleCandidate


class TestCADxDiagnosticEngine(unittest.TestCase):
    """Unit test suite for Level 3 CADx Malignancy Prediction and ACR Lung-RADS v2022."""

    def setUp(self):
        self.engine = CADxDiagnosticEngine()

    def test_01_benign_subcentimeter_nodule(self):
        """Test small solid nodule (<6mm) assigned to Lung-RADS Category 2."""
        res = self.engine.evaluate_nodule(
            nodule_id=1,
            voxel_box=(100, 100, 10, 108, 108, 12),
            centroid_voxel=(104.0, 104.0, 11.0),
            diameter_mm=4.5,
            volume_mm3=47.7,
            mean_hu=45.0,
            density_type="Solid Nodule",
        )
        self.assertEqual(res.lung_rads_category, "Category 2")
        self.assertEqual(res.lung_rads_numeric, 2)
        self.assertLess(res.malignancy_probability, 0.05)
        self.assertEqual(res.severity_color, "#00e676")
        self.assertIn("12 months", res.clinical_recommendation)

    def test_02_intermediate_nodule(self):
        """Test 6-8mm solid nodule assigned to Lung-RADS Category 3."""
        res = self.engine.evaluate_nodule(
            nodule_id=2,
            voxel_box=(120, 120, 20, 132, 132, 23),
            centroid_voxel=(126.0, 126.0, 21.5),
            diameter_mm=7.2,
            volume_mm3=195.4,
            mean_hu=60.0,
            density_type="Solid Nodule",
        )
        self.assertEqual(res.lung_rads_category, "Category 3")
        self.assertEqual(res.lung_rads_numeric, 3)
        self.assertEqual(res.severity_color, "#ffaa00")
        self.assertIn("6 months", res.clinical_recommendation)

    def test_03_suspicious_nodule_4a(self):
        """Test 8-15mm smooth solid nodule assigned to Lung-RADS Category 4A."""
        res = self.engine.evaluate_nodule(
            nodule_id=3,
            voxel_box=(150, 150, 30, 168, 168, 34),
            centroid_voxel=(159.0, 159.0, 32.0),
            diameter_mm=10.5,
            volume_mm3=606.1,
            mean_hu=75.0,
            density_type="Solid Nodule",
        )
        self.assertIn(res.lung_rads_category, ["Category 4A", "Category 4X"])
        self.assertEqual(res.lung_rads_numeric, 4)
        self.assertGreater(res.malignancy_probability, 0.05)

    def test_04_very_suspicious_mass_4b(self):
        """Test large mass (>=15mm) assigned to Lung-RADS Category 4B."""
        res = self.engine.evaluate_nodule(
            nodule_id=4,
            voxel_box=(180, 180, 40, 215, 215, 48),
            centroid_voxel=(197.5, 197.5, 44.0),
            diameter_mm=18.5,
            volume_mm3=3315.0,
            mean_hu=85.0,
            density_type="Solid Nodule",
        )
        self.assertIn(res.lung_rads_category, ["Category 4B", "Category 4X"])
        self.assertEqual(res.lung_rads_numeric, 4)
        self.assertGreater(res.malignancy_probability, 0.20)
        self.assertEqual(res.severity_color, "#ff2d55")

    def test_05_calcified_benign_granuloma(self):
        """Test dense calcification (>200 HU) classified as benign Category 2."""
        # Create synthetic volume with calcification
        dummy_vol = np.full((30, 100, 100), -700.0, dtype=np.float32)
        dummy_vol[10:16, 45:55, 45:55] = 350.0  # Dense calcification

        res = self.engine.evaluate_nodule(
            nodule_id=5,
            voxel_box=(45, 45, 10, 55, 55, 16),
            centroid_voxel=(50.0, 50.0, 13.0),
            diameter_mm=6.5,
            volume_mm3=143.0,
            mean_hu=350.0,
            density_type="Solid Nodule",
            volume_hu=dummy_vol,
        )
        self.assertTrue(res.is_calcified)
        self.assertEqual(res.lung_rads_category, "Category 2")
        self.assertLess(res.malignancy_probability, 0.05)

    def test_06_nodule_candidate_to_dict_integration(self):
        """Test NoduleCandidate serialization with attached CADx assessment."""
        cadx = self.engine.evaluate_nodule(
            nodule_id=10,
            voxel_box=(50, 50, 5, 60, 60, 8),
            centroid_voxel=(55.0, 55.0, 6.5),
            diameter_mm=5.2,
            volume_mm3=73.6,
            mean_hu=-20.0,
            density_type="Solid Nodule",
        )
        cand = NoduleCandidate(
            nodule_id=10,
            voxel_box=(50, 50, 5, 60, 60, 8),
            centroid_voxel=(55.0, 55.0, 6.5),
            world_coords_mm=(10.0, 20.0, 30.0),
            diameter_mm=5.2,
            volume_mm3=73.6,
            mean_hu=-20.0,
            confidence=0.85,
            density_type="Solid Nodule",
            slice_index=6,
            sop_instance_uid="1.2.3.4.5",
            cadx=cadx,
        )
        d = cand.to_dict()
        self.assertIn("lung_rads_category", d)
        self.assertIn("malignancy_percent", d)
        self.assertIn("severity_color", d)
        self.assertEqual(d["lung_rads_category"], "Category 2")


if __name__ == "__main__":
    unittest.main()
