from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import scipy.ndimage as ndi
from pydicom.dataset import Dataset

from src.app.core.logger import logger
from src.app.services.cadx_engine import CADxDiagnosticEngine, CADxAssessment


class NoduleCandidate:
    """Represents a detected 3D pulmonary nodule finding with geometric, spatial, and CADx diagnostic metrics."""

    def __init__(
        self,
        nodule_id: int,
        voxel_box: Tuple[int, int, int, int, int, int],  # xmin, ymin, zmin, xmax, ymax, zmax
        centroid_voxel: Tuple[float, float, float],       # xc, yc, zc
        world_coords_mm: Tuple[float, float, float],      # X, Y, Z in Patient Coordinates
        diameter_mm: float,
        volume_mm3: float,
        mean_hu: float,
        confidence: float,
        density_type: str,
        slice_index: int,
        sop_instance_uid: str,
        cadx: Optional[CADxAssessment] = None,
    ):
        self.nodule_id = nodule_id
        self.voxel_box = voxel_box
        self.centroid_voxel = centroid_voxel
        self.world_coords_mm = world_coords_mm
        self.diameter_mm = diameter_mm
        self.volume_mm3 = volume_mm3
        self.mean_hu = mean_hu
        self.confidence = confidence
        self.density_type = density_type
        self.slice_index = slice_index
        self.sop_instance_uid = sop_instance_uid
        self.cadx = cadx

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "nodule_id": self.nodule_id,
            "slice_index": self.slice_index,
            "sop_instance_uid": self.sop_instance_uid,
            "diameter_mm": round(self.diameter_mm, 2),
            "volume_mm3": round(self.volume_mm3, 2),
            "mean_hu": round(self.mean_hu, 1),
            "confidence": round(self.confidence, 3),
            "density_type": self.density_type,
            "voxel_box": [int(v) for v in self.voxel_box],
            "centroid_voxel": [round(v, 1) for v in self.centroid_voxel],
            "world_coords_mm": [round(v, 2) for v in self.world_coords_mm],
            "fleischner_category": self._get_fleischner_category(),
        }
        if self.cadx:
            data.update({
                "lung_rads_category": self.cadx.lung_rads_category,
                "lung_rads_numeric": self.cadx.lung_rads_numeric,
                "lung_rads_suffix": self.cadx.lung_rads_suffix,
                "malignancy_probability": round(self.cadx.malignancy_probability, 3),
                "malignancy_percent": round(self.cadx.malignancy_probability * 100.0, 1),
                "risk_level": self.cadx.risk_level,
                "margin_type": self.cadx.margin_type,
                "spiculation_index": round(self.cadx.spiculation_index, 2),
                "calcification_pattern": self.cadx.calcification_pattern,
                "is_calcified": self.cadx.is_calcified,
                "clinical_recommendation": self.cadx.clinical_recommendation,
                "severity_color": self.cadx.severity_color,
                "cadx": self.cadx.to_dict(),
            })
        return data

    def _get_fleischner_category(self) -> str:
        """Categorize nodule based on Fleischner Society 2017 Guidelines."""
        if self.diameter_mm < 6.0:
            return "Low Risk (<6mm): Routine follow-up optional"
        elif self.diameter_mm <= 8.0:
            return "Intermediate Risk (6-8mm): CT follow-up at 6-12 months"
        else:
            return "High Risk (>8mm): Consider CT at 3 months, PET/CT, or biopsy"


class PulmonaryNoduleDetector:
    """
    Level 2: Computer-Aided Detection (CADe) for 3D Chest CT scans.
    Implements Cascade AI: utilizes Level 1 Lung segmentation to constrain search space,
    extracts 3D nodule candidates, filters airways/vessels, and calculates clinical metrics.
    """

    def __init__(
        self,
        min_diameter_mm: float = 3.0,
        max_diameter_mm: float = 30.0,
        min_confidence: float = 0.50,
    ):
        self.min_diameter_mm = min_diameter_mm
        self.max_diameter_mm = max_diameter_mm
        self.min_confidence = min_confidence

    def detect_nodules(
        self,
        volume_hu: np.ndarray,
        sorted_datasets: List[Dataset],
        spatial_meta: Dict[str, Any],
        lung_mask: Optional[np.ndarray] = None,
    ) -> List[NoduleCandidate]:
        """
        Execute 3D nodule detection on the CT volume.

        Args:
            volume_hu: 3D numpy array (Z, Y, X) in Hounsfield Units
            sorted_datasets: List of pydicom Datasets in Z-order
            spatial_meta: Spatial metadata dictionary
            lung_mask: Optional binary mask (Z, Y, X) from Level 1 segmentation

        Returns:
            List of NoduleCandidate objects
        """
        logger.info("=== Starting Level 2: 3D Pulmonary Nodule Detection (CADe) ===")
        dx, dy, dz = spatial_meta["voxel_spacing"]
        voxel_vol = dx * dy * dz
        num_slices, rows, cols = volume_hu.shape

        # Step 1: Ensure Lung Mask exists (Cascade AI constraint)
        if lung_mask is None:
            logger.info("Generating internal lung search mask for cascade detection...")
            lung_mask = np.zeros_like(volume_hu, dtype=bool)
            for i in range(num_slices):
                s = volume_hu[i]
                if np.any(s > -300):
                    body = ndi.binary_fill_holes(s > -300)
                    lung_roi = (s < -350) & body
                    labeled_l, num_l = ndi.label(lung_roi)
                    if num_l > 0:
                        sizes = ndi.sum(lung_roi, labeled_l, range(num_l + 1))
                        sizes[0] = 0
                        top2 = np.argsort(sizes)[-2:]
                        lung_mask[i] = np.isin(labeled_l, top2)

        # Step 2: Extract nodule candidate regions
        # In CT, pulmonary nodules have soft tissue/ground-glass density [-550 to +150 HU]
        # surrounded by hypodense lung parenchyma [-950 to -600 HU]
        logger.info("Segmenting dense lesions within aerated lung parenchyma...")
        
        # Erode lung boundary slightly to avoid detecting pleural chest wall interfaces
        eroded_lung = ndi.binary_erosion(lung_mask, structure=np.ones((1, 3, 3)))
        
        # Candidate voxels: soft tissue or subsolid density inside the lung
        nodule_voxels = (volume_hu >= -450) & (volume_hu <= 150) & eroded_lung

        # Step 3: 3D Connected Component Analysis
        struct_3d = ndi.generate_binary_structure(3, 2)  # 18-connectivity for 3D
        labeled_nodules, num_features = ndi.label(nodule_voxels, structure=struct_3d)
        logger.info(f"Identified {num_features} initial 3D connected density clusters.")

        if num_features == 0:
            return []

        candidates: List[NoduleCandidate] = []
        objects = ndi.find_objects(labeled_nodules)

        origin_ipp = spatial_meta["origin_ipp"]

        for idx, loc in enumerate(objects):
            if loc is None:
                continue

            z_slice, y_slice, x_slice = loc
            cluster_mask = labeled_nodules[loc] == (idx + 1)
            voxel_count = int(np.sum(cluster_mask))
            
            if voxel_count < 8:  # Filter micro-noise (< 8 voxels)
                continue

            vol_mm3 = voxel_count * voxel_vol

            # Approximate equivalent spherical diameter: d = 2 * (3V / 4pi)^(1/3)
            equiv_diameter_mm = 2.0 * ((3.0 * vol_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0)

            # Check dimensions in physical mm
            dz_cluster = (z_slice.stop - z_slice.start) * dz
            dy_cluster = (y_slice.stop - y_slice.start) * dy
            dx_cluster = (x_slice.stop - x_slice.start) * dx
            max_extent_mm = max(dx_cluster, dy_cluster, dz_cluster)

            if equiv_diameter_mm < self.min_diameter_mm or max_extent_mm > self.max_diameter_mm:
                continue

            # Check Sphericity / Compactness to filter tubular vessels
            # Sphericity = (pi^(1/3) * (6 * V)^(2/3)) / SurfaceArea
            # Elongated vessels have high extent compared to spherical diameter
            elongation = max_extent_mm / (min(dx_cluster, dy_cluster) + 1e-5)
            if elongation > 3.8:  # Vessel rejection heuristic
                continue

            # Calculate mean HU
            cluster_hu = volume_hu[loc][cluster_mask]
            mean_hu = float(np.mean(cluster_hu))

            # Density classification
            if mean_hu > -100:
                density_type = "Solid Nodule"
            elif mean_hu > -350:
                density_type = "Part-Solid (Subsolid)"
            else:
                density_type = "Ground-Glass Nodule (GGN)"

            # Centroid computation
            coords = np.argwhere(cluster_mask)
            zc_rel, yc_rel, xc_rel = np.mean(coords, axis=0)
            zc = z_slice.start + zc_rel
            yc = y_slice.start + yc_rel
            xc = x_slice.start + xc_rel

            # Physical world coordinates in mm
            world_x = origin_ipp[0] + xc * dx
            world_y = origin_ipp[1] + yc * dy
            world_z = origin_ipp[2] + zc * dz

            # Central slice index
            primary_slice = int(round(zc))
            primary_slice = max(0, min(primary_slice, num_slices - 1))
            sop_uid = sorted_datasets[primary_slice].SOPInstanceUID

            # Voxel bounding box: [xmin, ymin, zmin, xmax, ymax, zmax]
            voxel_box = (
                x_slice.start,
                y_slice.start,
                z_slice.start,
                x_slice.stop,
                y_slice.stop,
                z_slice.stop,
            )

            # Confidence score based on circularity, size, and density contrast
            contrast = mean_hu - (-750.0)  # Contrast against aerated lung
            norm_contrast = min(1.0, max(0.0, contrast / 800.0))
            shape_score = max(0.0, 1.0 - (elongation - 1.0) / 2.8)
            confidence = float(0.40 * shape_score + 0.40 * norm_contrast + 0.20 * min(1.0, equiv_diameter_mm / 10.0))

            if confidence >= self.min_confidence:
                candidates.append(
                    NoduleCandidate(
                        nodule_id=len(candidates) + 1,
                        voxel_box=voxel_box,
                        centroid_voxel=(xc, yc, zc),
                        world_coords_mm=(world_x, world_y, world_z),
                        diameter_mm=equiv_diameter_mm,
                        volume_mm3=vol_mm3,
                        mean_hu=mean_hu,
                        confidence=confidence,
                        density_type=density_type,
                        slice_index=primary_slice,
                        sop_instance_uid=sop_uid,
                    )
                )

        # Sort candidates descending by confidence score
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        # Re-index
        for i, c in enumerate(candidates):
            c.nodule_id = i + 1

        # Step 4: Level 3 CADx Diagnostic Evaluation (Malignancy & ACR Lung-RADS v2022)
        cadx_engine = CADxDiagnosticEngine()
        for c in candidates:
            c.cadx = cadx_engine.evaluate_nodule(
                nodule_id=c.nodule_id,
                voxel_box=c.voxel_box,
                centroid_voxel=c.centroid_voxel,
                diameter_mm=c.diameter_mm,
                volume_mm3=c.volume_mm3,
                mean_hu=c.mean_hu,
                density_type=c.density_type,
                volume_hu=volume_hu,
                num_total_slices=num_slices,
            )

        logger.info(f"Level 2 and 3 complete. Evaluated {len(candidates)} nodule candidates with CADx Lung-RADS.")
        return candidates
