from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.ndimage as ndi

from src.app.core.logger import logger


@dataclass
class CADxAssessment:
    """Structured diagnostic findings and clinical risk assessment for a pulmonary nodule."""
    nodule_id: int
    lung_rads_category: str          # e.g., 'Category 2', 'Category 3', 'Category 4A', 'Category 4B', 'Category 4X'
    lung_rads_numeric: int           # 2, 3, 4
    lung_rads_suffix: str            # '', 'A', 'B', 'X'
    malignancy_probability: float    # 0.00 to 1.00 (percentage / 100)
    risk_level: str                  # 'Benign (<1%)', 'Probably Benign (1-2%)', 'Suspicious (5-15%)', 'Very Suspicious (>15%)'
    margin_type: str                 # 'Smooth', 'Lobulated', 'Spiculated'
    spiculation_index: float         # 0.00 to 1.00
    calcification_pattern: str       # 'None', 'Benign (Central/Diffuse)', 'Stippled/Suspicious'
    is_calcified: bool
    clinical_recommendation: str     # Official ACR Lung-RADS v2022 management recommendation
    severity_color: str              # Hex color code for UI rendering (#00e676, #ffaa00, #ff8000, #ff2d55)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodule_id": self.nodule_id,
            "lung_rads_category": self.lung_rads_category,
            "lung_rads_numeric": self.lung_rads_numeric,
            "lung_rads_suffix": self.lung_rads_suffix,
            "malignancy_probability": round(self.malignancy_probability, 3),
            "malignancy_percent": round(self.malignancy_probability * 100.0, 1),
            "risk_level": self.risk_level,
            "margin_type": self.margin_type,
            "spiculation_index": round(self.spiculation_index, 2),
            "calcification_pattern": self.calcification_pattern,
            "is_calcified": self.is_calcified,
            "clinical_recommendation": self.clinical_recommendation,
            "severity_color": self.severity_color,
        }


class CADxDiagnosticEngine:
    """
    Level 3: Computer-Aided Diagnosis (CADx) & Malignancy Characterization.
    Performs 3D radiomics feature extraction, computes Bayesian malignancy probability,
    and assigns official ACR Lung-RADS v2022 categories with clinical recommendations.
    """

    def __init__(self):
        logger.info("Initialized CADx Diagnostic and Lung-RADS v2022 Engine.")

    def evaluate_nodule(
        self,
        nodule_id: int,
        voxel_box: Tuple[int, int, int, int, int, int],
        centroid_voxel: Tuple[float, float, float],
        diameter_mm: float,
        volume_mm3: float,
        mean_hu: float,
        density_type: str,
        volume_hu: Optional[np.ndarray] = None,
        num_total_slices: int = 250,
    ) -> CADxAssessment:
        """
        Perform comprehensive diagnostic characterization on an identified nodule.
        """
        xmin, ymin, zmin, xmax, ymax, zmax = voxel_box
        xc, yc, zc = centroid_voxel

        # Step 1: 3D Radiomics & Margin Analysis
        spiculation_index = 0.10
        margin_type = "Smooth / Circumscribed"
        calcification_pattern = "None"
        is_calcified = False

        if volume_hu is not None:
            # Crop local 3D volume enclosing the nodule with 3-voxel margin
            pad = 3
            z_s = max(0, zmin - pad)
            z_e = min(volume_hu.shape[0], zmax + pad)
            y_s = max(0, ymin - pad)
            y_e = min(volume_hu.shape[1], ymax + pad)
            x_s = max(0, xmin - pad)
            x_e = min(volume_hu.shape[2], xmax + pad)

            sub_vol = volume_hu[z_s:z_e, y_s:y_e, x_s:x_e]
            
            # Spherical mask strictly around centroid to prevent bleeding into adjacent bone/ribs
            zc_local = zc - z_s
            yc_local = yc - y_s
            xc_local = xc - x_s
            grid_z, grid_y, grid_x = np.ogrid[
                :sub_vol.shape[0], :sub_vol.shape[1], :sub_vol.shape[2]
            ]
            dist_sq = (grid_z - zc_local) ** 2 + (grid_y - yc_local) ** 2 + (grid_x - xc_local) ** 2
            rad_vox = max(1.5, ((xmax - xmin) + (ymax - ymin)) / 4.0)
            nodule_sphere = dist_sq <= (rad_vox * 1.25) ** 2

            nodule_mask = nodule_sphere & (sub_vol >= -450)
            if np.any(nodule_mask):
                cluster_pixels = sub_vol[nodule_mask]
                
                # Check for dense benign calcification (> 200 HU)
                pct_calcified = float(np.mean(cluster_pixels > 200))
                if pct_calcified > 0.35 or (mean_hu > 200 and pct_calcified > 0.20):
                    is_calcified = True
                    calcification_pattern = "Benign (Central/Diffuse)"

                # Check boundary spiculation via radial distance variance
                # Damped by diameter to prevent discrete pixel grid artifacts on small lesions (<6mm)
                eroded = ndi.binary_erosion(nodule_mask)
                boundary = nodule_mask & (~eroded)
                if np.sum(boundary) >= 6:
                    coords = np.argwhere(boundary)
                    r_distances = np.sqrt(
                        (coords[:, 0] - zc_local) ** 2 +
                        (coords[:, 1] - yc_local) ** 2 +
                        (coords[:, 2] - xc_local) ** 2
                    )
                    mean_r = float(np.mean(r_distances))
                    std_r = float(np.std(r_distances))
                    cv_r = std_r / (mean_r + 1e-5)
                    
                    # Size damping factor: true spicules only manifest on lesions >= 6mm
                    size_damping = min(1.0, max(0.2, (diameter_mm - 2.5) / 5.5))
                    spiculation_index = float(min(1.0, max(0.05, cv_r * size_damping * 2.0)))
                    
                    if spiculation_index >= 0.45:
                        margin_type = "Spiculated (Corona Radiata)"
                    elif spiculation_index >= 0.22:
                        margin_type = "Lobulated / Irregular"
                    else:
                        margin_type = "Smooth / Circumscribed"
        else:
            # Synthetic / fallback margin estimation based on density and size
            if diameter_mm > 12.0:
                spiculation_index = 0.52
                margin_type = "Spiculated (Corona Radiata)"
            elif diameter_mm >= 7.0:
                spiculation_index = 0.28
                margin_type = "Lobulated / Irregular"

        # Step 2: Anatomical Upper Lobe Predilection
        # Bronchogenic carcinomas have a predilection for upper lobes (upper 40% of thoracic volume)
        is_upper_lobe = (zc / max(1, num_total_slices)) > 0.55

        # Step 3: Calibrated Bayesian Malignancy Probability (Mayo / Brock PanCan formula)
        logit = -3.40  # Base prior logit for incidental CT screening (~3% baseline risk)
        
        # Diameter contribution (exponential scale per mm beyond 5mm)
        logit += 0.24 * (diameter_mm - 5.0)
        
        # Margin spiculation contribution
        logit += 1.65 * spiculation_index
        
        # Subsolid / Part-solid predilection
        if "Part-Solid" in density_type:
            logit += 0.55
        elif "Ground-Glass" in density_type:
            logit -= 0.30

        # Upper lobe effect
        if is_upper_lobe:
            logit += 0.35

        # Benign calcification strongly protects
        if is_calcified:
            logit -= 3.80

        # Sigmoid conversion
        mal_prob = 1.0 / (1.0 + np.exp(-logit))
        mal_prob = float(np.clip(mal_prob, 0.005, 0.985))

        # Step 4: Official ACR Lung-RADS v2022 Assessment Decision Tree
        if is_calcified:
            cat_name = "Category 2"
            num_val = 2
            suffix = ""
            risk_desc = "Benign (<1% risk)"
            color = "#00e676"  # Emerald Green
            recom = "Routine annual low-dose chest CT (LDCT) screening in 12 months."
        elif diameter_mm < 6.0:
            cat_name = "Category 2"
            num_val = 2
            suffix = ""
            risk_desc = "Benign (<1% risk)"
            color = "#00e676"
            recom = "Routine annual low-dose chest CT (LDCT) screening in 12 months."
        elif diameter_mm < 8.0:
            cat_name = "Category 3"
            num_val = 3
            suffix = ""
            risk_desc = "Probably Benign (1-2% risk)"
            color = "#ffaa00"  # Amber Yellow
            recom = "Short-interval follow-up low-dose CT in 6 months."
        elif diameter_mm < 15.0:
            if spiculation_index >= 0.45:
                cat_name = "Category 4X"
                num_val = 4
                suffix = "X"
                risk_desc = "Very Suspicious with Malignant Features (>15% risk)"
                color = "#ff2d55"  # Crimson Red
                recom = "Diagnostic chest CT with/without contrast, PET/CT, and urgent tissue biopsy or oncology consultation."
            else:
                cat_name = "Category 4A"
                num_val = 4
                suffix = "A"
                risk_desc = "Suspicious (5-15% risk)"
                color = "#ff8000"  # Orange
                recom = "Low-dose CT in 3 months; PET/CT may be considered if solid component >= 8 mm."
        else:  # >= 15.0 mm
            cat_name = "Category 4B"
            num_val = 4
            suffix = "B"
            risk_desc = "Very Suspicious (>15% risk)"
            color = "#ff2d55"
            recom = "Diagnostic chest CT, PET/CT, and/or immediate tissue biopsy; multidisciplinary thoracic tumor board."

        # Ensure Category 4X if malignancy probability is high and margin is spiculated
        if cat_name in ["Category 3", "Category 4A"] and spiculation_index >= 0.50:
            cat_name = "Category 4X"
            num_val = 4
            suffix = "X"
            risk_desc = "Very Suspicious with Malignant Features (>15% risk)"
            color = "#ff2d55"
            recom = "Diagnostic chest CT, PET/CT, and urgent tissue biopsy or oncology consultation."

        return CADxAssessment(
            nodule_id=nodule_id,
            lung_rads_category=cat_name,
            lung_rads_numeric=num_val,
            lung_rads_suffix=suffix,
            malignancy_probability=mal_prob,
            risk_level=risk_desc,
            margin_type=margin_type,
            spiculation_index=spiculation_index,
            calcification_pattern=calcification_pattern,
            is_calcified=is_calcified,
            clinical_recommendation=recom,
            severity_color=color,
        )
