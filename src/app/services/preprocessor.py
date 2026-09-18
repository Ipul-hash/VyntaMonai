from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from src.app.core.logger import logger
from src.app.config import settings


class MedicalPreprocessor:
    """
    CPU-optimized preprocessing and postprocessing using PyTorch and MONAI transforms.
    Handles HU clipping, normalization, anisotropic spacing resampling, and mask restoration.
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = settings.TARGET_SPACING,
        hu_min: float = settings.HU_MIN,
        hu_max: float = settings.HU_MAX,
    ):
        self.target_spacing = target_spacing  # (X, Y, Z) in mm
        self.hu_min = hu_min
        self.hu_max = hu_max

    def preprocess(
        self,
        volume_hu: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess CT volume:
        1. Clip HU to soft tissue range [hu_min, hu_max]
        2. Normalize to [0.0, 1.0]
        3. Resample to target spacing for fast CPU inference

        Args:
            volume_hu: 3D numpy array of shape (Z, Y, X)
            voxel_spacing: (dx, dy, dz) in mm

        Returns:
            tensor: 5D torch.Tensor of shape (1, 1, Z_resampled, Y_resampled, X_resampled)
            meta: Dictionary containing original shape and scale factors for inversion
        """
        orig_shape = volume_hu.shape  # (Z, Y, X)
        logger.info(f"Original volume shape (Z, Y, X): {orig_shape}")

        # 1. Clip and normalize to [0, 1]
        clipped = np.clip(volume_hu, self.hu_min, self.hu_max)
        normalized = (clipped - self.hu_min) / (self.hu_max - self.hu_min)

        # Convert to PyTorch Tensor: (1, 1, Z, Y, X)
        tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        # Convert DICOM LPS coordinate frame to MONAI canonical RAS (flip Y and X axes)
        tensor = torch.flip(tensor, dims=[3, 4])

        # 2. Compute resampling scale factors based on spacing
        # voxel_spacing is (dx, dy, dz) -> order in tensor is (Z, Y, X)
        orig_spacing_zyx = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])
        target_spacing_zyx = (self.target_spacing[2], self.target_spacing[1], self.target_spacing[0])

        scale_factors = (
            orig_spacing_zyx[0] / target_spacing_zyx[0],
            orig_spacing_zyx[1] / target_spacing_zyx[1],
            orig_spacing_zyx[2] / target_spacing_zyx[2],
        )

        target_depth = max(1, int(round(orig_shape[0] * scale_factors[0])))
        target_height = max(1, int(round(orig_shape[1] * scale_factors[1])))
        target_width = max(1, int(round(orig_shape[2] * scale_factors[2])))
        target_size = (target_depth, target_height, target_width)

        logger.info(
            f"Resampling volume from {orig_shape} to {target_size} for CPU inference efficiency..."
        )

        # Trilinear interpolation for CT intensity volume
        resampled_tensor = F.interpolate(
            tensor,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

        meta = {
            "orig_shape": orig_shape,
            "resampled_shape": target_size,
            "orig_spacing": voxel_spacing,
            "target_spacing": self.target_spacing,
        }

        return resampled_tensor, meta

    def postprocess(
        self,
        pred_mask_tensor: torch.Tensor,
        meta: Dict[str, Any],
    ) -> np.ndarray:
        """
        Restore predicted mask to original DICOM slice resolution and coordinate grid.

        Args:
            pred_mask_tensor: Tensor of shape (1, 1, Z, Y, X) or (Z, Y, X) with binary/probabilities
            meta: Metadata from preprocess step

        Returns:
            binary_mask: 3D uint8 numpy array of shape (Z, Y, X) matching original DICOM series
        """
        orig_shape = meta["orig_shape"]  # (Z, Y, X)

        if pred_mask_tensor.ndim == 3:
            pred_mask_tensor = pred_mask_tensor.unsqueeze(0).unsqueeze(0)
        elif pred_mask_tensor.ndim == 4:
            pred_mask_tensor = pred_mask_tensor.unsqueeze(0)

        # Nearest neighbor interpolation to prevent label bleeding and preserve binary mask
        restored = F.interpolate(
            pred_mask_tensor.float(),
            size=orig_shape,
            mode="nearest",
        )
        # Invert RAS back to DICOM LPS coordinate frame
        restored = torch.flip(restored, dims=[3, 4])

        binary_mask = (restored.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        logger.info(
            f"Restored mask to original DICOM grid: {binary_mask.shape} (Positive voxels: {int(np.sum(binary_mask))})"
        )
        return binary_mask
