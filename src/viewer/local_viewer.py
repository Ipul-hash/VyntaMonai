import argparse
from pathlib import Path
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from src.app.services.dicom_reader import DICOMSeriesReader
from src.app.core.logger import logger


class LocalSliceViewer:
    """
    Interactive Python desktop viewer to inspect 3D CT slices with DICOM-SEG overlays.
    Enables slice scrolling, HU windowing, and overlay transparency toggle.
    """

    def __init__(self, ct_dir: Path, seg_file: Path):
        self.ct_dir = ct_dir
        self.seg_file = seg_file
        
        logger.info(f"Loading CT slices from: {ct_dir}")
        self.volume_hu, self.sorted_datasets, self.meta = (
            DICOMSeriesReader.load_series_from_directory(ct_dir)
        )
        
        logger.info(f"Loading DICOM-SEG from: {seg_file}")
        self.seg_ds = pydicom.dcmread(str(seg_file))
        self.seg_mask = self._extract_seg_mask()

        self.num_slices = self.volume_hu.shape[0]
        self.current_slice = self.num_slices // 2

    def _extract_seg_mask(self) -> np.ndarray:
        """Extract multi-frame binary mask from DICOM-SEG dataset matching sorted slices."""
        raw_pixels = self.seg_ds.pixel_array
        if raw_pixels.ndim == 4:
            raw_pixels = raw_pixels[0]

        # Case 1: All frames present 1:1
        if raw_pixels.shape[0] == self.volume_hu.shape[0]:
            return (raw_pixels > 0).astype(np.uint8)

        # Case 2: Sparse/omitted empty frames mapped via PerFrameFunctionalGroupsSequence
        full_mask = np.zeros(self.volume_hu.shape, dtype=np.uint8)
        if hasattr(self.seg_ds, "PerFrameFunctionalGroupsSequence"):
            sop_to_idx = {ds.SOPInstanceUID: i for i, ds in enumerate(self.sorted_datasets)}
            for f_idx, frame_group in enumerate(self.seg_ds.PerFrameFunctionalGroupsSequence):
                try:
                    ref_sop = frame_group.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
                    if ref_sop in sop_to_idx:
                        slice_idx = sop_to_idx[ref_sop]
                        full_mask[slice_idx] = (raw_pixels[f_idx] > 0).astype(np.uint8)
                except Exception:
                    if f_idx < self.volume_hu.shape[0]:
                        full_mask[f_idx] = (raw_pixels[f_idx] > 0).astype(np.uint8)
            return full_mask

        # Fallback: Pad or truncate
        min_slices = min(raw_pixels.shape[0], self.volume_hu.shape[0])
        full_mask[:min_slices] = (raw_pixels[:min_slices] > 0).astype(np.uint8)
        return full_mask

    def show(self):
        """Render interactive matplotlib visualizer."""
        fig, ax = plt.subplots(figsize=(9, 9))
        plt.subplots_adjust(bottom=0.2)

        # HU Window: Window Width 350, Window Level 40 (Soft Tissue)
        vmin, vmax = -135, 215

        # Display background CT
        ct_slice = self.volume_hu[self.current_slice]
        im_ct = ax.imshow(ct_slice, cmap="gray", vmin=vmin, vmax=vmax)

        # Display overlay mask (red colormap with transparency)
        mask_slice = self.seg_mask[self.current_slice]
        overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        overlay[mask_slice > 0] = [1.0, 0.2, 0.2, 0.45]  # Semi-transparent Red
        im_mask = ax.imshow(overlay)

        ax.axis("off")
        title = ax.set_title(
            f"Slice {self.current_slice + 1}/{self.num_slices} | Z: {self.sorted_datasets[self.current_slice].ImagePositionPatient[2]:.1f} mm\n"
            f"Spleen Voxels: {int(np.sum(mask_slice))}"
        )

        # Slider widget
        ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
        slider = Slider(
            ax_slider,
            "Slice",
            1,
            self.num_slices,
            valinit=self.current_slice + 1,
            valstep=1,
            valfmt="%d",
        )

        def update(val):
            idx = int(slider.val) - 1
            ct_s = self.volume_hu[idx]
            mask_s = self.seg_mask[idx]
            
            im_ct.set_data(ct_s)
            
            new_overlay = np.zeros((*mask_s.shape, 4), dtype=np.float32)
            new_overlay[mask_s > 0] = [1.0, 0.2, 0.2, 0.45]
            im_mask.set_data(new_overlay)

            has_organ = "YES" if np.any(mask_s) else "NO"
            title.set_text(
                f"Slice {idx + 1}/{self.num_slices} | Z: {self.sorted_datasets[idx].ImagePositionPatient[2]:.1f} mm\n"
                f"Spleen Detected: {has_organ} (Voxels: {int(np.sum(mask_s))})"
            )
            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Keyboard navigation
        def on_key(event):
            if event.key in ["up", "right"] and slider.val < self.num_slices:
                slider.set_val(slider.val + 1)
            elif event.key in ["down", "left"] and slider.val > 1:
                slider.set_val(slider.val - 1)

        fig.canvas.mpl_connect("key_press_event", on_key)

        print("\n[Controls] Use Left/Right Arrow Keys or the Slider to navigate slices.")
        plt.show()

    def save_montage_preview(self, output_path: Path, num_thumbnails: int = 6):
        """Export static montage image with segmentation overlay for portfolio."""
        slices_with_organ = [
            i for i in range(self.num_slices) if np.sum(self.seg_mask[i]) > 0
        ]
        if not slices_with_organ:
            slices_with_organ = list(range(0, self.num_slices, max(1, self.num_slices // num_thumbnails)))

        step = max(1, len(slices_with_organ) // num_thumbnails)
        chosen_indices = slices_with_organ[::step][:num_thumbnails]

        cols = 3
        rows = int(np.ceil(len(chosen_indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for i, idx in enumerate(chosen_indices):
            ax = axes[i]
            ax.imshow(self.volume_hu[idx], cmap="gray", vmin=-135, vmax=215)
            mask_s = self.seg_mask[idx]
            overlay = np.zeros((*mask_s.shape, 4), dtype=np.float32)
            overlay[mask_s > 0] = [1.0, 0.2, 0.2, 0.5]
            ax.imshow(overlay)
            ax.set_title(f"Slice {idx + 1} (Z={self.sorted_datasets[idx].ImagePositionPatient[2]:.1f}mm)")
            ax.axis("off")

        for j in range(len(chosen_indices), len(axes)):
            axes[j].axis("off")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info(f"Saved montage preview to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local DICOM-SEG Slice Viewer")
    parser.add_argument("--ct", type=str, required=True, help="Path to CT DICOM directory")
    parser.add_argument("--seg", type=str, required=True, help="Path to DICOM-SEG file")
    parser.add_argument("--save-preview", type=str, default=None, help="Optional path to save montage PNG")
    args = parser.parse_args()

    viewer = LocalSliceViewer(Path(args.ct), Path(args.seg))
    if args.save_preview:
        viewer.save_montage_preview(Path(args.save_preview))
    else:
        viewer.show()
