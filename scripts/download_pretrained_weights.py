from pathlib import Path
import urllib.request
from src.app.core.logger import logger
from src.app.config import settings

# Pretrained model URL from MONAI Model Zoo / Medical Segmentation Decathlon Task09
MONAI_SPLEEN_WEIGHTS_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/spleen_model.pt"
)


def download_weights(target_path: Path = settings.SPLEEN_MODEL_PATH):
    """Download official MONAI Spleen 3D UNet checkpoint."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        logger.info(f"Model checkpoint already exists at: {target_path}")
        return

    logger.info(f"Downloading pretrained weights from {MONAI_SPLEEN_WEIGHTS_URL}...")
    try:
        urllib.request.urlretrieve(MONAI_SPLEEN_WEIGHTS_URL, str(target_path))
        logger.info(f"Downloaded weights successfully to: {target_path}")
    except Exception as e:
        logger.warning(
            f"Could not download weights automatically ({e}). Pipeline will run with standard weights."
        )


if __name__ == "__main__":
    download_weights()
