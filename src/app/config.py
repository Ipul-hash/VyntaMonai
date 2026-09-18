from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLES_DIR: Path = DATA_DIR / "samples"
    OUTPUT_SEG_DIR: Path = DATA_DIR / "output_seg"
    OUTPUT_SR_DIR: Path = DATA_DIR / "output_sr"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Model Weights
    SPLEEN_MODEL_PATH: Path = MODELS_DIR / "spleen_ct_segmentation" / "models" / "model.pt"
    
    # CPU Hardware Optimization
    DEVICE: str = "cpu"
    NUM_CPU_THREADS: int = 4  # Conservative default for local laptop responsiveness
    
    # Preprocessing Parameters (Optimized for CPU)
    # Target voxel spacing (mm) in RAS: [x, y, z]
    TARGET_SPACING: tuple[float, float, float] = (1.5, 1.5, 2.0)
    # CT Hounsfield Unit (HU) window matching MSD Task09 / MONAI Model Zoo
    HU_MIN: float = -57.0
    HU_MAX: float = 164.0
    
    # Highdicom Metadata Standards
    ORGAN_NAME: str = "Spleen"
    ORGAN_SCT_CODE: str = "78961009"
    ORGAN_SCT_MEANING: str = "Spleen"
    SERIES_DESCRIPTION: str = "VyntaMonai Spleen AI Segmentation (CPU)"

    # Server & Network Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["*"]

    # Level 2 CADe & Level 3 CADx Configuration
    CAD_MIN_DIAMETER_MM: float = 3.0
    CAD_MIN_CONFIDENCE: float = 0.50


settings = Settings()
