"""Project-wide constants and paths.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
METADATA_PATH = DATA_DIR / "SoilClassificationTargets.csv"
IMAGE_DIR = DATA_DIR / "images"

OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
MODEL_DIR = OUTPUT_DIR / "models"

IMAGE_SIZE = 224
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15
N_SPLITS = 5
IMAGES_PER_CODE = 5

REQUIRED_COLUMNS = [
    "Code",
    "Material",
    "Density",
    "Percentage",
    "Weight",
    "Total weight",
]


def ensure_output_dirs():
    dirs = [OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, PREDICTION_DIR, MODEL_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs
