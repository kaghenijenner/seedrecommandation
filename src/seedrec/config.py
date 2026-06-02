from pathlib import Path

try:
	from dotenv import load_dotenv
except Exception:
	load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
	load_dotenv(PROJECT_ROOT / ".env")

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_CROP = "maize"
RANDOM_SEED = 42
MIN_RECOMMENDATION_THRESHOLD = float(__import__("os").environ.get("MIN_RECOMMENDATION_THRESHOLD", "0.0"))

