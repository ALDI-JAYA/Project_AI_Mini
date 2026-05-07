from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DATASET_PATH = PROJECT_ROOT / "data" / "AI_Impact_on_Jobs_2030.csv"
DOWNLOADS_DATASET_PATH = (
    Path.home() / "Downloads" / "archive" / "AI_Impact_on_Jobs_2030.csv"
)

DEFAULT_DATASET_PATH = (
    REPO_DATASET_PATH if REPO_DATASET_PATH.exists() else DOWNLOADS_DATASET_PATH
)

RISK_ORDER = ["Low", "Medium", "High"]

DEFAULT_USD_TO_IDR_RATE = 16000
