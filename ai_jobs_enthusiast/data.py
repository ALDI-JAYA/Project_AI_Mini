from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATASET_PATH

REQUIRED_COLUMNS = [
    "Job_Title",
    "Average_Salary",
    "Years_Experience",
    "Education_Level",
    "AI_Exposure_Index",
    "Tech_Growth_Factor",
    "Automation_Probability_2030",
    "Risk_Category",
]

SKILL_COLUMNS = [f"Skill_{index}" for index in range(1, 11)]

NUMERIC_COLUMNS = [
    "Average_Salary",
    "Years_Experience",
    "AI_Exposure_Index",
    "Tech_Growth_Factor",
    "Automation_Probability_2030",
    *SKILL_COLUMNS,
]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the AI jobs dataset."""
    csv_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {csv_path}. "
            "Gunakan argumen --data untuk menentukan lokasi CSV."
        )

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Kolom wajib belum ada di dataset: {joined}")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.drop_duplicates().reset_index(drop=True)
