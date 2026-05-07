import pandas as pd

from .config import DEFAULT_USD_TO_IDR_RATE
from .currency import add_salary_idr
from .data import SKILL_COLUMNS


def _min_max(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return pd.Series(0.5, index=series.index)
    return (series - minimum) / (maximum - minimum)


def add_ai_enthusiast_features(
    df: pd.DataFrame,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> pd.DataFrame:
    """Add career metrics that are useful for AI enthusiast exploration."""
    result = add_salary_idr(df, exchange_rate)
    available_skill_columns = [column for column in SKILL_COLUMNS if column in result.columns]

    result["Skill_Readiness"] = result[available_skill_columns].mean(axis=1)
    result["Automation_Risk_Score"] = (
        result["Automation_Probability_2030"].clip(0, 1) * 100
    ).round(2)

    salary_score = _min_max(result["Average_Salary"])
    growth_score = _min_max(result["Tech_Growth_Factor"])
    ai_exposure = result["AI_Exposure_Index"].clip(0, 1)
    lower_risk = 1 - result["Automation_Probability_2030"].clip(0, 1)

    result["AI_Opportunity_Score"] = (
        lower_risk * 40
        + ai_exposure * 20
        + growth_score * 20
        + result["Skill_Readiness"].clip(0, 1) * 10
        + salary_score * 10
    ).round(2)

    result["AI_Enthusiast_Segment"] = result.apply(_segment_job, axis=1)
    return result


def _segment_job(row: pd.Series) -> str:
    risk = row["Automation_Probability_2030"]
    opportunity = row["AI_Opportunity_Score"]
    exposure = row["AI_Exposure_Index"]

    if opportunity >= 70 and risk <= 0.45:
        return "AI Builder Track"
    if exposure >= 0.6 and risk <= 0.65:
        return "AI-Augmented Career"
    if risk >= 0.7 and opportunity < 55:
        return "Reskill Priority"
    return "Monitor and Upskill"


def build_summary(
    df: pd.DataFrame,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> dict[str, object]:
    featured = add_ai_enthusiast_features(df, exchange_rate)
    return {
        "rows": len(featured),
        "columns": len(featured.columns),
        "average_salary": round(featured["Average_Salary"].mean(), 2),
        "average_salary_idr": round(featured["Average_Salary_IDR"].mean(), 2),
        "average_automation_probability": round(
            featured["Automation_Probability_2030"].mean(), 3
        ),
        "average_ai_opportunity_score": round(
            featured["AI_Opportunity_Score"].mean(), 2
        ),
        "risk_distribution": featured["Risk_Category"].value_counts().to_dict(),
        "segment_distribution": featured[
            "AI_Enthusiast_Segment"
        ].value_counts().to_dict(),
    }


def top_jobs(
    df: pd.DataFrame,
    sort_by: str = "AI_Opportunity_Score",
    limit: int = 10,
    ascending: bool = False,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> pd.DataFrame:
    featured = add_ai_enthusiast_features(df, exchange_rate)
    columns = [
        "Job_Title",
        "Education_Level",
        "Risk_Category",
        "Average_Salary",
        "Average_Salary_IDR",
        "Years_Experience",
        "AI_Exposure_Index",
        "Automation_Risk_Score",
        "AI_Opportunity_Score",
        "AI_Enthusiast_Segment",
    ]
    return featured.sort_values(sort_by, ascending=ascending).head(limit)[columns]
