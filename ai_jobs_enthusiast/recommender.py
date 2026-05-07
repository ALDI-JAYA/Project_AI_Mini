import pandas as pd

from .analysis import add_ai_enthusiast_features
from .config import DEFAULT_USD_TO_IDR_RATE
from .data import SKILL_COLUMNS


def _ensure_featured(
    df: pd.DataFrame,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> pd.DataFrame:
    required_features = {
        "Average_Salary_IDR",
        "Automation_Risk_Score",
        "AI_Opportunity_Score",
        "AI_Enthusiast_Segment",
    }
    if required_features.issubset(df.columns):
        return df.copy()
    return add_ai_enthusiast_features(df, exchange_rate)


def recommend_jobs(
    df: pd.DataFrame,
    preferred_education: str | None = None,
    max_risk: float = 0.65,
    limit: int = 10,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> pd.DataFrame:
    """Recommend jobs with strong AI opportunity and acceptable automation risk."""
    featured = _ensure_featured(df, exchange_rate)
    candidates = featured[
        featured["Automation_Probability_2030"].le(max_risk)
    ].copy()

    if preferred_education:
        candidates = candidates[candidates["Education_Level"].eq(preferred_education)]

    columns = [
        "Job_Title",
        "Education_Level",
        "Risk_Category",
        "Average_Salary",
        "Average_Salary_IDR",
        "AI_Exposure_Index",
        "Automation_Risk_Score",
        "AI_Opportunity_Score",
        "AI_Enthusiast_Segment",
    ]
    return (
        candidates.sort_values("AI_Opportunity_Score", ascending=False)
        .head(limit)[columns]
        .reset_index(drop=True)
    )


def recommend_jobs_for_profile(
    df: pd.DataFrame,
    education_level: str | None,
    years_experience: int,
    user_skills: dict[str, float],
    max_risk: float = 0.65,
    minimum_salary: int = 0,
    limit: int = 10,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> pd.DataFrame:
    """Rank jobs using the user's career profile and skill readiness."""
    featured = _ensure_featured(df, exchange_rate)
    candidates = featured[featured["Automation_Probability_2030"].le(max_risk)].copy()

    if education_level and education_level != "Any":
        candidates = candidates[candidates["Education_Level"].eq(education_level)]

    if minimum_salary > 0:
        candidates = candidates[
            candidates["Average_Salary_IDR"].ge(minimum_salary * 0.75)
        ]

    if candidates.empty:
        return pd.DataFrame()

    skill_match_scores = []
    for _, row in candidates.iterrows():
        gaps = []
        for skill in SKILL_COLUMNS:
            if skill in candidates.columns:
                current = float(user_skills.get(skill, 0))
                target = float(row[skill])
                gaps.append(max(target - current, 0))
        average_gap = sum(gaps) / len(gaps) if gaps else 0
        skill_match_scores.append(max(1 - average_gap, 0))

    candidates["Skill_Match_Score"] = skill_match_scores
    candidates["Experience_Match_Score"] = (
        1 - ((candidates["Years_Experience"] - years_experience).abs() / 30)
    ).clip(0, 1)

    if minimum_salary > 0:
        candidates["Salary_Fit_Score"] = (
            candidates["Average_Salary_IDR"] / minimum_salary
        ).clip(0, 1)
    else:
        candidates["Salary_Fit_Score"] = 1

    candidates["Risk_Fit_Score"] = (
        1 - (candidates["Automation_Probability_2030"] / max(max_risk, 0.01))
    ).clip(0, 1)

    candidates["Personal_Fit_Score"] = (
        candidates["AI_Opportunity_Score"] * 0.35
        + candidates["Risk_Fit_Score"] * 100 * 0.20
        + candidates["Skill_Match_Score"] * 100 * 0.25
        + candidates["Experience_Match_Score"] * 100 * 0.10
        + candidates["Salary_Fit_Score"] * 100 * 0.10
    ).round(2)

    columns = [
        "Job_Title",
        "Education_Level",
        "Risk_Category",
        "Average_Salary",
        "Average_Salary_IDR",
        "Years_Experience",
        "Automation_Risk_Score",
        "AI_Opportunity_Score",
        "Skill_Match_Score",
        "Personal_Fit_Score",
        "AI_Enthusiast_Segment",
    ]
    return (
        candidates.sort_values("Personal_Fit_Score", ascending=False)
        .head(limit)[columns]
        .reset_index(drop=True)
    )
