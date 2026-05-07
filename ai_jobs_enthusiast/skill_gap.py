import pandas as pd

from .analysis import add_ai_enthusiast_features
from .config import DEFAULT_USD_TO_IDR_RATE
from .data import SKILL_COLUMNS

SKILL_LABELS = {
    "Skill_1": "AI Literacy",
    "Skill_2": "Data Thinking",
    "Skill_3": "Automation Mindset",
    "Skill_4": "Digital Collaboration",
    "Skill_5": "Problem Solving",
    "Skill_6": "Domain Expertise",
    "Skill_7": "Prompting and AI Tools",
    "Skill_8": "Analytical Communication",
    "Skill_9": "Adaptability",
    "Skill_10": "Continuous Learning",
}


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


def get_job_skill_profile(
    df: pd.DataFrame,
    job_title: str,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> pd.Series:
    """Return the average skill profile for one job title."""
    featured = _ensure_featured(df, exchange_rate)
    job_rows = featured[featured["Job_Title"].eq(job_title)]
    if job_rows.empty:
        raise ValueError(f"Pekerjaan '{job_title}' tidak ditemukan.")

    available_skills = [column for column in SKILL_COLUMNS if column in featured.columns]
    return job_rows[available_skills].mean()


def analyze_skill_gap(
    df: pd.DataFrame,
    job_title: str,
    user_skills: dict[str, float],
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> pd.DataFrame:
    """Compare user skills against the target job's average skill profile."""
    target_profile = get_job_skill_profile(df, job_title, exchange_rate)
    rows = []

    for skill in target_profile.index:
        target_value = float(target_profile[skill])
        current_value = float(user_skills.get(skill, 0))
        gap = max(target_value - current_value, 0)
        readiness = 1 - gap

        if gap >= 0.35:
            priority = "High"
        elif gap >= 0.18:
            priority = "Medium"
        else:
            priority = "Low"

        rows.append(
            {
                "Skill": SKILL_LABELS.get(skill, skill),
                "Skill_Code": skill,
                "Current_Level": round(current_value, 2),
                "Target_Level": round(target_value, 2),
                "Gap": round(gap, 2),
                "Readiness": round(max(readiness, 0), 2),
                "Priority": priority,
                "Recommendation": _recommendation_text(skill, gap),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Gap", "Target_Level"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_target_job(
    df: pd.DataFrame,
    job_title: str,
    exchange_rate: int = DEFAULT_USD_TO_IDR_RATE,
) -> dict[str, object]:
    featured = _ensure_featured(df, exchange_rate)
    job_rows = featured[featured["Job_Title"].eq(job_title)]
    if job_rows.empty:
        raise ValueError(f"Pekerjaan '{job_title}' tidak ditemukan.")

    segment = job_rows["AI_Enthusiast_Segment"].mode()
    return {
        "job_title": job_title,
        "records": len(job_rows),
        "average_salary": round(job_rows["Average_Salary"].mean(), 2),
        "average_salary_idr": round(job_rows["Average_Salary_IDR"].mean(), 2),
        "average_risk": round(job_rows["Automation_Risk_Score"].mean(), 2),
        "average_opportunity": round(job_rows["AI_Opportunity_Score"].mean(), 2),
        "common_segment": segment.iloc[0] if not segment.empty else "Unknown",
    }


def _recommendation_text(skill: str, gap: float) -> str:
    label = SKILL_LABELS.get(skill, skill)
    if gap >= 0.35:
        return f"Prioritaskan latihan {label} melalui mini project dan studi kasus."
    if gap >= 0.18:
        return f"Tingkatkan {label} dengan latihan mingguan yang terukur."
    return f"Pertahankan {label}; level saat ini sudah mendekati kebutuhan target."
