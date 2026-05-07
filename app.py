from pathlib import Path

import plotly.express as px
import streamlit as st

from ai_jobs_enthusiast.analysis import add_ai_enthusiast_features
from ai_jobs_enthusiast.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_USD_TO_IDR_RATE,
    RISK_ORDER,
)
from ai_jobs_enthusiast.currency import format_rupiah
from ai_jobs_enthusiast.data import SKILL_COLUMNS, load_dataset
from ai_jobs_enthusiast.ml_model import compare_risk_category_models
from ai_jobs_enthusiast.recommender import recommend_jobs, recommend_jobs_for_profile
from ai_jobs_enthusiast.skill_gap import (
    SKILL_LABELS,
    analyze_skill_gap,
    summarize_target_job,
)

st.set_page_config(
    page_title="AI Jobs Enthusiast 2030",
    page_icon="AI",
    layout="wide",
)


@st.cache_data
def load_featured_data(csv_path: str, exchange_rate: int):
    df = load_dataset(Path(csv_path))
    return add_ai_enthusiast_features(df, exchange_rate)


@st.cache_data
def compare_cached_models(csv_path: str, include_automation_probability: bool):
    df = load_dataset(Path(csv_path))
    return compare_risk_category_models(
        df,
        include_automation_probability=include_automation_probability,
    )


st.title("AI Jobs Enthusiast 2030")
st.caption("Eksplorasi risiko otomatisasi, peluang AI, dan arah karier berbasis data.")

with st.sidebar:
    st.header("Data dan Filter")
    dataset_path = st.text_input("Path dataset", value=str(DEFAULT_DATASET_PATH))
    exchange_rate = st.number_input(
        "Kurs USD ke IDR",
        min_value=1000,
        max_value=50000,
        value=DEFAULT_USD_TO_IDR_RATE,
        step=500,
    )

    try:
        data = load_featured_data(dataset_path, exchange_rate)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    risk_options = [risk for risk in RISK_ORDER if risk in data["Risk_Category"].unique()]
    selected_risks = st.multiselect(
        "Risk category",
        options=risk_options,
        default=risk_options,
    )

    education_options = sorted(data["Education_Level"].dropna().unique())
    selected_educations = st.multiselect(
        "Education level",
        options=education_options,
        default=education_options,
    )

    max_risk = st.slider(
        "Batas risiko rekomendasi",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )

filtered = data[
    data["Risk_Category"].isin(selected_risks)
    & data["Education_Level"].isin(selected_educations)
].copy()

overview_tab, recommender_tab, skill_gap_tab, ml_tab = st.tabs(
    [
        "Overview",
        "AI Career Recommender",
        "Skill Gap Analyzer",
        "ML Model",
    ]
)

with overview_tab:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total pekerjaan", f"{len(filtered):,}")
    metric_columns[1].metric(
        "Rata-rata opportunity",
        f"{filtered['AI_Opportunity_Score'].mean():.2f}",
    )
    metric_columns[2].metric(
        "Rata-rata risiko",
        f"{filtered['Automation_Risk_Score'].mean():.2f}",
    )
    metric_columns[3].metric(
        "Rata-rata gaji",
        format_rupiah(filtered["Average_Salary_IDR"].mean()),
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Peta Risiko vs Peluang AI")
        scatter = px.scatter(
            filtered,
            x="Automation_Risk_Score",
            y="AI_Opportunity_Score",
            color="Risk_Category",
            size="Average_Salary",
            hover_name="Job_Title",
            hover_data=[
                "Education_Level",
                "Years_Experience",
                "AI_Exposure_Index",
                "Tech_Growth_Factor",
                "AI_Enthusiast_Segment",
            ],
            category_orders={"Risk_Category": RISK_ORDER},
            color_discrete_map={
                "Low": "#2a9d8f",
                "Medium": "#e9c46a",
                "High": "#e76f51",
            },
        )
        scatter.update_layout(height=520)
        st.plotly_chart(scatter, width="stretch")

    with right:
        st.subheader("Distribusi Segmentasi")
        segment_counts = (
            filtered["AI_Enthusiast_Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Count")
        )
        bar = px.bar(
            segment_counts,
            x="Count",
            y="Segment",
            orientation="h",
            color="Segment",
            color_discrete_sequence=["#264653", "#2a9d8f", "#f4a261", "#e76f51"],
        )
        bar.update_layout(showlegend=False, height=520)
        st.plotly_chart(bar, width="stretch")

    st.subheader("Top Pekerjaan untuk AI Enthusiast")
    top_table = (
        filtered.sort_values("AI_Opportunity_Score", ascending=False)
        .head(15)[
            [
                "Job_Title",
                "Education_Level",
                "Risk_Category",
                "Average_Salary_IDR",
                "AI_Exposure_Index",
                "Automation_Risk_Score",
                "AI_Opportunity_Score",
                "AI_Enthusiast_Segment",
            ]
        ]
        .reset_index(drop=True)
    )
    st.dataframe(top_table, width="stretch")

    st.subheader("Rekomendasi Karier Umum")
    education_filter = (
        selected_educations[0] if len(selected_educations) == 1 else None
    )
    recommendation_table = recommend_jobs(
        filtered,
        preferred_education=education_filter,
        max_risk=max_risk,
        limit=12,
        exchange_rate=exchange_rate,
    )
    st.dataframe(recommendation_table, width="stretch")

with recommender_tab:
    st.subheader("Profil Karier")
    profile_left, profile_right = st.columns(2)

    with profile_left:
        profile_education = st.selectbox(
            "Pendidikan saat ini",
            options=["Any", *education_options],
            index=0,
        )
        profile_experience = st.slider(
            "Pengalaman kerja",
            min_value=0,
            max_value=30,
            value=5,
        )
        profile_minimum_salary = st.number_input(
        "Target minimum gaji",
        min_value=0,
        max_value=3000000000,
        value=1000000000,
        step=50000000,
    )

    with profile_right:
        profile_max_risk = st.slider(
            "Toleransi risiko otomatisasi",
            min_value=0.05,
            max_value=1.0,
            value=max_risk,
            step=0.05,
        )
        st.write("Level skill saat ini")
        user_skills = {}
        skill_columns = st.columns(2)
        for index, skill in enumerate(SKILL_COLUMNS):
            with skill_columns[index % 2]:
                user_skills[skill] = st.slider(
                    SKILL_LABELS.get(skill, skill),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key=f"profile_{skill}",
                )

    personalized = recommend_jobs_for_profile(
        filtered,
        education_level=profile_education,
        years_experience=profile_experience,
        user_skills=user_skills,
        max_risk=profile_max_risk,
        minimum_salary=profile_minimum_salary,
        limit=15,
        exchange_rate=exchange_rate,
    )

    st.subheader("Rekomendasi Personal")
    if personalized.empty:
        st.warning("Belum ada pekerjaan yang cocok dengan filter profil ini.")
    else:
        display_personalized = personalized.copy()
        display_personalized["Skill_Match_Score"] = (
            display_personalized["Skill_Match_Score"] * 100
        ).round(2)
        st.dataframe(display_personalized, width="stretch")

        fit_chart = px.bar(
            display_personalized.head(10),
            x="Personal_Fit_Score",
            y="Job_Title",
            color="AI_Enthusiast_Segment",
            orientation="h",
            color_discrete_sequence=["#2a9d8f", "#264653", "#f4a261", "#e76f51"],
        )
        fit_chart.update_layout(height=460, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fit_chart, width="stretch")

with skill_gap_tab:
    st.subheader("Target Pekerjaan")
    job_options = sorted(filtered["Job_Title"].dropna().unique())
    target_job = st.selectbox("Pilih pekerjaan target", options=job_options)

    target_summary = summarize_target_job(filtered, target_job, exchange_rate)
    target_metrics = st.columns(4)
    target_metrics[0].metric("Jumlah sampel", f"{target_summary['records']}")
    target_metrics[1].metric(
        "Rata-rata opportunity",
        f"{target_summary['average_opportunity']:.2f}",
    )
    target_metrics[2].metric(
        "Rata-rata risiko",
        f"{target_summary['average_risk']:.2f}",
    )
    target_metrics[3].metric(
        "Rata-rata gaji",
        format_rupiah(target_summary["average_salary_idr"]),
    )

    st.caption(f"Segmentasi dominan: {target_summary['common_segment']}")

    st.subheader("Skill Saat Ini")
    gap_user_skills = {}
    gap_skill_columns = st.columns(2)
    for index, skill in enumerate(SKILL_COLUMNS):
        with gap_skill_columns[index % 2]:
            gap_user_skills[skill] = st.slider(
                SKILL_LABELS.get(skill, skill),
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"gap_{skill}",
            )

    gap_table = analyze_skill_gap(
        filtered,
        target_job,
        gap_user_skills,
        exchange_rate,
    )
    st.subheader("Prioritas Pengembangan Skill")
    st.dataframe(gap_table, width="stretch")

    gap_chart = px.bar(
        gap_table,
        x="Gap",
        y="Skill",
        color="Priority",
        orientation="h",
        color_discrete_map={
            "High": "#e76f51",
            "Medium": "#e9c46a",
            "Low": "#2a9d8f",
        },
    )
    gap_chart.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(gap_chart, width="stretch")

with ml_tab:
    st.subheader("Risk Category Prediction")
    ml_mode = st.radio(
        "Mode evaluasi",
        options=["No-Leakage", "Full Features"],
        horizontal=True,
    )
    include_automation_probability = ml_mode == "Full Features"
    comparison_result = compare_cached_models(
        dataset_path,
        include_automation_probability,
    )
    model_result = comparison_result["best"]

    st.subheader("Model Comparison")
    st.dataframe(comparison_result["comparison"], width="stretch")
    comparison_chart = px.bar(
        comparison_result["comparison"],
        x="Model",
        y="Macro_F1",
        color="Accuracy",
        color_continuous_scale="Viridis",
    )
    comparison_chart.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(comparison_chart, width="stretch")

    model_metrics = st.columns(4)
    model_metrics[0].metric("Best Model", model_result["model_name"])
    model_metrics[1].metric("Accuracy", f"{model_result['accuracy']:.3f}")
    model_metrics[2].metric("Macro F1", f"{model_result['macro_f1']:.3f}")
    model_metrics[3].metric(
        "Training time",
        f"{model_result['training_time_seconds']:.3f}s",
    )

    row_metrics = st.columns(2)
    row_metrics[0].metric("Train rows", f"{model_result['train_rows']:,}")
    row_metrics[1].metric("Test rows", f"{model_result['test_rows']:,}")

    confusion_chart = px.imshow(
        model_result["confusion_matrix"],
        text_auto=True,
        color_continuous_scale="Tealrose",
        aspect="auto",
    )
    confusion_chart.update_layout(height=420)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion Matrix")
        st.plotly_chart(confusion_chart, width="stretch")

    with right:
        st.subheader("Classification Report")
        st.dataframe(model_result["report"], width="stretch")

    st.subheader("Top Feature Importance")
    importance_chart = px.bar(
        model_result["feature_importance"],
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
    )
    importance_chart.update_layout(
        height=520,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
    )
    st.plotly_chart(importance_chart, width="stretch")

    st.caption(
        "No-Leakage menghapus Automation_Probability_2030 dari training agar "
        "evaluasi tidak terlalu bergantung pada fitur yang sangat dekat dengan label."
    )
