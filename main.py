import argparse

from ai_jobs_enthusiast.analysis import build_summary, top_jobs
from ai_jobs_enthusiast.config import DEFAULT_USD_TO_IDR_RATE
from ai_jobs_enthusiast.currency import format_rupiah
from ai_jobs_enthusiast.data import load_dataset
from ai_jobs_enthusiast.ml_model import (
    compare_full_vs_no_leakage,
    compare_risk_category_models,
)
from ai_jobs_enthusiast.recommender import recommend_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisis dataset AI Impact on Jobs 2030 untuk AI enthusiast."
    )
    parser.add_argument(
        "--data",
        help="Path ke file AI_Impact_on_Jobs_2030.csv. Jika kosong, memakai path default.",
    )
    parser.add_argument(
        "--education",
        help="Filter rekomendasi berdasarkan education level, contoh: Bachelor's.",
    )
    parser.add_argument(
        "--max-risk",
        type=float,
        default=0.65,
        help="Batas maksimum Automation_Probability_2030 untuk rekomendasi.",
    )
    parser.add_argument(
        "--usd-to-idr",
        type=int,
        default=DEFAULT_USD_TO_IDR_RATE,
        help="Kurs konversi USD ke IDR. Default: 16000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data)
    summary = build_summary(df, exchange_rate=args.usd_to_idr)

    print("AI Jobs Enthusiast 2030")
    print("=" * 28)
    print(f"Jumlah data: {summary['rows']} baris")
    print(f"Jumlah kolom setelah fitur tambahan: {summary['columns']}")
    print(
        "Rata-rata gaji: "
        f"{format_rupiah(summary['average_salary_idr'])} "
        f"(${summary['average_salary']:,.2f})"
    )
    print(
        "Rata-rata probabilitas otomatisasi: "
        f"{summary['average_automation_probability']:.3f}"
    )
    print(
        "Rata-rata AI opportunity score: "
        f"{summary['average_ai_opportunity_score']:.2f}"
    )

    print("\nDistribusi risiko:")
    for risk, count in summary["risk_distribution"].items():
        print(f"- {risk}: {count}")

    print("\nSegmentasi AI enthusiast:")
    for segment, count in summary["segment_distribution"].items():
        print(f"- {segment}: {count}")

    print("\nTop 10 pekerjaan dengan AI opportunity tertinggi:")
    print(
        top_jobs(df, limit=10, exchange_rate=args.usd_to_idr).to_string(index=False)
    )

    print("\nRekomendasi karier AI enthusiast:")
    recommendations = recommend_jobs(
        df,
        preferred_education=args.education,
        max_risk=args.max_risk,
        limit=10,
        exchange_rate=args.usd_to_idr,
    )
    print(recommendations.to_string(index=False))

    print("\nMachine Learning Model Comparison")
    comparison = compare_full_vs_no_leakage(df)
    print(comparison.to_string(index=False))

    print("\nBest No-Leakage Model")
    no_leakage_result = compare_risk_category_models(
        df,
        include_automation_probability=False,
    )["best"]
    print(f"Model: {no_leakage_result['model_name']}")
    print(f"Train rows: {no_leakage_result['train_rows']}")
    print(f"Test rows: {no_leakage_result['test_rows']}")
    print(f"Accuracy: {no_leakage_result['accuracy']:.3f}")
    print(f"Macro F1: {no_leakage_result['macro_f1']:.3f}")
    print("\nTop feature importance:")
    print(no_leakage_result["feature_importance"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
