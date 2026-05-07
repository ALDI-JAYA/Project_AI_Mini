from time import perf_counter

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RISK_ORDER
from .data import SKILL_COLUMNS

CATEGORICAL_FEATURES = ["Job_Title", "Education_Level"]
NUMERIC_FEATURES = [
    "Average_Salary",
    "Years_Experience",
    "AI_Exposure_Index",
    "Tech_Growth_Factor",
    "Automation_Probability_2030",
    *SKILL_COLUMNS,
]
TARGET_COLUMN = "Risk_Category"


def train_risk_category_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    include_automation_probability: bool = True,
    model_name: str = "Random Forest",
) -> dict[str, object]:
    """Train and evaluate a model that predicts job automation risk category."""
    numeric_features = _numeric_features(include_automation_probability)
    feature_columns = CATEGORICAL_FEATURES + numeric_features
    model_data = df[[*feature_columns, TARGET_COLUMN]].dropna().copy()

    x = model_data[feature_columns]
    y = model_data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = _build_preprocessor(categorical_pipeline, numeric_pipeline, numeric_features)
    classifier = _build_classifier(model_name, random_state)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    started_at = perf_counter()
    pipeline.fit(x_train, y_train)
    training_time = perf_counter() - started_at
    predictions = pipeline.predict(x_test)
    labels = [label for label in RISK_ORDER if label in y.unique()]

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Class"})
        .round(3)
    )

    confusion_df = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    )

    return {
        "model": pipeline,
        "model_name": model_name,
        "include_automation_probability": include_automation_probability,
        "accuracy": round(accuracy_score(y_test, predictions), 3),
        "macro_f1": round(f1_score(y_test, predictions, average="macro"), 3),
        "report": report_df,
        "confusion_matrix": confusion_df,
        "feature_importance": _feature_importance(pipeline),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "features": feature_columns,
        "training_time_seconds": round(training_time, 3),
    }


def compare_risk_category_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    include_automation_probability: bool = True,
) -> dict[str, object]:
    """Compare several classifiers for portfolio-ready model evaluation."""
    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    ]
    results = [
        train_risk_category_model(
            df,
            test_size=test_size,
            random_state=random_state,
            include_automation_probability=include_automation_probability,
            model_name=model_name,
        )
        for model_name in model_names
    ]

    comparison = pd.DataFrame(
        [
            {
                "Model": result["model_name"],
                "Accuracy": result["accuracy"],
                "Macro_F1": result["macro_f1"],
                "Training_Time_Seconds": result["training_time_seconds"],
                "Mode": (
                    "Full Features"
                    if include_automation_probability
                    else "No-Leakage"
                ),
            }
            for result in results
        ]
    ).sort_values(["Macro_F1", "Accuracy"], ascending=False)

    best_model_name = comparison.iloc[0]["Model"]
    best_result = next(
        result for result in results if result["model_name"] == best_model_name
    )
    return {
        "comparison": comparison.reset_index(drop=True),
        "best": best_result,
        "results": results,
    }


def compare_full_vs_no_leakage(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare all models in full-feature and no-leakage modes."""
    full = compare_risk_category_models(
        df,
        test_size=test_size,
        random_state=random_state,
        include_automation_probability=True,
    )["comparison"]
    no_leakage = compare_risk_category_models(
        df,
        test_size=test_size,
        random_state=random_state,
        include_automation_probability=False,
    )["comparison"]
    return pd.concat([full, no_leakage], ignore_index=True).sort_values(
        ["Mode", "Macro_F1", "Accuracy"],
        ascending=[True, False, False],
    )


def _numeric_features(include_automation_probability: bool) -> list[str]:
    if include_automation_probability:
        return NUMERIC_FEATURES
    return [
        feature
        for feature in NUMERIC_FEATURES
        if feature != "Automation_Probability_2030"
    ]


def _build_preprocessor(
    categorical_pipeline: Pipeline,
    numeric_pipeline: Pipeline,
    numeric_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )


def _build_classifier(model_name: str, random_state: int):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=4,
            random_state=random_state,
            class_weight="balanced",
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=3,
            random_state=random_state,
        )
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=4,
        random_state=random_state,
        class_weight="balanced",
    )


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = (
        pd.Series(feature_names)
        .str.replace("categorical__", "", regex=False)
        .str.replace("numeric__", "", regex=False)
    )

    if hasattr(classifier, "feature_importances_"):
        scores = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        scores = abs(classifier.coef_).mean(axis=0)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    importance = pd.DataFrame(
        {
            "Feature": clean_feature_names,
            "Importance": scores,
        }
    )
    return (
        importance.sort_values("Importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
        .round({"Importance": 4})
    )
