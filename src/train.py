"""Run EDA, compare classifiers, and save evaluation artifacts."""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance_synthetic.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
TARGET = "needs_support"

NUMERIC_FEATURES = [
    "attendance_pct",
    "assignment_completion_pct",
    "previous_score",
    "study_hours_weekly",
    "platform_activity_days",
    "support_sessions_attended",
]


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, NUMERIC_FEATURES)],
        remainder="drop",
    )


def make_models() -> dict:
    return {
        "Dummy baseline": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        "Logistic regression": Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ]),
        "Random forest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python src/generate_data.py` first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme()

    df = pd.read_csv(DATA_PATH)
    required = set(NUMERIC_FEATURES + [TARGET])
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    # Basic data-quality summary
    quality = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_by_column": {
            key: int(value) for key, value in df.isna().sum().items()
        },
        "target_distribution": {
            str(key): float(value)
            for key, value in df[TARGET].value_counts(normalize=True).sort_index().items()
        },
    }
    (REPORTS_DIR / "data_quality_summary.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    # EDA chart: feature distributions
    df[NUMERIC_FEATURES].hist(figsize=(12, 8), bins=25, edgecolor="white")
    plt.suptitle("Synthetic Student Data: Feature Distributions")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_distributions.png", dpi=160)
    plt.close()

    # Correlation heatmap (target included for exploratory demonstration)
    plt.figure(figsize=(9, 6))
    sns.heatmap(df[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True),
                annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Feature Correlation Heatmap (Synthetic Data)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    # Split before fitting preprocessing; stratify maintains target proportions.
    X = df[NUMERIC_FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    metrics = {}
    best_name = None
    best_pipeline = None
    best_f1 = -1.0

    for name, pipeline in make_models().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        model_metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
        }

        if hasattr(pipeline, "predict_proba") and len(np.unique(y_test)) == 2:
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            model_metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        else:
            probabilities = None
            model_metrics["roc_auc"] = None

        metrics[name] = model_metrics

        if model_metrics["f1"] > best_f1:
            best_f1 = model_metrics["f1"]
            best_name = name
            best_pipeline = pipeline

    # Save comparison summary
    summary = {
        "note": (
            "Metrics are from a synthetic demonstration dataset and must not be "
            "interpreted as real-world evidence."
        ),
        "test_size": 0.20,
        "random_state": 42,
        "models": metrics,
        "selected_by_test_f1_for_demo_only": best_name,
    }
    (REPORTS_DIR / "model_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Model comparison chart
    comparison = pd.DataFrame([
        {"model": name, **{metric: result[metric]
         for metric in ["precision", "recall", "f1", "roc_auc"]}}
        for name, result in metrics.items()
    ])
    comparison = comparison.melt(id_vars="model", var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Classifier Comparison — Synthetic Demonstration")
    plt.ylabel("Score")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=160)
    plt.close()

    # Confusion matrix for the selected demonstration model
    predictions = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not flagged", "Flagged"],
                yticklabels=["Not support", "Needs support"])
    plt.xlabel("Predicted label")
    plt.ylabel("Synthetic target")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    joblib.dump(best_pipeline, MODELS_DIR / "student_support_pipeline.joblib")
    print("Evaluation complete.")
    print(f"Selected demo model by test F1: {best_name}")
    print(f"Outputs saved to: {REPORTS_DIR}")
    print("Reminder: results are synthetic and are not evidence of real-world performance.")


if __name__ == "__main__":
    main()
