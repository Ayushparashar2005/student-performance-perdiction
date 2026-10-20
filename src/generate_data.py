"""Generate a synthetic student-performance dataset for portfolio demonstrations.

No real student records are used. The target is simulated and is not a validated
measure of actual academic risk.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "student_performance_synthetic.csv"


def generate_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    attendance = np.clip(rng.normal(82, 12, n_samples), 35, 100)
    assignment = np.clip(rng.normal(78, 17, n_samples), 0, 100)
    previous_score = np.clip(rng.normal(68, 15, n_samples), 10, 100)
    study_hours = np.clip(rng.gamma(shape=2.5, scale=2.0, size=n_samples), 0, 25)
    platform_days = rng.integers(0, 8, n_samples)
    support_sessions = rng.poisson(lam=1.2, size=n_samples)

    # Illustrative synthetic rule with noise, not a real-world validated label.
    risk_score = (
        0.035 * (75 - attendance)
        + 0.025 * (70 - assignment)
        + 0.045 * (65 - previous_score)
        - 0.055 * study_hours
        - 0.10 * platform_days
        - 0.08 * support_sessions
        + rng.normal(0, 1.0, n_samples)
    )
    threshold = np.quantile(risk_score, 0.65)
    needs_support = (risk_score >= threshold).astype(int)

    df = pd.DataFrame({
        "attendance_pct": np.round(attendance, 1),
        "assignment_completion_pct": np.round(assignment, 1),
        "previous_score": np.round(previous_score, 1),
        "study_hours_weekly": np.round(study_hours, 1),
        "platform_activity_days": platform_days,
        "support_sessions_attended": support_sessions,
        "needs_support": needs_support,
    })

    # Add a small amount of missingness to demonstrate EDA/preprocessing.
    for col in ["attendance_pct", "assignment_completion_pct", "previous_score"]:
        mask = rng.random(n_samples) < 0.025
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Created synthetic dataset: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("Target distribution:")
    print(df["needs_support"].value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    main()
