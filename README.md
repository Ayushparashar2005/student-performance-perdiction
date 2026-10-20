# Student Academic Performance Prediction

A Python data science project that demonstrates an end-to-end workflow for exploring student academic indicators and building a machine-learning classifier to identify students who may benefit from additional academic support.

> **Important:** This repository is an educational prototype. The included dataset is synthetic, generated locally by the code, and contains no real student records. Model results on synthetic data are not evidence of real-world effectiveness. Predictions must never be used for disciplinary, admissions, or other high-stakes decisions.

## Project goals

- Generate a reproducible sample dataset for development and demonstration.
- Validate and explore data with Pandas.
- Visualize distributions and relationships using Matplotlib and Seaborn.
- Train baseline and candidate classifiers with scikit-learn pipelines.
- Evaluate using precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Save charts, metrics, and a trained model artifact.
- Document privacy, fairness, and responsible-use considerations.

## Repository structure

```text
student-performance-prediction/
├── data/                  # Generated synthetic data (not committed by default)
├── models/                # Trained model artifacts (generated locally)
├── notebooks/
│   └── exploratory_analysis.ipynb  # Optional notebook outline
├── reports/               # Generated charts and evaluation summary
├── src/
│   ├── generate_data.py    # Creates synthetic demonstration data
│   └── train.py            # EDA, training, evaluation, artifacts
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start

Use Python 3.10+.

### 1. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python src/generate_data.py
```

### 4. Run analysis and model evaluation

```bash
python src/train.py
```

The scripts create `data/student_performance_synthetic.csv`, plots in `reports/`, a metrics JSON file, and a model pipeline under `models/`.

## Dataset fields

| Feature | Description |
|---|---|
| `attendance_pct` | Simulated attendance percentage |
| `assignment_completion_pct` | Simulated assignment completion percentage |
| `previous_score` | Simulated prior assessment score out of 100 |
| `study_hours_weekly` | Simulated weekly study hours |
| `platform_activity_days` | Simulated active learning-platform days per week |
| `support_sessions_attended` | Simulated support sessions attended |
| `needs_support` | Synthetic binary target, 1 = meets the generated support-need rule |

The target is generated using a transparent rule with noise so the pipeline can be demonstrated. It is **not** a clinically, educationally, or institutionally validated label.

## Workflow

1. Generate data (or replace it with authorized, de-identified data).
2. Check schema, ranges, missing values, and duplicates.
3. Explore distributions and relationships.
4. Split data into training and test sets.
5. Fit preprocessing inside scikit-learn pipelines to reduce leakage.
6. Compare a DummyClassifier, Logistic Regression, and Random Forest.
7. Report precision, recall, F1, ROC-AUC, and the confusion matrix.
8. Save outputs and discuss limitations before any real-world use.

## Responsible-use principles

- Use only data for which access and processing are authorized.
- Minimize personal information and remove direct identifiers.
- Treat model output as a prompt for human review, never a definitive label.
- Assess subgroup performance and calibration before any pilot.
- Explain uncertainty and provide a way to correct inaccurate records.
- Do not infer causation from feature importance or correlation.
- Do not deploy on real students without institutional approval and a formal privacy/fairness review.

## Limitations

- Synthetic data does not represent the full complexity of real education data.
- Metrics may be artificially high or low because the target is generated from simulated variables.
- A random split may not reflect future-semester performance; real deployments may need time-based or student-grouped validation.
- This project is a portfolio demonstration, not a production-ready decision system.

## Future improvements

- Add a notebook with deeper EDA and written interpretation.
- Add cross-validation and threshold selection.
- Add calibration and subgroup evaluation when appropriate data and approvals exist.
- Add a small Streamlit dashboard for demonstration using synthetic data only.
- Add automated tests and continuous integration.
