# Heart Disease Prediction Using Machine Learning

## Task 56: Feature Importance Technical Report

### Objective

Rank features using standardized absolute logistic coefficients, repeated validation-set permutation impact on F1, and mean absolute exact Shapley values. Compare global rankings, show a selected case’s local Shapley contributions, and report across-case attribution variability.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_56_Feature_Importance_Technical_Report/app.py
```

Run the notebook for a direct-code implementation. The app also downloads the ranked table as CSV.

See [FEATURE_IMPORTANCE_REPORT.md](FEATURE_IMPORTANCE_REPORT.md) for the measured rankings, split details, interpretation, and technical limitations.