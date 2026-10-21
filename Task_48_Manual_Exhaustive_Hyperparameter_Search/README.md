# Heart Disease Prediction Using Machine Learning

## Task 48: Manual Exhaustive Hyperparameter Search

### Objective

Manually enumerate an exhaustive grid of learning rates, L2 strengths, and iteration counts for scratch logistic regression. Select by mean stratified four-fold F1, report fold variability, refit on development data, and evaluate once on a held-out test split.

This task mirrors a GridSearchCV workflow conceptually but intentionally implements loops and scoring directly without `GridSearchCV` or any model-selection helper.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_48_Manual_Exhaustive_Hyperparameter_Search/app.py
```

Run all cells in `Task_48_Manual_Exhaustive_Hyperparameter_Search/Task_48_Manual_Exhaustive_Hyperparameter_Search.ipynb` for the standalone notebook.