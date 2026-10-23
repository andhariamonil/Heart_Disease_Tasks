# Heart Disease Prediction Using Machine Learning

## Task 53: Manual SHAP-Style Explanations

### Objective

Compute exact interventional Shapley values by enumerating feature coalitions and averaging model predictions over an empirical background. Visualize contributions for an individual held-out prediction, verify additivity, and summarize feature-contribution variability across cases.

Exact enumeration scales as $2^p$ for $p$ features. The implementation batches coalition rows, but larger feature spaces require approximation. Independent feature replacement may create implausible combinations and does not establish causality.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_53_Manual_SHAP_Explanations/app.py
```

Run all cells in `Task_53_Manual_SHAP_Explanations/Task_53_Manual_SHAP_Explanations.ipynb` for the standalone notebook.