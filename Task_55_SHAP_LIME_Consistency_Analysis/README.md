# Heart Disease Prediction Using Machine Learning

## Task 55: SHAP and LIME Consistency Analysis

### Objective

Compute exact interventional Shapley values and manually fitted LIME coefficients for the same held-out cases. Quantify per-case Pearson correlation, Spearman correlation of absolute-attribution ranks, sign agreement, top-feature overlap, feature-level variability, and independent LIME weighted fidelity.

Agreement is not proof of correctness. SHAP depends on its background and has exponential cost; LIME depends on its perturbation distribution, kernel width, and surrogate fit. Correlated features and ordinal-coded categorical variables are particularly challenging.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_55_SHAP_LIME_Consistency_Analysis/app.py
```

Run all cells in `Task_55_SHAP_LIME_Consistency_Analysis/Task_55_SHAP_LIME_Consistency_Analysis.ipynb` for the standalone notebook.