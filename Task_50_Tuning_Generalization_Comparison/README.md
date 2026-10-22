# Heart Disease Prediction Using Machine Learning

## Task 50: Before/After Tuning and Generalization

### Objective

Compare baseline and manually tuned logistic regression, decision tree, and random forest models. Tune on stratified cross-validation within development data, report held-out precision/recall/F1, fold variability, and train-validation gaps, and plot development-only logistic convergence curves.

The held-out test set is reserved for final before/after reporting and is not used to select hyperparameters.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_50_Tuning_Generalization_Comparison/app.py
```

Run all cells in `Task_50_Tuning_Generalization_Comparison/Task_50_Tuning_Generalization_Comparison.ipynb` for the standalone notebook.