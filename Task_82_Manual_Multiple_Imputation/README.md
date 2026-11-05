# Heart Disease Prediction Using Machine Learning

## Task 82: Manual Multiple Imputation Comparison

### Objective

Compare manual KNN imputation, iterative chained conditional prediction (MICE-like), and one-pass predictive imputation. Select a method per column using validation error: IQR-normalized MAE for continuous fields and misclassification rate for coded categorical fields.

The supplied CSV has no missing feature values in prior checks, so this task creates a seeded 12% artificial missingness mask, evaluates only hidden original values, and leaves the source dataset unchanged. The MICE-like implementation is an educational approximation, not a substitute for a validated clinical imputation protocol.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_82_Manual_Multiple_Imputation/app.py
```

Run all cells in `Task_82_Manual_Multiple_Imputation/Task_82_Manual_Multiple_Imputation.ipynb` for the standalone notebook.