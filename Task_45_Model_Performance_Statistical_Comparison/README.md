# Heart Disease Prediction Using Machine Learning

## Task 45: Model Performance and Statistical Comparison

### Objective

Compare seven scratch model families using common five-fold out-of-fold predictions and manually computed accuracy, precision, recall, F1, fold variability, and train/test gaps. Use a paired, class-stratified bootstrap interval for F1 differences and interpret bias/variance trade-offs.

The confidence intervals are exploratory because the leading model is selected from the same folds. Do not treat them as proof of clinical superiority; confirm selection on a new external cohort.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_45_Model_Performance_Statistical_Comparison/app.py
```

Run all cells in `Task_45_Model_Performance_Statistical_Comparison/Task_45_Model_Performance_Statistical_Comparison.ipynb` for the standalone notebook.