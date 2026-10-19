# Heart Disease Prediction Using Machine Learning

## Task 46: Cost-Sensitive Model Selection

### Objective

Balance training classes with a manual inverse-frequency resampling strategy. Compare scratch model families and validation thresholds using a configurable weighted utility over precision, recall, and F1; select the model and threshold on validation data and report once on the held-out test set.

The data is deduplicated before splitting. Resampling and threshold tuning never use test labels. Resampling may trade precision for minority-class recall and must reflect deployment costs and prevalence.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_46_Cost_Sensitive_Model_Selection/app.py
```

Run all cells in `Task_46_Cost_Sensitive_Model_Selection/Task_46_Cost_Sensitive_Model_Selection.ipynb` for the standalone notebook.