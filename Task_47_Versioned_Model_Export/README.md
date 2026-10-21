# Heart Disease Prediction Using Machine Learning

## Task 47: Versioned Model Export

### Objective

Fit the project’s scratch logistic model, then export its explicit weights, bias, preprocessing, feature order, threshold, artifact schema/model versions, data fingerprint, and runtime metadata. The bundle is compressed with explicit joblib protocol settings, atomically replaced, reloaded, schema-checked, and prediction-compared.

The artifact is written to `artifacts/heart_logistic_v1.0.0.joblib` when the app runs and can be downloaded from the UI. Deployment requires a compatible `manual_ml.py` and recorded runtime versions. Joblib artifacts can execute code while loading; only load artifacts from trusted sources.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_47_Versioned_Model_Export/app.py
```

Run all cells in `Task_47_Versioned_Model_Export/Task_47_Versioned_Model_Export.ipynb` for the standalone notebook.