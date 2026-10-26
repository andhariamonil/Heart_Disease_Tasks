# Heart Disease Prediction Using Machine Learning

## Task 58: Error Analysis and Calibration

### Objective

Inspect held-out false positives and false negatives, summarize error counts by age band, fit a manual Platt calibration model on validation probabilities, and select an F1 threshold using validation data only. Compare raw and calibrated test metrics and reliability diagrams.

The test split is not used to fit calibration or select the threshold. Calibration and thresholding change probability interpretation/decision policy; neither guarantees improved performance on future populations.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_58_Error_Analysis_and_Calibration/app.py
```

Run all cells in `Task_58_Error_Analysis_and_Calibration/Task_58_Error_Analysis_and_Calibration.ipynb` for the standalone notebook.