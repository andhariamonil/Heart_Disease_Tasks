# Heart Disease Prediction Using Machine Learning

## Task 60: Manual Fairness Metrics

### Objective

Compute group selection rates, disparate impact, demographic-parity gap, and equalized-odds TPR/FPR gaps directly from held-out predictions and confusion counts. The source dataset only provides a binary `sex` field; findings do not cover other demographic groups or intersectional fairness.

Metrics are descriptive and sample-size sensitive. Fairness thresholds depend on context and should not be interpreted as legal certification or clinical approval.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_60_Manual_Fairness_Metrics/app.py
```

Run all cells in `Task_60_Manual_Fairness_Metrics/Task_60_Manual_Fairness_Metrics.ipynb` for the standalone notebook.