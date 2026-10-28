# Heart Disease Prediction Using Machine Learning

## Task 64: Out-of-Distribution Stress Tests

### Objective

Create synthetic univariate marginal-tail and combined edge cases, then compare predicted confidence, marginal-range violations, and nearest standardized training distance. Synthetic rows have no labels, so this is a confidence/OOD stress test, not an accuracy evaluation.

Marginally plausible values can still form implausible combinations, especially with coded categorical variables. Confidence far from training support should motivate abstention or review, not be treated as reliable prediction.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_64_Out_of_Distribution_Stress_Testing/app.py
```

Run all cells in `Task_64_Out_of_Distribution_Stress_Testing/Task_64_Out_of_Distribution_Stress_Testing.ipynb` for the standalone notebook.