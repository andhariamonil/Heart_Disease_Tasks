# Heart Disease Prediction Using Machine Learning

## Task 62: Performance Stability Across Data Splits

### Objective

Repeat seeded random and stratified train/test splits, fit the same manually implemented logistic model, and compare accuracy, precision, recall, F1, and positive-class composition. Summarize mean, standard deviation, and range and visualize split-to-split variation.

The deduplicated examples recur across splits, so repetitions are sensitivity analysis, not independent samples for significance testing.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_62_Split_Stability_Analysis/app.py
```

Run all cells in `Task_62_Split_Stability_Analysis/Task_62_Split_Stability_Analysis.ipynb` for the standalone notebook.