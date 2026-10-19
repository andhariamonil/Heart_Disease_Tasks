# Heart Disease Prediction Using Machine Learning

## Task 44: Manual Ensemble Voting

### Objective

Combine scratch logistic regression, decision tree, random forest, and gradient boosting with manually calculated hard and soft votes. Base learner weights are normalized validation precision, recall, or F1 selected in the app. Optional class-balanced resampling is available for training.

The project data is deduplicated before splitting. Validation selects weights, and the held-out test split is reserved for reporting.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_44_Manual_Ensemble_Voting/app.py
```

Run all cells in `Task_44_Manual_Ensemble_Voting/Task_44_Manual_Ensemble_Voting.ipynb` for the standalone notebook.