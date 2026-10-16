# Heart Disease Prediction Using Machine Learning

## Task 43: Manual Model Stacking

### Objective

Tune logistic regression, decision-tree, and random-forest base learners on a dedicated training-only tuning partition. Build meta-training features from four-fold out-of-fold probabilities, fit a manual logistic-regression meta-model, refit base models on the full outer training set, and score once on the untouched test set.

The separated tuning partition, out-of-fold base predictions, and held-back test data are used to reduce leakage. All models and metrics use the project’s scratch implementations.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_43_Manual_Model_Stacking/app.py
```

Run all cells in `Task_43_Manual_Model_Stacking/Task_43_Manual_Model_Stacking.ipynb` for the standalone notebook.