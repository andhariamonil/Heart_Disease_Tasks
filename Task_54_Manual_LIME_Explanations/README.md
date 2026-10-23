# Heart Disease Prediction Using Machine Learning

## Task 54: Manual LIME Explanations

### Objective

Construct synthetic local samples by drawing feature values from observed training rows, weight samples by distance to a selected case, and fit a ridge-regularized linear surrogate manually. Validate explanation fidelity with independently generated weighted perturbations and display local contributions.

The surrogate is a local approximation, not a replacement for the prediction model. Independent feature perturbations can violate real-world feature dependencies, especially for coded categorical variables.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_54_Manual_LIME_Explanations/app.py
```

Run all cells in `Task_54_Manual_LIME_Explanations/Task_54_Manual_LIME_Explanations.ipynb` for the standalone notebook.