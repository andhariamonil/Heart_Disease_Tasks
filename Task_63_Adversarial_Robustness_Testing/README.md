# Heart Disease Prediction Using Machine Learning

## Task 63: Manual Adversarial Robustness Testing

### Objective

Construct bounded, gradient-directed L-infinity perturbations from the scratch logistic model, compare against random Gaussian noise, and create analytically boundary-directed worst cases. Report performance changes and inspect individual feature shifts.

Perturbations operate in standardized units and are clipped to observed training bounds. This does not make them clinically realistic: correlated fields and integer-coded categories can still become invalid.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_63_Adversarial_Robustness_Testing/app.py
```

Run all cells in `Task_63_Adversarial_Robustness_Testing/Task_63_Adversarial_Robustness_Testing.ipynb` for the standalone notebook.