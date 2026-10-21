# Heart Disease Prediction Using Machine Learning

## Task 49: Manual Bayesian Hyperparameter Optimization

### Objective

Use an RBF Gaussian-process regression surrogate and expected improvement, implemented directly with NumPy, to select scratch logistic-regression L2 strength, learning rate, and iteration count. Optimize mean stratified-fold F1, report variability, refit on development data, and evaluate once on held-out test data.

No Bayesian-optimization library is used. Keep the search small for this educational dataset; for reliable model selection use nested cross-validation and a separate external validation population.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_49_Manual_Bayesian_Hyperparameter_Optimization/app.py
```

Run all cells in `Task_49_Manual_Bayesian_Hyperparameter_Optimization/Task_49_Manual_Bayesian_Hyperparameter_Optimization.ipynb` for the standalone notebook.