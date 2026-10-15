# Heart Disease Prediction Using Machine Learning

## Task 40: Bayesian Neural-Network Architecture Tuning

### Objective

Evaluate candidate neural-network depths and widths with a manual Gaussian-process surrogate and expected-improvement acquisition. Select architectures using validation F1 and evaluate the winner once on a separate held-out test set. The scratch MLP includes the dropout and batch-normalization implementation from Task 39.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_40_Bayesian_Neural_Network_Architecture_Tuning/app.py
```

Run all cells in `Task_40_Bayesian_Neural_Network_Architecture_Tuning/Task_40_Bayesian_Neural_Network_Architecture_Tuning.ipynb` for the standalone notebook.