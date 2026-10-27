# Heart Disease Prediction Using Machine Learning

## Task 61: Bias-Mitigation Strategy Comparison

### Objective

Compare a baseline logistic model with two manually implemented strategies: inverse-frequency reweighting across `sex × target` training cells, and preprocessing that removes the explicit sensitive feature. Evaluate accuracy, precision, recall, F1, disparate impact, demographic-parity gap, and equalized-odds gaps on the same held-out split.

Removing the sensitive field cannot remove correlated proxies. Reweighting changes the fitting objective and can affect utility. This dataset only has one binary group field and is insufficient for a comprehensive fairness audit.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_61_Bias_Mitigation_Strategies/app.py
```

Run all cells in `Task_61_Bias_Mitigation_Strategies/Task_61_Bias_Mitigation_Strategies.ipynb` for the standalone notebook.