# Heart Disease Prediction Using Machine Learning

## Task 38: Model Comparison and Decision Boundaries

### Objective

Compare a manually trained linear SVM, KD-tree k-NN, and scratch decision tree using accuracy, precision, recall, and F1. Plot their two-feature decision regions in the same standardized age/maximum-heart-rate space and discuss interpretability, scaling, and prediction-cost trade-offs. Duplicate records are removed before splitting.

The two-feature boundary view is deliberately educational and is not a clinical prediction tool.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_38_Model_Comparison_Decision_Boundaries/app.py
```

Run all cells in `Task_38_Model_Comparison_Decision_Boundaries/Task_38_Model_Comparison_Decision_Boundaries.ipynb` for the notebook version.