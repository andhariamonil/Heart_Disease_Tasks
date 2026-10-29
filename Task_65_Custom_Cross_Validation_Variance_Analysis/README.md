# Heart Disease Prediction Using Machine Learning

## Task 65: Custom Cross-Validation and Variance Analysis

### Objective

Design and implement custom $k$-fold and stratified split mechanisms from scratch without third-party split abstractions. Ensure balanced class representation across all validation folds, and quantitatively evaluate model performance variance across iterations and varying fold counts ($k \in \{3, 5, 10, 20\}$).

Key components:
1. **Custom Partitioning Logic**: Implements deterministic and pseudo-random index grouping for standard $k$-fold and class-stratified $k$-fold cross-validation.
2. **Balanced Class Representation**: Guarantees identical positive/negative ratios ($\sim 54.3\%$ positive in deduplicated Cleveland data) across all validation partitions, avoiding class-frequency distribution shift that destabilizes unstratified partitions.
3. **Variance Across Iterations**: Formally measures fold-to-fold performance variance ($\sigma^2$), standard error ($\mathrm{SE} = \sigma / \sqrt{k}$), interquartile ranges, and 95% confidence intervals across Accuracy, Precision, Recall, and F1.
4. **Fold Size Sensitivity & Bias-Variance Tradeoff**: Investigates the impact of fold count $k$ on training sample size, pessimistic estimation bias, and estimator correlation across folds.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_65_Custom_Cross_Validation_Variance_Analysis/app.py
```

Run all cells in `Task_65_Custom_Cross_Validation_Variance_Analysis/Task_65_Custom_Cross_Validation_Variance_Analysis.ipynb` for the standalone notebook.

