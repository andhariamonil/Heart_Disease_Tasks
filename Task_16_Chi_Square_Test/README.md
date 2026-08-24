# Heart Disease Prediction Using Machine Learning

## Task 16: Chi-Square Test

### Objective

Perform a chi-square test between categorical columns and the target by manually calculating observed frequencies, expected frequencies, and the chi-square statistic.

### Requirements Implemented

- Manual observed frequency calculation
- Manual expected frequency calculation
- Manual chi-square calculation
- Manual degrees of freedom calculation
- Chi-square decision using a critical value
- No statistical libraries

### Categorical Columns

The following columns can be tested:

- sex
- cp
- fbs
- restecg
- exang
- slope
- ca
- thal

The target column is:

- target

### Formula

Expected frequency:

```text
Expected = (Row Total × Column Total) / Grand Total
χ² = Σ((Observed - Expected)² / Expected)