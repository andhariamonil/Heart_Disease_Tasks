
---

# Task 8 — `Task_8_One_Hot_Encoding/README.md`

```md
# Heart Disease Prediction Using Machine Learning

## Task 8: Manual One-Hot Encoding

### Objective

Perform one-hot encoding manually using a custom encoding function that can dynamically adapt to new unseen categories.

### Requirements Implemented

- Manual category detection
- Manual one-hot encoding
- Dynamic category mapping
- Handling of unseen categories
- Encoded dataset display

### Categorical Columns

The following columns are treated as categorical:

- sex
- cp
- fbs
- restecg
- exang
- slope
- ca
- thal

### Implementation

The program first identifies all unique categories in the selected column.

A separate binary column is then created for every category.

For example:

```text
sex_0
sex_1