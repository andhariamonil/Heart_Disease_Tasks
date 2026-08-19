# Heart Disease Prediction Using Machine Learning

## Task 6: Manual Categorical Encoding

### Objective

Convert categorical columns into numerical form without using built-in encoding functions and dynamically handle unknown values.

### Requirements Implemented

- Manual categorical mapping
- Numerical encoding
- Dynamic category detection
- Unknown category handling
- Display of category-to-number mapping

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

Each unique category is manually assigned a numerical value.

For example:

```text
category A → 0
category B → 1
category C → 2