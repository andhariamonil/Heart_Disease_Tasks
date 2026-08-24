
# Heart Disease Prediction Using Machine Learning

## Task 17: Stacked Bar Chart

### Objective

Create a stacked bar chart comparing sex and target using raw Matplotlib patches with manually calculated proportions.

### Requirements Implemented

- Manual CSV loading
- Manual frequency calculation
- Sex vs target comparison
- Manual proportion calculation
- Stacked bar chart
- Raw Matplotlib patches
- Proper bar alignment
- Percentage annotations

### Variables

Sex:

- 0
- 1

Target:

- 0 = No heart disease
- 1 = Heart disease

### Implementation

The frequency of each sex and target combination is calculated manually.

The target proportions are calculated for each sex category.

The bars are created using Matplotlib Rectangle patches instead of high-level bar plotting functions.

### Run

```bash
streamlit run app.py