
# Heart Disease Prediction Using Machine Learning

## Task 3: Missing Value Identification and Imputation

### Objective
Identify missing values without using .isna() or .dropna() and apply multiple imputation techniques.

### Requirements Implemented
- Missing value detection
- No use of .isna()
- No use of .dropna()
- Multiple imputation techniques
- Method justification
- Demo missing values option

### Files
- app.py
- heart.csv
- REPORT.md
- requirements.txt

### Run

```bash
streamlit run app.py
```

### Features

#### Missing Value Detection
Missing values are detected manually by checking:

- None
- Empty values
- NA
- N/A
- NULL
- ?

#### Imputation Methods

Numerical Columns:
- Mean Imputation
- Median Imputation

Categorical Columns:
- Mode Imputation

#### Demo Missing Values
The original dataset contains no missing values.

A demo option is provided to temporarily create missing values for testing and demonstrating imputation techniques.

### Dataset
Heart Disease Dataset