
---

# Task 9 — `Task_9_Feature_Engineering/README.md`

```md
# Heart Disease Prediction Using Machine Learning

## Task 9: Feature Engineering

### Objective

Create a new feature called `risk_factor` using the `chol` and `thalach` features with domain-based components, interaction terms, and percentile-based ranking.

### Requirements Implemented

- Uses `chol`
- Uses `thalach`
- Cholesterol risk component
- Heart-rate component
- Interaction term
- Percentile-based ranking
- New `risk_factor` feature

### Feature Components

The `risk_factor` is calculated using:

- Cholesterol-based risk
- Heart-rate-based risk
- Interaction between cholesterol and heart rate
- Cholesterol percentile
- Maximum-heart-rate percentile

### Percentile Ranking

The percentile of each `chol` and `thalach` value is calculated manually by comparing it with all values in the corresponding column.

### Interaction Term

An interaction term is created by multiplying the cholesterol-related component and heart-rate-related component.

### Result

The application adds the following new features:

- chol_percentile
- thalach_percentile
- risk_factor

### Run

```bash
streamlit run app.py