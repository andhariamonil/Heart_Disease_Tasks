# Heart Disease Prediction Using Machine Learning

## Task 7: Normalization and Standardization

### Objective

Normalize or standardize numerical features using manual calculations instead of built-in functions, while considering outliers and skewed distributions.

### Requirements Implemented

- Manual Min-Max normalization
- Manual Z-Score standardization
- Manual Robust scaling
- Manual mean calculation
- Manual median calculation
- Manual standard deviation calculation
- Manual MAD calculation
- Manual skewness calculation
- Handling of skewed distributions

### Numerical Features

The following numerical columns are used:

- age
- trestbps
- chol
- thalach
- oldpeak
- ca

### Scaling Methods

The application provides three methods:

1. Min-Max Normalization
2. Z-Score Standardization
3. Robust Scaling using Median and MAD

Robust scaling is useful when the data contains outliers because it uses the median and median absolute deviation instead of the mean and standard deviation.

### Implementation

All calculations are performed manually without using built-in preprocessing functions such as:

- MinMaxScaler
- StandardScaler
- RobustScaler

### Run

```bash
streamlit run app.py