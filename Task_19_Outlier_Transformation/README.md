# Heart Disease Prediction Using Machine Learning

## Task 19: Outlier Handling

### Objective

Remove or transform extreme outliers using multiple methods and compare the results.

### Methods Implemented

- Winsorization
- Log transformation
- Manual percentile calculation
- Manual mean calculation
- Manual standard deviation calculation
- Comparative analysis

### Numerical Features

- age
- trestbps
- chol
- thalach
- oldpeak
- ca

### Winsorization

The selected lower and upper percentile values are used as limits.

Values below the lower limit are replaced with the lower limit.

Values above the upper limit are replaced with the upper limit.

### Log Transformation

Positive values are transformed using:

```text
log(x)