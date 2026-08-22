
---

## Task 15 — `Task_15_Correlation_Pair_Plots/README.md`

```md
# Heart Disease Prediction Using Machine Learning

## Task 15: Correlation Pair Plots

### Objective

Calculate correlation coefficients manually, dynamically select highly correlated numerical feature pairs, and create scatter plots with manually calculated trendlines.

### Requirements Implemented

- Manual numerical data loading
- Manual mean calculation
- Manual correlation calculation
- Dynamic correlation threshold
- Dynamic selection of correlated feature pairs
- Scatter plots
- Manual trendline calculation
- Raw Matplotlib plotting

### Numerical Features

The following numerical features are analyzed:

- age
- trestbps
- chol
- thalach
- oldpeak
- ca

### Correlation

The Pearson correlation coefficient is calculated manually for every pair of numerical features.

The correlation threshold can be changed in the application.

Only feature pairs whose absolute correlation is greater than or equal to the selected threshold are displayed.

### Trendline

For every selected pair, the slope and intercept of the trendline are calculated manually.

The trendline is then drawn using Matplotlib.

### Built-in Functions Not Used

The following functions are not used:

```python
DataFrame.corr()