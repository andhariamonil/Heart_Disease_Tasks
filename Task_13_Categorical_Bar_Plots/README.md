# Heart Disease Prediction Using Machine Learning

## Task 13: Categorical Bar Plots

### Objective

Display bar plots for categorical variables using low-level Matplotlib plotting commands with dynamically adjusted bar widths and category frequencies.

### Requirements Implemented

- Manual CSV loading
- Manual categorical value counting
- Bar plot generation
- Dynamic bar width
- Frequency display
- Category labels
- Raw Matplotlib plotting
- No Seaborn

### Categorical Columns

The following categorical columns are used:

- sex
- cp
- fbs
- restecg
- exang
- slope
- ca
- thal

### Implementation

The program manually counts the frequency of every category.

The bar width is adjusted according to the number of categories.

The frequency of each category is also displayed above the corresponding bar.

### Visualization

Matplotlib is used directly to create the bar plot.

### Run

```bash
streamlit run app.py