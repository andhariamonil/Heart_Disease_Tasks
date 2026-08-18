
### Objective

The objective of this task is to identify missing values in the Heart Disease dataset without using .isna() or .dropna() and apply suitable imputation techniques.

---

## Missing Value Detection

Missing values are identified manually by checking whether a value is:

- None
- Empty
- NA
- N/A
- NULL
- ?

This approach avoids the use of built-in missing value handling functions.

---

## Dataset Observation

The provided Heart Disease dataset does not contain missing values.

To demonstrate the imputation process, a demo option was implemented that temporarily creates missing values within selected columns.

The original dataset remains unchanged.

---

## Imputation Techniques Applied

### 1. Mean Imputation

Used for numerical columns.

Formula:

Mean = Sum of Values / Number of Values

Examples:
- age
- trestbps
- chol
- thalach
- oldpeak
- ca

Advantages:
- Simple to implement
- Preserves dataset size
- Suitable when data distribution is approximately normal

---

### 2. Median Imputation

Used for numerical columns.

Formula:

Median = Middle value after sorting

Advantages:
- Less affected by outliers
- More robust than mean for skewed data

---

### 3. Mode Imputation

Used for categorical columns.

Examples:
- sex
- cp
- fbs
- restecg
- exang
- slope
- thal

Formula:

Mode = Most frequently occurring value

Advantages:
- Suitable for categorical attributes
- Maintains the most common category

---

## Method Selection Justification

For numerical columns, Mean and Median imputation techniques were implemented to provide multiple options.

For categorical columns, Mode imputation was selected because replacing missing values with the most frequent category is a common and effective approach.

---

## Result

The implemented system successfully:

1. Detects missing values manually.
2. Demonstrates missing value handling using demo mode.
3. Applies Mean Imputation.
4. Applies Median Imputation.
5. Applies Mode Imputation.
6. Removes all missing values after imputation.

The task requirements were successfully completed.