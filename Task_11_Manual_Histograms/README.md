
---

## Task 11 — `Task_11_Manual_Histograms/README.md`

```md
# Heart Disease Prediction Using Machine Learning

## Task 11: Manual Histograms

### Objective

Create histograms for all numerical features by manually calculating bins and frequencies without using built-in histogram functions.

### Requirements Implemented

- Manual numerical data extraction
- Manual bin calculation
- Manual frequency calculation
- Histogram for each numerical feature
- Raw Matplotlib visualization
- No built-in histogram function

### Numerical Features

The following numerical columns are visualized:

- age
- trestbps
- chol
- thalach
- oldpeak
- ca

### Implementation

The minimum and maximum values are calculated manually.

The range is divided into a fixed number of bins.

Each value is then assigned to its corresponding bin and the frequency of every bin is calculated manually.

The calculated frequencies are visualized using Matplotlib bar drawing functions.

The following built-in histogram function is not used:

```python
plt.hist()