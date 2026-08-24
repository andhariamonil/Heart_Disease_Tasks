import streamlit as st
import matplotlib.pyplot as plt

# Load CSV manually
def load_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    columns = lines[0].strip().split(",")
    data = []

    for line in lines[1:]:
        if line.strip() == "":
            continue

        values = line.strip().split(",")
        row = {}

        for i in range(len(columns)):
            try:
                row[columns[i]] = float(values[i])
            except ValueError:
                row[columns[i]] = values[i]

        data.append(row)

    return columns, data

# Calculate mean manually
def mean(values):
    total = 0

    for value in values:
        total += value

    return total / len(values)

# Calculate median manually
def median(values):
    sorted_values = values.copy()
    sorted_values.sort()

    n = len(sorted_values)

    if n % 2 == 0:
        return (
            sorted_values[n // 2 - 1] +
            sorted_values[n // 2]
        ) / 2

    return sorted_values[n // 2]

# Calculate quartiles manually
def quartiles(values):
    sorted_values = values.copy()
    sorted_values.sort()

    n = len(sorted_values)

    if n % 2 == 0:
        lower = sorted_values[:n // 2]
        upper = sorted_values[n // 2:]
    else:
        lower = sorted_values[:n // 2]
        upper = sorted_values[n // 2 + 1:]

    q1 = median(lower)
    q3 = median(upper)

    return q1, q3

# Calculate standard deviation manually
def standard_deviation(values):
    avg = mean(values)
    total = 0

    for value in values:
        total += (value - avg) ** 2

    return (total / len(values)) ** 0.5

columns, data = load_csv("heart.csv")

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

st.title("Heart Disease - Outlier Detection")

column = st.selectbox(
    "Select numerical column",
    numerical_columns
)

values = []

for row in data:
    values.append(row[column])

q1, q3 = quartiles(values)

iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

iqr_outliers = []

for value in values:
    if value < lower_fence or value > upper_fence:
        iqr_outliers.append(value)

avg = mean(values)
std = standard_deviation(values)

z_score_outliers = []

for value in values:
    if std == 0:
        z_score = 0
    else:
        z_score = (value - avg) / std

    if abs(z_score) > 3:
        z_score_outliers.append(value)

st.subheader("IQR Method")

st.write("Q1:", q1)
st.write("Q3:", q3)
st.write("IQR:", iqr)
st.write("Lower Fence:", lower_fence)
st.write("Upper Fence:", upper_fence)
st.write("IQR Outliers:", iqr_outliers)

st.subheader("Z-Score Method")

st.write("Mean:", avg)
st.write("Standard Deviation:", std)
st.write("Z-Score Threshold: ±3")
st.write("Z-Score Outliers:", z_score_outliers)

fig, ax = plt.subplots()

ax.scatter(
    range(len(values)),
    values
)

for i, value in enumerate(values):
    if value in iqr_outliers:
        ax.plot(i, value, "ro")

    elif value in z_score_outliers:
        ax.plot(i, value, "go")

ax.set_xlabel("Row")
ax.set_ylabel(column)
ax.set_title("Outliers in " + column)

st.pyplot(fig)

plt.close(fig)