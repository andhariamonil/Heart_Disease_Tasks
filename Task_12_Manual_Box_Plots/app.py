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

# Calculate median
def median(values):
    values = values.copy()
    values.sort()

    n = len(values)

    if n % 2 == 0:
        return (values[n // 2 - 1] + values[n // 2]) / 2

    return values[n // 2]

# Calculate quartiles
def quartiles(values):
    values = values.copy()
    values.sort()

    n = len(values)

    med = median(values)

    if n % 2 == 0:
        lower = values[:n // 2]
        upper = values[n // 2:]
    else:
        lower = values[:n // 2]
        upper = values[n // 2 + 1:]

    q1 = median(lower)
    q3 = median(upper)

    return q1, med, q3

columns, data = load_csv("heart.csv")

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

st.title("Heart Disease - Manual Box Plots")

for column in numerical_columns:
    values = []

    for row in data:
        values.append(row[column])

    q1, med, q3 = quartiles(values)

    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = []

    for value in values:
        if value < lower_fence or value > upper_fence:
            outliers.append(value)

    normal_values = []

    for value in values:
        if lower_fence <= value <= upper_fence:
            normal_values.append(value)

    lower_whisker = min(normal_values)
    upper_whisker = max(normal_values)

    fig, ax = plt.subplots()

    ax.plot(
        [q1, q3],
        [1, 1],
        linewidth=12
    )

    ax.plot(
        [med, med],
        [0.85, 1.15],
        linewidth=2
    )

    ax.plot(
        [lower_whisker, q1],
        [1, 1],
        linewidth=2
    )

    ax.plot(
        [q3, upper_whisker],
        [1, 1],
        linewidth=2
    )

    ax.plot(
        [lower_whisker, lower_whisker],
        [0.9, 1.1],
        linewidth=2
    )

    ax.plot(
        [upper_whisker, upper_whisker],
        [0.9, 1.1],
        linewidth=2
    )

    for outlier in outliers:
        ax.plot(outlier, 1, "o")

    ax.set_title("Box Plot of " + column)
    ax.set_xlabel(column)
    ax.set_yticks([])

    st.write("Q1:", q1)
    st.write("Median:", med)
    st.write("Q3:", q3)
    st.write("IQR:", iqr)
    st.write("Lower Fence:", lower_fence)
    st.write("Upper Fence:", upper_fence)
    st.write("Outliers:", outliers)

    st.pyplot(fig)
    plt.close(fig)