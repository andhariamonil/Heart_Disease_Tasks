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

# Calculate correlation manually
def correlation(x, y):
    mean_x = mean(x)
    mean_y = mean(y)

    numerator = 0
    x_total = 0
    y_total = 0

    for i in range(len(x)):
        x_difference = x[i] - mean_x
        y_difference = y[i] - mean_y

        numerator += x_difference * y_difference
        x_total += x_difference ** 2
        y_total += y_difference ** 2

    denominator = (x_total * y_total) ** 0.5

    if denominator == 0:
        return 0

    return numerator / denominator

# Calculate trendline
def trendline(x, y):
    mean_x = mean(x)
    mean_y = mean(y)

    numerator = 0
    denominator = 0

    for i in range(len(x)):
        numerator += (x[i] - mean_x) * (y[i] - mean_y)
        denominator += (x[i] - mean_x) ** 2

    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator

    intercept = mean_y - slope * mean_x

    return slope, intercept

columns, data = load_csv("heart.csv")

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

correlation_pairs = []

for i in range(len(numerical_columns)):
    for j in range(i + 1, len(numerical_columns)):
        column_x = numerical_columns[i]
        column_y = numerical_columns[j]

        x = []
        y = []

        for row in data:
            x.append(row[column_x])
            y.append(row[column_y])

        value = correlation(x, y)

        correlation_pairs.append(
            (column_x, column_y, value)
        )

st.title("Heart Disease - Correlation Pair Plots")

threshold = st.slider(
    "Correlation threshold",
    0.1,
    1.0,
    0.5
)

selected_pairs = []

for pair in correlation_pairs:
    if abs(pair[2]) >= threshold:
        selected_pairs.append(pair)

st.write("Selected pairs:", len(selected_pairs))

if len(selected_pairs) == 0:
    st.write("No highly correlated pairs found for this threshold.")
else:
    for pair in selected_pairs:
        column_x = pair[0]
        column_y = pair[1]
        correlation_value = pair[2]

        x = []
        y = []

        for row in data:
            x.append(row[column_x])
            y.append(row[column_y])

        slope, intercept = trendline(x, y)

        trend_x = [min(x), max(x)]
        trend_y = []

        for value in trend_x:
            trend_y.append(slope * value + intercept)

        fig, ax = plt.subplots()

        ax.scatter(x, y)
        ax.plot(trend_x, trend_y)

        ax.set_xlabel(column_x)
        ax.set_ylabel(column_y)
        ax.set_title(
            column_x
            + " vs "
            + column_y
            + " (r = "
            + str(round(correlation_value, 3))
            + ")"
        )

        st.pyplot(fig)

        plt.close(fig)