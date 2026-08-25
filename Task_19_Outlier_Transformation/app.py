import streamlit as st
import math

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

# Calculate percentile manually
def percentile(values, percentage):
    sorted_values = values.copy()
    sorted_values.sort()

    position = (len(sorted_values) - 1) * percentage
    lower = int(position)
    upper = lower + 1

    if upper >= len(sorted_values):
        return sorted_values[lower]

    fraction = position - lower

    return (
        sorted_values[lower]
        + fraction * (
            sorted_values[upper] -
            sorted_values[lower]
        )
    )

# Calculate mean manually
def mean(values):
    total = 0

    for value in values:
        total += value

    return total / len(values)

# Calculate standard deviation manually
def standard_deviation(values):
    avg = mean(values)
    total = 0

    for value in values:
        total += (value - avg) ** 2

    return (total / len(values)) ** 0.5

# Calculate Winsorized values
def winsorize(values, lower_limit, upper_limit):
    result = []

    for value in values:
        if value < lower_limit:
            result.append(lower_limit)
        elif value > upper_limit:
            result.append(upper_limit)
        else:
            result.append(value)

    return result

# Calculate log transformed values
def log_transform(values):
    result = []

    for value in values:
        if value > 0:
            result.append(math.log(value))
        else:
            result.append(0)

    return result

columns, data = load_csv("heart.csv")

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

st.title("Heart Disease - Outlier Handling")

column = st.selectbox(
    "Select numerical column",
    numerical_columns
)

values = []

for row in data:
    values.append(row[column])

q1 = percentile(values, 0.25)
q3 = percentile(values, 0.75)

lower_limit = q1
upper_limit = q3

winsorized = winsorize(
    values,
    lower_limit,
    upper_limit
)

logged = log_transform(values)

st.write("Original mean:", round(mean(values), 3))
st.write("Original standard deviation:", round(standard_deviation(values), 3))

st.subheader("Winsorization")

st.write("Lower limit:", lower_limit)
st.write("Upper limit:", upper_limit)
st.write("Winsorized mean:", round(mean(winsorized), 3))
st.write(
    "Winsorized standard deviation:",
    round(standard_deviation(winsorized), 3)
)

st.subheader("Log Transformation")

st.write(
    "Log transformed mean:",
    round(mean(logged), 3)
)

st.write(
    "Log transformed standard deviation:",
    round(standard_deviation(logged), 3)
)

st.subheader("Comparison")

if standard_deviation(winsorized) < standard_deviation(logged):
    st.write(
        "Winsorization produces lower variability and is selected."
    )
else:
    st.write(
        "Log transformation produces lower variability and is selected."
    )