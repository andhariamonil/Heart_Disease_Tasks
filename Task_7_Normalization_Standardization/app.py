import streamlit as st

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

# Calculate mean
def mean(values):
    total = 0

    for value in values:
        total += value

    return total / len(values)

# Calculate standard deviation
def standard_deviation(values):
    avg = mean(values)
    total = 0

    for value in values:
        total += (value - avg) ** 2

    return (total / len(values)) ** 0.5

# Calculate skewness
def skewness(values):
    avg = mean(values)
    std = standard_deviation(values)

    if std == 0:
        return 0

    total = 0

    for value in values:
        total += ((value - avg) / std) ** 3

    return total / len(values)

# Calculate MAD
def mad(values):
    med = median(values)
    differences = []

    for value in values:
        differences.append(abs(value - med))

    return median(differences)

# Load dataset
columns, data = load_csv("heart.csv")

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

st.title("Heart Disease - Feature Scaling")

column = st.selectbox("Select numerical column", numerical_columns)
method = st.selectbox(
    "Scaling method",
    ["Min-Max Normalization", "Z-Score Standardization", "Robust Scaling"]
)

values = []

for row in data:
    values.append(row[column])

# Apply scaling
scaled_values = []

if method == "Min-Max Normalization":
    minimum = min(values)
    maximum = max(values)

    for value in values:
        if maximum == minimum:
            scaled_values.append(0)
        else:
            scaled_values.append((value - minimum) / (maximum - minimum))

elif method == "Z-Score Standardization":
    avg = mean(values)
    std = standard_deviation(values)

    for value in values:
        if std == 0:
            scaled_values.append(0)
        else:
            scaled_values.append((value - avg) / std)

else:
    med = median(values)
    median_absolute_deviation = mad(values)

    for value in values:
        if median_absolute_deviation == 0:
            scaled_values.append(0)
        else:
            scaled_values.append((value - med) / median_absolute_deviation)

# Show result
result = []

for i in range(len(values)):
    result.append({
        "Original": values[i],
        "Scaled": scaled_values[i]
    })

st.dataframe(result)

st.write("Skewness:", skewness(values))

if abs(skewness(values)) > 1:
    st.write("The distribution is strongly skewed, so robust scaling can be preferred.")
else:
    st.write("The distribution is not strongly skewed, so standardization or normalization can be used.")