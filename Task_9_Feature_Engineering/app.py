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
            row[columns[i]] = float(values[i])

        data.append(row)

    return columns, data

# Calculate percentile
def percentile(values, value):
    count = 0

    for item in values:
        if item <= value:
            count += 1

    return (count / len(values)) * 100

# Create risk factor
def calculate_risk_factor(chol, thalach, chol_percentile, thalach_percentile):
    cholesterol_risk = chol / 200
    heart_rate_risk = 220 / thalach
    interaction = cholesterol_risk * heart_rate_risk

    return (
        cholesterol_risk
        + heart_rate_risk
        + interaction
        + chol_percentile / 100
        + (100 - thalach_percentile) / 100
    )

# Load dataset
columns, data = load_csv("heart.csv")

chol_values = []
thalach_values = []

for row in data:
    chol_values.append(row["chol"])
    thalach_values.append(row["thalach"])

# Create new feature
result = []

for row in data:
    chol_percentile = percentile(chol_values, row["chol"])
    thalach_percentile = percentile(thalach_values, row["thalach"])

    risk_factor = calculate_risk_factor(
        row["chol"],
        row["thalach"],
        chol_percentile,
        thalach_percentile
    )

    new_row = row.copy()
    new_row["chol_percentile"] = chol_percentile
    new_row["thalach_percentile"] = thalach_percentile
    new_row["risk_factor"] = risk_factor

    result.append(new_row)

st.title("Heart Disease - Feature Engineering")

st.write("New feature: risk_factor")

st.dataframe(result)