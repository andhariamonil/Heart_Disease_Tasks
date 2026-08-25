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

# Matrix multiplication
def multiply_matrix(a, b):
    result = []

    for i in range(len(a)):
        row = []

        for j in range(len(b[0])):
            total = 0

            for k in range(len(b)):
                total += a[i][k] * b[k][j]

            row.append(total)

        result.append(row)

    return result

# Matrix inverse using Gauss-Jordan elimination
def inverse_matrix(matrix):
    n = len(matrix)

    augmented = []

    for i in range(n):
        row = matrix[i].copy()

        for j in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)

        augmented.append(row)

    for i in range(n):
        pivot = augmented[i][i]

        if abs(pivot) < 0.0000001:
            return None

        for j in range(2 * n):
            augmented[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = augmented[k][i]

                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    result = []

    for i in range(n):
        result.append(
            augmented[i][n:]
        )

    return result

# Calculate R-squared manually
def calculate_r_squared(data, target, features):
    x = []
    y = []

    for row in data:
        x_row = [1]

        for feature in features:
            x_row.append(row[feature])

        x.append(x_row)
        y.append([row[target]])

    x_transpose = []

    for j in range(len(x[0])):
        column = []

        for i in range(len(x)):
            column.append(x[i][j])

        x_transpose.append(column)

    xtx = multiply_matrix(
        x_transpose,
        x
    )

    inverse = inverse_matrix(xtx)

    if inverse is None:
        return 0

    xty = multiply_matrix(
        x_transpose,
        y
    )

    coefficients = multiply_matrix(
        inverse,
        xty
    )

    predictions = []

    for row in x:
        prediction = 0

        for j in range(len(coefficients)):
            prediction += (
                row[j] *
                coefficients[j][0]
            )

        predictions.append(prediction)

    mean_y = 0

    for value in y:
        mean_y += value[0]

    mean_y /= len(y)

    ss_total = 0
    ss_residual = 0

    for i in range(len(y)):
        ss_total += (
            y[i][0] - mean_y
        ) ** 2

        ss_residual += (
            y[i][0] - predictions[i]
        ) ** 2

    if ss_total == 0:
        return 0

    return 1 - (
        ss_residual / ss_total
    )

columns, data = load_csv("heart.csv")

features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

st.title("Heart Disease - Manual VIF Analysis")

results = []

for feature in features:
    other_features = []

    for other in features:
        if other != feature:
            other_features.append(other)

    r_squared = calculate_r_squared(
        data,
        feature,
        other_features
    )

    if r_squared >= 0.999999:
        vif = float("inf")
    else:
        vif = 1 / (1 - r_squared)

    results.append(
        {
            "Feature": feature,
            "R-squared": round(r_squared, 4),
            "VIF": round(vif, 4)
            if vif != float("inf")
            else "Very High"
        }
    )

st.dataframe(results)

st.write("VIF interpretation:")

st.write("VIF < 5: Low multicollinearity")
st.write("VIF 5-10: Moderate multicollinearity")
st.write("VIF > 10: High multicollinearity")