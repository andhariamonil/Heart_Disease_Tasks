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

# Calculate Gini impurity
def gini(data):
    if len(data) == 0:
        return 0

    count_0 = 0
    count_1 = 0

    for row in data:
        if row["target"] == 0:
            count_0 += 1
        elif row["target"] == 1:
            count_1 += 1

    total = len(data)

    probability_0 = count_0 / total
    probability_1 = count_1 / total

    return 1 - (
        probability_0 ** 2 +
        probability_1 ** 2
    )

# Split data manually
def split_data(data, feature, threshold):
    left = []
    right = []

    for row in data:
        if row[feature] <= threshold:
            left.append(row)
        else:
            right.append(row)

    return left, right

# Calculate Gini gain
def gini_gain(data, left, right):
    parent_gini = gini(data)

    left_weight = len(left) / len(data)
    right_weight = len(right) / len(data)

    child_gini = (
        left_weight * gini(left)
        + right_weight * gini(right)
    )

    return parent_gini - child_gini

# Find best split
def best_split(data, feature):
    values = []

    for row in data:
        values.append(row[feature])

    values.sort()

    thresholds = []

    for i in range(len(values) - 1):
        threshold = (
            values[i] + values[i + 1]
        ) / 2

        if threshold not in thresholds:
            thresholds.append(threshold)

    best_gain = 0
    best_threshold = None

    for threshold in thresholds:
        left, right = split_data(
            data,
            feature,
            threshold
        )

        if len(left) == 0 or len(right) == 0:
            continue

        gain = gini_gain(
            data,
            left,
            right
        )

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold

columns, data = load_csv("heart.csv")

features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

st.title("Heart Disease - Decision Tree Feature Importance")

results = []

for feature in features:
    gain, threshold = best_split(
        data,
        feature
    )

    results.append(
        {
            "Feature": feature,
            "Best Threshold": round(threshold, 3)
            if threshold is not None
            else "None",
            "Gini Gain": round(gain, 6)
        }
    )

results.sort(
    key=lambda row: row["Gini Gain"],
    reverse=True
)

st.subheader("Feature Importance Ranking")

for index, result in enumerate(results):
    st.write(
        str(index + 1)
        + ". "
        + result["Feature"]
        + " - Gini Gain: "
        + str(result["Gini Gain"])
    )

st.subheader("Details")

st.dataframe(results)

st.write(
    "Higher Gini gain indicates greater feature importance."
)