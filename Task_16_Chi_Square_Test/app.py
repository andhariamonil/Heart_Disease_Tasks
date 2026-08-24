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
            row[columns[i]] = values[i].strip()

        data.append(row)

    return columns, data

# Create observed frequency table
def observed_table(data, column, target):
    categories = []
    targets = []

    for row in data:
        category = row[column]
        target_value = row[target]

        if category not in categories:
            categories.append(category)

        if target_value not in targets:
            targets.append(target_value)

    table = {}

    for category in categories:
        table[category] = {}

        for target_value in targets:
            table[category][target_value] = 0

    for row in data:
        table[row[column]][row[target]] += 1

    return categories, targets, table

# Calculate chi-square manually
def chi_square(data, column, target):
    categories, targets, observed = observed_table(
        data,
        column,
        target
    )

    row_totals = {}
    column_totals = {}

    for category in categories:
        row_totals[category] = 0

        for target_value in targets:
            row_totals[category] += observed[category][target_value]

    for target_value in targets:
        column_totals[target_value] = 0

        for category in categories:
            column_totals[target_value] += observed[category][target_value]

    total = len(data)
    expected = {}
    statistic = 0

    for category in categories:
        expected[category] = {}

        for target_value in targets:
            expected_value = (
                row_totals[category] *
                column_totals[target_value]
            ) / total

            expected[category][target_value] = expected_value

            if expected_value > 0:
                observed_value = observed[category][target_value]

                statistic += (
                    (observed_value - expected_value) ** 2
                ) / expected_value

    degrees_of_freedom = (
        (len(categories) - 1) *
        (len(targets) - 1)
    )

    return (
        categories,
        targets,
        observed,
        expected,
        statistic,
        degrees_of_freedom
    )

# Approximate critical value for common significance levels
def critical_value(degrees_of_freedom):
    values = {
        1: 3.841,
        2: 5.991,
        3: 7.815,
        4: 9.488,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
        10: 18.307,
        11: 19.675,
        12: 21.026,
        13: 22.362,
        14: 23.685,
        15: 24.996
    }

    return values.get(degrees_of_freedom, None)

columns, data = load_csv("heart.csv")

categorical_columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

st.title("Heart Disease - Chi-Square Test")

column = st.selectbox(
    "Select categorical column",
    categorical_columns
)

(
    categories,
    targets,
    observed,
    expected,
    statistic,
    degrees_of_freedom
) = chi_square(
    data,
    column,
    "target"
)

st.subheader("Observed Frequencies")

observed_display = []

for category in categories:
    row = {"Category": category}

    for target_value in targets:
        row["Target " + target_value] = observed[category][target_value]

    observed_display.append(row)

st.dataframe(observed_display)

st.subheader("Expected Frequencies")

expected_display = []

for category in categories:
    row = {"Category": category}

    for target_value in targets:
        row["Target " + target_value] = round(
            expected[category][target_value],
            3
        )

    expected_display.append(row)

st.dataframe(expected_display)

st.write("Chi-square statistic:", round(statistic, 4))
st.write("Degrees of freedom:", degrees_of_freedom)

critical = critical_value(degrees_of_freedom)

if critical is not None:
    st.write("Critical value at 0.05:", critical)

    if statistic > critical:
        st.write("Result: Reject the null hypothesis.")
    else:
        st.write("Result: Fail to reject the null hypothesis.")
else:
    st.write("Critical value is not included for this degree of freedom.")