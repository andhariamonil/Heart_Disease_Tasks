import streamlit as st


# Load CSV manually
def load_csv(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Detect delimiter
    first_line = lines[0]

    if "," in first_line:
        delimiter = ","
    elif ";" in first_line:
        delimiter = ";"
    else:
        delimiter = "\t"

    columns = first_line.strip().split(delimiter)

    data = []

    for line in lines[1:]:

        if line.strip() == "":
            continue

        values = line.strip().split(delimiter)

        row = {}

        for i in range(len(columns)):

            try:
                row[columns[i]] = float(values[i])
            except ValueError:
                row[columns[i]] = values[i]

        data.append(row)

    return columns, data


# Load dataset
columns, data = load_csv("heart.csv")


# Select a numerical column
numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

column = st.selectbox(
    "Select a column",
    numerical_columns
)


# Get values
values = []

for row in data:
    values.append(row[column])


# ------------------------------------------------
# MEAN
# ------------------------------------------------

total = 0

for value in values:
    total += value

mean = total / len(values)


# ------------------------------------------------
# MEDIAN
# ------------------------------------------------

sorted_values = values.copy()
sorted_values.sort()

n = len(sorted_values)

if n % 2 == 0:
    median = (
        sorted_values[n // 2 - 1]
        + sorted_values[n // 2]
    ) / 2
else:
    median = sorted_values[n // 2]


# ------------------------------------------------
# MODE
# ------------------------------------------------

counts = {}

for value in values:

    if value in counts:
        counts[value] += 1
    else:
        counts[value] = 1

mode = values[0]
highest_count = counts[mode]

for value in counts:

    if counts[value] > highest_count:
        highest_count = counts[value]
        mode = value


# ------------------------------------------------
# MINIMUM
# ------------------------------------------------

minimum = values[0]

for value in values:

    if value < minimum:
        minimum = value


# ------------------------------------------------
# MAXIMUM
# ------------------------------------------------

maximum = values[0]

for value in values:

    if value > maximum:
        maximum = value


# ------------------------------------------------
# STANDARD DEVIATION
# ------------------------------------------------

squared_difference = 0

for value in values:

    difference = value - mean
    squared_difference += difference * difference

variance = squared_difference / len(values)

std = variance ** 0.5


# ------------------------------------------------
# DISPLAY
# ------------------------------------------------

st.title("Heart Disease - Summary Statistics")

st.write("Column:", column)

st.write("Mean:", mean)
st.write("Median:", median)
st.write("Mode:", mode)
st.write("Minimum:", minimum)
st.write("Maximum:", maximum)
st.write("Standard Deviation:", std)