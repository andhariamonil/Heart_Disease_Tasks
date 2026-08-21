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

# Calculate histogram bins manually
def create_histogram(values, number_of_bins=10):
    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [minimum], [len(values)]

    bin_width = (maximum - minimum) / number_of_bins
    frequencies = [0] * number_of_bins

    for value in values:
        index = int((value - minimum) / bin_width)

        if index >= number_of_bins:
            index = number_of_bins - 1

        frequencies[index] += 1

    bin_values = []

    for i in range(number_of_bins):
        bin_values.append(minimum + i * bin_width)

    return bin_values, frequencies

columns, data = load_csv("heart.csv")

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

st.title("Heart Disease - Manual Histograms")

for column in numerical_columns:
    values = []

    for row in data:
        values.append(row[column])

    bins, frequencies = create_histogram(values)

    fig, ax = plt.subplots()

    width = bins[1] - bins[0] if len(bins) > 1 else 1

    ax.bar(
        bins,
        frequencies,
        width=width,
        align="edge",
        edgecolor="black"
    )

    ax.set_title("Histogram of " + column)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    plt.close(fig)