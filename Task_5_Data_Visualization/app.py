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

columns, data = load_csv("heart.csv")

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

st.title("Heart Disease - Data Visualization")

st.subheader("Distribution of Numerical Columns")

for column in numerical_columns:
    values = []

    for row in data:
        values.append(row[column])

    fig, ax = plt.subplots()

    ax.hist(values, bins=10, edgecolor="black")
    ax.set_title("Distribution of " + column)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.annotate(
        "Number of values: " + str(len(values)),
        xy=(0.05, 0.90),
        xycoords="axes fraction"
    )
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

st.subheader("Comparative Distribution")

fig, ax = plt.subplots()

for column in numerical_columns:
    values = []

    for row in data:
        values.append(row[column])

    ax.hist(
        values,
        bins=10,
        alpha=0.4,
        label=column
    )

ax.set_title("Comparison of Numerical Columns")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.subheader("Box Plot Comparison")

all_values = []

for column in numerical_columns:
    values = []

    for row in data:
        values.append(row[column])

    all_values.append(values)

fig, ax = plt.subplots()

ax.boxplot(
    all_values,
    label=numerical_columns
)

ax.set_title("Numerical Column Comparison")
ax.set_xlabel("Columns")
ax.set_ylabel("Values")
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)