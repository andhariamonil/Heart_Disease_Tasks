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
            row[columns[i]] = values[i].strip()

        data.append(row)

    return columns, data

# Count categories manually
def count_categories(data, column):
    categories = []
    counts = []

    for row in data:
        value = row[column]

        if value in categories:
            index = categories.index(value)
            counts[index] += 1
        else:
            categories.append(value)
            counts.append(1)

    return categories, counts

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

st.title("Heart Disease - Categorical Bar Plots")

column = st.selectbox(
    "Select categorical column",
    categorical_columns
)

categories, counts = count_categories(data, column)

fig, ax = plt.subplots()

bar_width = 0.6

if len(categories) > 8:
    bar_width = 0.4

bars = ax.bar(
    range(len(categories)),
    counts,
    width=bar_width
)

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories)
ax.set_xlabel(column)
ax.set_ylabel("Frequency")
ax.set_title("Distribution of " + column)

for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count,
        str(count),
        ha="center",
        va="bottom"
    )

st.pyplot(fig)

plt.close(fig)