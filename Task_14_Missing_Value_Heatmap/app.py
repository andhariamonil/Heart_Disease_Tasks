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
        row = []

        for value in values:
            row.append(value.strip())

        data.append(row)

    return columns, data

# Create missing value grid
def create_grid(data, demo_missing):
    grid = []

    for i in range(len(data)):
        row = []

        for j in range(len(data[i])):
            value = data[i][j]

            if value == "":
                row.append(1)
            elif demo_missing and i == 0 and j == 0:
                row.append(1)
            else:
                row.append(0)

        grid.append(row)

    return grid

columns, data = load_csv("heart.csv")

st.title("Heart Disease - Missing Value Heatmap")

demo_missing = st.checkbox("Demo missing value")

grid = create_grid(data, demo_missing)

missing_count = 0

for row in grid:
    for value in row:
        if value == 1:
            missing_count += 1

st.write("Missing values:", missing_count)

if missing_count == 0:
    st.write("No missing values were found in the dataset.")
else:
    fig, ax = plt.subplots()

    ax.imshow(
        grid,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_title("Missing Value Heatmap")

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90)

    st.pyplot(fig)

    plt.close(fig)