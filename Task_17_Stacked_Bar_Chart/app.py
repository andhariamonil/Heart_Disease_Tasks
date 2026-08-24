import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# Count sex and target combinations
def count_data(data):
    counts = {
        "0": {"0": 0, "1": 0},
        "1": {"0": 0, "1": 0}
    }

    for row in data:
        sex = row["sex"]
        target = row["target"]

        if sex in counts and target in counts[sex]:
            counts[sex][target] += 1

    return counts

columns, data = load_csv("heart.csv")

counts = count_data(data)

st.title("Heart Disease - Sex vs Target")

fig, ax = plt.subplots()

bar_width = 0.6
x_positions = [0, 1]

for index, sex in enumerate(["0", "1"]):
    total = counts[sex]["0"] + counts[sex]["1"]

    if total == 0:
        continue

    proportion_0 = counts[sex]["0"] / total
    proportion_1 = counts[sex]["1"] / total

    rectangle_0 = Rectangle(
        (
            index - bar_width / 2,
            0
        ),
        bar_width,
        proportion_0
    )

    rectangle_1 = Rectangle(
        (
            index - bar_width / 2,
            proportion_0
        ),
        bar_width,
        proportion_1
    )

    ax.add_patch(rectangle_0)
    ax.add_patch(rectangle_1)

    ax.text(
        index,
        proportion_0 / 2,
        str(round(proportion_0 * 100, 1)) + "%",
        ha="center",
        va="center"
    )

    ax.text(
        index,
        proportion_0 + proportion_1 / 2,
        str(round(proportion_1 * 100, 1)) + "%",
        ha="center",
        va="center"
    )

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 1)

ax.set_xticks(x_positions)
ax.set_xticklabels(["Sex 0", "Sex 1"])

ax.set_ylabel("Proportion")
ax.set_xlabel("Sex")
ax.set_title("Sex vs Target")

st.pyplot(fig)

plt.close(fig)

st.write("Target 0 = No heart disease")
st.write("Target 1 = Heart disease")