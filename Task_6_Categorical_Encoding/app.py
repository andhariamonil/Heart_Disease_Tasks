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
            row[columns[i]] = values[i].strip()

        data.append(row)

    return columns, data

# Encode categories manually
def encode_column(data, column):
    mapping = {}
    next_number = 0
    encoded_data = []

    for row in data:
        value = row[column]

        if value not in mapping:
            mapping[value] = next_number
            next_number += 1

        encoded_row = row.copy()
        encoded_row[column] = mapping[value]
        encoded_data.append(encoded_row)

    return encoded_data, mapping

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

st.title("Heart Disease - Categorical Encoding")

column = st.selectbox(
    "Select categorical column",
    categorical_columns
)

encoded_data, mapping = encode_column(data, column)

st.subheader("Category Mapping")

for category in mapping:
    st.write(category, "→", mapping[category])

st.subheader("Encoded Dataset")

st.dataframe(encoded_data)

st.subheader("Unknown Value Handling")

unknown_value = st.text_input("Enter a new category")

if unknown_value != "":
    if unknown_value in mapping:
        st.write("Existing category:", mapping[unknown_value])
    else:
        new_number = len(mapping)

        st.write("Unknown category detected.")
        st.write("New numerical value:", new_number)