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

# Perform manual one-hot encoding
def one_hot_encode(data, column):
    categories = []

    for row in data:
        value = row[column]

        if value not in categories:
            categories.append(value)

    encoded_data = []

    for row in data:
        new_row = row.copy()

        del new_row[column]

        for category in categories:
            new_column = column + "_" + str(category)

            if row[column] == category:
                new_row[new_column] = 1
            else:
                new_row[new_column] = 0

        encoded_data.append(new_row)

    return encoded_data, categories

# Load dataset
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

st.title("Heart Disease - One Hot Encoding")

column = st.selectbox(
    "Select categorical column",
    categorical_columns
)

encoded_data, categories = one_hot_encode(data, column)

st.write("Categories found:", categories)

st.subheader("Encoded Dataset")

st.dataframe(encoded_data)

# Test unseen category
st.subheader("Unseen Category")

new_category = st.text_input("Enter a new category")

if new_category != "":
    if new_category not in categories:
        categories.append(new_category)

    st.write("Updated categories:", categories)