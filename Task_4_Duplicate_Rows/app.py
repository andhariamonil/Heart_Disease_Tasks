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
        row = []

        for value in values:
            row.append(value.strip())

        data.append(row)

    return columns, data

# Compare two rows manually
def rows_are_equal(row1, row2):
    if len(row1) != len(row2):
        return False

    for i in range(len(row1)):
        if row1[i] != row2[i]:
            return False

    return True

# Find duplicate rows
def find_duplicates(data):
    duplicate_indexes = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if rows_are_equal(data[i], data[j]):
                duplicate_indexes.append(j)

    return duplicate_indexes

# Remove duplicates manually
def remove_duplicates(data):
    unique_data = []

    for row in data:
        duplicate = False

        for existing_row in unique_data:
            if rows_are_equal(row, existing_row):
                duplicate = True
                break

        if not duplicate:
            unique_data.append(row)

    return unique_data

st.title("Heart Disease - Duplicate Rows")

columns, data = load_csv("heart.csv")

st.write("Original rows:", len(data))

duplicate_indexes = find_duplicates(data)
duplicate_count = len(duplicate_indexes)

st.write("Duplicate rows:", duplicate_count)

cleaned_data = remove_duplicates(data)

st.write("Rows after removing duplicates:", len(cleaned_data))

st.subheader("Data Integrity")

if duplicate_count == 0:
    st.write("No duplicate rows were found in the dataset.")
else:
    st.write("Duplicate rows can cause repeated observations and affect data integrity.")
    st.write("In machine learning, duplicates can give repeated records more influence during training and may lead to misleading model evaluation.")

if duplicate_count > 0:
    st.subheader("Duplicate Rows")

    duplicate_rows = []

    for index in duplicate_indexes:
        row_dictionary = {}

        for i in range(len(columns)):
            row_dictionary[columns[i]] = data[index][i]

        duplicate_rows.append(row_dictionary)

    st.dataframe(duplicate_rows)

st.subheader("Cleaned Dataset")

cleaned_display = []

for row in cleaned_data:
    row_dictionary = {}

    for i in range(len(columns)):
        row_dictionary[columns[i]] = row[i]

    cleaned_display.append(row_dictionary)

st.dataframe(cleaned_display)