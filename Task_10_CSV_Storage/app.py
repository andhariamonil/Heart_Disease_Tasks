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

# Write CSV manually
def write_csv(file_path, columns, data):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")

        for row in data:
            values = []

            for column in columns:
                value = str(row[column]).replace(",", " ")
                values.append(value)

            file.write(",".join(values) + "\n")

columns, data = load_csv("heart.csv")

output_file = "processed_heart.csv"

write_csv(output_file, columns, data)

st.title("Heart Disease - CSV Storage")

st.write("Original rows:", len(data))
st.write("Columns:", len(columns))
st.write("Processed file:", output_file)

with open(output_file, "rb") as file:
    st.download_button(
        "Download Processed CSV",
        file,
        file_name=output_file,
        mime="text/csv"
    )