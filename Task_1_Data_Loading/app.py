import streamlit as st


# Read the CSV file manually
def load_csv(file_path):

    # Open file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Detect delimiter from first line
    first_line = lines[0]

    if "," in first_line:
        delimiter = ","
    elif ";" in first_line:
        delimiter = ";"
    elif "\t" in first_line:
        delimiter = "\t"
    else:
        delimiter = "|"

    # Read column names
    columns = first_line.strip().split(delimiter)

    # Read data rows manually
    data = []

    for line in lines[1:]:
        if line.strip() == "":
            continue

        values = line.strip().split(delimiter)

        row = {}

        for i in range(len(columns)):
            value = values[i].strip()

            # Convert numbers
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            row[columns[i]] = value

        data.append(row)

    return columns, data, delimiter


# Load dataset
columns, data, delimiter = load_csv("heart.csv")


# Title
st.title("Heart Disease Dataset")

# Show dataset information
st.write("Number of rows:", len(data))
st.write("Number of columns:", len(columns))
st.write("Detected delimiter:", repr(delimiter))


# Sorting
st.subheader("Sorting")

sort_column = st.selectbox(
    "Sort by",
    columns
)

sort_order = st.selectbox(
    "Order",
    ["Ascending", "Descending"]
)


# Sort data manually
column_index = columns.index(sort_column)

def sort_key(row):
    return row[sort_column]


data.sort(
    key=sort_key,
    reverse=(sort_order == "Descending")
)


# Pagination
st.subheader("Pagination")

rows_per_page = st.selectbox(
    "Rows per page",
    [5, 10, 20, 50, 100]
)

total_pages = (
    len(data) + rows_per_page - 1
) // rows_per_page

page = st.number_input(
    "Page",
    min_value=1,
    max_value=total_pages,
    value=1
)


# Calculate rows for current page
start = (page - 1) * rows_per_page
end = start + rows_per_page

page_data = data[start:end]


# Display dataset
st.subheader("Dataset")

st.dataframe(page_data)

st.write(
    "Page",
    page,
    "of",
    total_pages
)