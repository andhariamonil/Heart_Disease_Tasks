import streamlit as st


# --------------------------------------------------
# LOAD CSV MANUALLY
# --------------------------------------------------

def load_csv(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Detect delimiter
    first_line = lines[0]

    if "," in first_line:
        delimiter = ","
    elif ";" in first_line:
        delimiter = ";"
    else:
        delimiter = "\t"

    columns = first_line.strip().split(delimiter)

    data = []

    for line in lines[1:]:

        if line.strip() == "":
            continue

        values = line.strip().split(delimiter)

        row = []

        for value in values:

            value = value.strip()

            if value == "" or value.lower() in ["na", "n/a", "null", "?"]:
                row.append(None)
            else:
                try:
                    row.append(float(value))
                except ValueError:
                    row.append(value)

        data.append(row)

    return columns, data


# --------------------------------------------------
# MISSING VALUE CHECK
# --------------------------------------------------

def is_missing(value):

    return value is None


def count_missing(data, column_index):

    count = 0

    for row in data:

        if column_index >= len(row):
            count += 1

        elif is_missing(row[column_index]):
            count += 1

    return count


# --------------------------------------------------
# DEMO MISSING VALUES
# --------------------------------------------------

def create_demo_data(data, column_index, number):

    demo_data = []

    for row in data:
        demo_data.append(row.copy())

    count = 0

    for row in demo_data:

        if count >= number:
            break

        row[column_index] = None
        count += 1

    return demo_data


# --------------------------------------------------
# GET VALID VALUES
# --------------------------------------------------

def get_values(data, column_index):

    values = []

    for row in data:

        if column_index < len(row):

            value = row[column_index]

            if not is_missing(value):
                values.append(value)

    return values


# --------------------------------------------------
# MEAN
# --------------------------------------------------

def calculate_mean(values):

    total = 0

    for value in values:
        total += value

    return total / len(values)


# --------------------------------------------------
# MEDIAN
# --------------------------------------------------

def calculate_median(values):

    values = values.copy()
    values.sort()

    n = len(values)

    if n % 2 == 1:
        return values[n // 2]

    return (
        values[n // 2 - 1]
        + values[n // 2]
    ) / 2


# --------------------------------------------------
# MODE
# --------------------------------------------------

def calculate_mode(values):

    frequency = {}

    for value in values:

        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1

    mode = values[0]
    highest = frequency[mode]

    for value in frequency:

        if frequency[value] > highest:
            mode = value
            highest = frequency[value]

    return mode


# --------------------------------------------------
# IMPUTE VALUES
# --------------------------------------------------

def impute(data, column_index, replacement):

    new_data = []

    for row in data:

        new_row = row.copy()

        if is_missing(new_row[column_index]):
            new_row[column_index] = replacement

        new_data.append(new_row)

    return new_data


# --------------------------------------------------
# STREAMLIT
# --------------------------------------------------

st.title("Heart Disease - Missing Value Imputation")


# Load dataset
columns, original_data = load_csv("heart.csv")


# Demo option
demo = st.checkbox("Demo Missing Values")


if demo:

    column = st.selectbox(
        "Select column",
        columns[:-1]
    )

    number = st.number_input(
        "Number of missing values",
        min_value=1,
        max_value=20,
        value=5
    )

    column_index = columns.index(column)

    data = create_demo_data(
        original_data,
        column_index,
        number
    )

else:

    data = original_data


# --------------------------------------------------
# SHOW MISSING VALUES
# --------------------------------------------------

st.subheader("Missing Values")

missing_table = []

for i in range(len(columns)):

    missing = count_missing(data, i)

    missing_table.append(
        {
            "Column": columns[i],
            "Missing Values": missing
        }
    )

st.dataframe(missing_table)


# --------------------------------------------------
# SELECT COLUMN WITH MISSING VALUES
# --------------------------------------------------

columns_with_missing = []

for i in range(len(columns) - 1):

    if count_missing(data, i) > 0:
        columns_with_missing.append(columns[i])


if len(columns_with_missing) == 0:

    st.info(
        "No missing values found. "
        "Enable Demo Missing Values to test imputation."
    )

else:

    selected_column = st.selectbox(
        "Column for imputation",
        columns_with_missing
    )

    column_index = columns.index(selected_column)

    values = get_values(
        data,
        column_index
    )

    # Determine column type
    if selected_column in [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "ca"
    ]:

        column_type = "Numerical"

    else:

        column_type = "Categorical"


    # --------------------------------------------------
    # NUMERICAL IMPUTATION
    # --------------------------------------------------

    if column_type == "Numerical":

        mean = calculate_mean(values)
        median = calculate_median(values)

        method = st.selectbox(
            "Imputation method",
            [
                "Mean",
                "Median"
            ]
        )

        if method == "Mean":
            replacement = mean
        else:
            replacement = median


    # --------------------------------------------------
    # CATEGORICAL IMPUTATION
    # --------------------------------------------------

    else:

        mode = calculate_mode(values)

        method = "Mode"
        replacement = mode

        st.write(
            "Mode:",
            mode
        )


    # --------------------------------------------------
    # APPLY IMPUTATION
    # --------------------------------------------------

    imputed_data = impute(
        data,
        column_index,
        replacement
    )


    before = count_missing(
        data,
        column_index
    )

    after = count_missing(
        imputed_data,
        column_index
    )


    st.write("Method:", method)
    st.write("Replacement value:", replacement)
    st.write("Missing before:", before)
    st.write("Missing after:", after)


    if after == 0:
        st.success("Missing values successfully imputed.")


# --------------------------------------------------
# METHOD JUSTIFICATION
# --------------------------------------------------

st.subheader("Method Justification")

st.write(
    "Mean and Median are used for numerical columns. "
    "Median is less affected by extreme values."
)

st.write(
    "Mode is used for categorical columns because "
    "it replaces missing values with the most common category."
)