import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import load_data


def iqr_outliers(frame, column):
    q1 = frame[column].quantile(0.25)
    q3 = frame[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (frame[column] < lower) | (frame[column] > upper)
    return mask, lower, upper


st.title('Task 22: Outlier Handling')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

_, _, _, df = load_data(csv_path)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected = st.multiselect('Select numeric columns to inspect', numeric_cols, default=numeric_cols[:4])

for column in selected:
    mask, lower, upper = iqr_outliers(df, column)
    st.subheader(f'{column}')
    st.write(f'Lower bound: {lower:.3f} | Upper bound: {upper:.3f}')
    st.write(f'Outlier count: {int(mask.sum())}')
    clipped = df[column].clip(lower=lower, upper=upper)
    st.line_chart(pd.DataFrame({column: clipped}))

cleaned = df.copy()
for column in selected:
    mask, lower, upper = iqr_outliers(cleaned, column)
    cleaned.loc[mask, column] = np.clip(cleaned.loc[mask, column], lower, upper)

st.subheader('Cleaned snapshot')
st.dataframe(cleaned.head())
