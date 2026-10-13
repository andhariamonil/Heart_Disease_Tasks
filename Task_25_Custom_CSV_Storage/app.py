import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import load_data


def save_csv(frame, output_path):
    frame.to_csv(output_path, index=False)
    return output_path


st.title('Task 25: Custom CSV Storage')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

_, _, _, df = load_data(csv_path)
cleaned = df.drop_duplicates().dropna()
out_path = Path(__file__).with_name('processed_heart.csv')
result = save_csv(cleaned, out_path)

st.write('Saved file:', result)
st.write('Rows saved:', len(cleaned))
st.dataframe(cleaned.head())
