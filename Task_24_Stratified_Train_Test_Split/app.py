import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import load_data, stratified_split


st.title('Task 24: Stratified Train-Test Split')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, df = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=7)

label_values = sorted(set(y_train.tolist()))
train_counts = {int(label): int((y_train == label).sum()) for label in label_values}
test_counts = {int(label): int((y_test == label).sum()) for label in sorted(set(y_test.tolist()))}

st.write('Train shape:', X_train.shape)
st.write('Test shape:', X_test.shape)
st.write('Train target distribution:', train_counts)
st.write('Test target distribution:', test_counts)

summary = pd.DataFrame({
    'split': ['train', 'test'],
    'samples': [len(X_train), len(X_test)],
    'positive_rate': [y_train.mean(), y_test.mean()],
})
st.dataframe(summary)
