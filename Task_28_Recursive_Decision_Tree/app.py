import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, load_data, stratified_split


st.title('Task 28: Recursive Decision Tree')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=9)
model = DecisionTreeScratch(max_depth=4, min_samples_split=6)
model.fit(X_train, y_train, features)

sample = X_test[0]
path = model.decision_path(sample)
st.write('Decision path for the first test sample:')
for step in path:
    st.write('-', step)

st.write('Tree feature importance:')
st.bar_chart({features[i]: value for i, value in enumerate(model.importances) if value > 0})
