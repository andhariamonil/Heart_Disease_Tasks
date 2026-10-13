import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import RandomForestScratch, load_data, metrics, stratified_split


st.title('Task 31: Random Forest From Scratch')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=5)
model = RandomForestScratch(n_trees=18, max_depth=4, min_samples_split=6, seed=5)
model.fit(X_train, y_train, features)
preds = model.predict(X_test)

st.write('Model metrics:')
st.dataframe(pd.DataFrame([metrics(y_test, preds)]))
st.write('Feature importances:')
st.bar_chart({features[i]: value for i, value in enumerate(model.importances) if value > 0})
