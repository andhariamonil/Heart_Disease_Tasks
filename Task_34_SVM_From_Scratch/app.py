import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import SVMLinearScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 34: SVM From Scratch')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=31)
means, scales = standardize_fit(X_train)
X_train_scaled = standardize(X_train, means, scales)
X_test_scaled = standardize(X_test, means, scales)

model = SVMLinearScratch(learning_rate=0.01, epochs=1200, regularization=0.02)
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)

st.write('SVM metrics:')
st.dataframe(pd.DataFrame([metrics(y_test, preds)]))

st.write('Decision function values sample:')
st.dataframe(pd.DataFrame({'actual': y_test[:10], 'decision': model.decision_function(X_test_scaled[:10])}))
