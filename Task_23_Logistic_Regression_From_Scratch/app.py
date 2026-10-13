import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 23: Logistic Regression From Scratch')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, df = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=42)
means, scales = standardize_fit(X_train)
X_train_scaled = standardize(X_train, means, scales)
X_test_scaled = standardize(X_test, means, scales)

model = LogisticRegressionScratch(learning_rate=0.08, iterations=1500, l2=0.01)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
result = metrics(y_test, preds)

st.write('Feature count:', len(features))
st.write('Train samples:', len(X_train))
st.write('Test samples:', len(X_test))
st.table(pd.DataFrame([result]))

st.write('Prediction sample:')
st.dataframe(pd.DataFrame({'actual': y_test[:10], 'predicted': preds[:10]}))
