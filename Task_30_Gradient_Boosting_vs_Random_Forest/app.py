import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import GradientBoostingScratch, RandomForestScratch, load_data, metrics, stratified_split


st.title('Task 30: Gradient Boosting vs Random Forest')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=22)

gb = GradientBoostingScratch(n_estimators=20, learning_rate=0.12)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
forest = RandomForestScratch(n_trees=12, max_depth=4, min_samples_split=6, seed=22)
forest.fit(X_train, y_train, features)
forest_preds = forest.predict(X_test)

results = pd.DataFrame([
    {'model': 'Gradient Boosting', **metrics(y_test, gb_preds)},
    {'model': 'Random Forest', **metrics(y_test, forest_preds)},
])
st.dataframe(results[['model', 'accuracy', 'precision', 'recall', 'f1']])
