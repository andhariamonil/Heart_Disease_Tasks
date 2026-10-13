import sys
from pathlib import Path
import random

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import GradientBoostingScratch, load_data, metrics, stratified_split


st.title('Task 32: Gradient Boosting Random Search')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=18)

random.seed(18)
records = []
for _ in range(8):
    n_estimators = random.choice([8, 12, 16, 24, 32])
    learning_rate = random.choice([0.05, 0.08, 0.12, 0.18, 0.25])
    model = GradientBoostingScratch(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = metrics(y_test, preds)
    records.append({
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'accuracy': round(score['accuracy'], 4),
        'f1': round(score['f1'], 4),
    })

results = pd.DataFrame(records).sort_values(['accuracy', 'f1'], ascending=False)
st.dataframe(results)
st.write('Best configuration:')
st.write(results.iloc[0].to_dict())
