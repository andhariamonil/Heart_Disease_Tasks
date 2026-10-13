import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, load_data, metrics, stratified_split


st.title('Task 27: Decision Tree Grid Search')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=3)

records = []
for depth in [2, 3, 4, 5]:
    for min_samples in [4, 6, 8, 12]:
        model = DecisionTreeScratch(max_depth=depth, min_samples_split=min_samples)
        model.fit(X_train, y_train, features)
        preds = model.predict(X_test)
        result = metrics(y_test, preds)
        records.append({
            'depth': depth,
            'min_samples_split': min_samples,
            'accuracy': round(result['accuracy'], 4),
            'f1': round(result['f1'], 4),
        })

results = pd.DataFrame(records).sort_values(['accuracy', 'f1'], ascending=False)
st.dataframe(results)

best = results.iloc[0]
st.write('Best grid setting:')
st.write(best.to_dict())
