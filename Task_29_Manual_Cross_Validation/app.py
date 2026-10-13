import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, load_data, metrics


st.title('Task 29: Manual Cross Validation')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
folds = 5
indices = np.array_split(np.arange(len(y)), folds)
records = []

for fold_id, test_index in enumerate(indices, start=1):
    mask = np.zeros(len(y), dtype=bool)
    mask[test_index] = True
    X_train = X[~mask]
    X_test = X[mask]
    y_train = y[~mask]
    y_test = y[mask]
    model = DecisionTreeScratch(max_depth=3, min_samples_split=6)
    model.fit(X_train, y_train, features)
    preds = model.predict(X_test)
    score = metrics(y_test, preds)
    records.append({
        'fold': fold_id,
        'accuracy': round(score['accuracy'], 4),
        'f1': round(score['f1'], 4),
    })

summary = pd.DataFrame(records)
st.dataframe(summary)
st.write('Mean accuracy:', round(summary['accuracy'].mean(), 4))
st.write('Mean f1:', round(summary['f1'].mean(), 4))
