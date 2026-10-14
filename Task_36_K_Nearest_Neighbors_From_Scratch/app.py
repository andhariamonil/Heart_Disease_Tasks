import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import KNearestNeighborsScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 36: k-Nearest Neighbors From Scratch')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X = frame[features].to_numpy(dtype=float)
y = frame['target'].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=36)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_test = standardize(X_test, means, scales)
k = st.slider('Number of neighbors (k)', min_value=1, max_value=25, value=5, step=2)
leaf_size = st.slider('KD-tree leaf size', min_value=4, max_value=32, value=16, step=4)

model = KNearestNeighborsScratch(k=k, leaf_size=leaf_size).fit(X_train, y_train)
predictions = model.predict(X_test)
model.predict_one(X_test[0])
st.subheader('Held-out performance')
st.dataframe(pd.DataFrame([metrics(y_test, predictions)]))
st.write(f'KD-tree nodes visited for the first test query: {model.tree.last_nodes_visited}')
st.caption('Distances, KD-tree construction, branch pruning, nearest-neighbor search, and tie-breaking are implemented manually.')
st.write('Feature columns:', ', '.join(features))