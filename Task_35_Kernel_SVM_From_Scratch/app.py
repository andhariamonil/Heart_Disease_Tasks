import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import KernelSVMScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 35: Polynomial and RBF Kernel SVM')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X = frame[features].to_numpy(dtype=float)
y = frame['target'].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=35)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_test = standardize(X_test, means, scales)

models = {
    'Polynomial (degree 2)': KernelSVMScratch('polynomial', degree=2),
    'RBF': KernelSVMScratch('rbf', gamma=1 / X_train.shape[1]),
}
rows = []
distribution = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rows.append({'Kernel': name, **metrics(y_test, predictions)})
    for label in (0, 1):
        mask = model.support_mask & (y_train == label)
        distribution.append({'Kernel': name, 'Training class': label, 'Support vectors': int(mask.sum()), 'Share of class': float(mask.sum() / max((y_train == label).sum(), 1))})

st.subheader('Held-out performance')
st.dataframe(pd.DataFrame(rows).set_index('Kernel'))
st.subheader('Support-vector distribution')
st.dataframe(pd.DataFrame(distribution))
st.caption(f'Unique records: {len(frame)}; training rows: {len(y_train)}; test rows: {len(y_test)}; features: {len(features)}. The kernels and SMO optimizer are implemented manually.')