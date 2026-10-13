import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, roc_points, standardize, standardize_fit, stratified_split


st.title('Task 33: Manual ROC-AUC')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=27)
means, scales = standardize_fit(X_train)
X_train_scaled = standardize(X_train, means, scales)
X_test_scaled = standardize(X_test, means, scales)

model = LogisticRegressionScratch(learning_rate=0.05, iterations=2000)
model.fit(X_train_scaled, y_train)
scores = model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, auc = roc_points(y_test, scores)

st.write('Manual AUC:', round(auc, 4))

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Manual ROC Curve')
ax.legend()
st.pyplot(fig)

st.dataframe(pd.DataFrame({'fpr': fpr[:10], 'tpr': tpr[:10]}))
