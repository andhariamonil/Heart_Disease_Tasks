import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, confusion_frame, confusion_matrix_manual, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 26: Manual Confusion Matrix')

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'

X, y, features, _ = load_data(csv_path)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=12)
means, scales = standardize_fit(X_train)
X_train_scaled = standardize(X_train, means, scales)
X_test_scaled = standardize(X_test, means, scales)

model = LogisticRegressionScratch(learning_rate=0.08, iterations=1800)
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)
cm = confusion_matrix_manual(y_test, preds)

st.dataframe(confusion_frame(cm))
st.write('Accuracy:', metrics(y_test, preds)['accuracy'])

fig, ax = plt.subplots()
ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred 0', 'Pred 1'])
ax.set_yticklabels(['Actual 0', 'Actual 1'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
ax.set_title('Manual Confusion Matrix')
st.pyplot(fig)
