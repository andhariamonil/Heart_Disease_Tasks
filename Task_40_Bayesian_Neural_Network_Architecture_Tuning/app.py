import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import MLPClassifierScratch, bayesian_architecture_search, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 40: Bayesian Neural-Network Architecture Tuning')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(dtype=float), frame['target'].to_numpy(dtype=int)
X_train_all, X_test, y_train_all, y_test = stratified_split(X, y, 0.2, seed=40)
X_train, X_validation, y_train, y_validation = stratified_split(X_train_all, y_train_all, 0.25, seed=41)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_validation = standardize(X_validation, means, scales)
X_test = standardize(X_test, means, scales)

architectures = [(8,), (16,), (24,), (12, 8), (16, 8), (24, 12), (16, 12, 8)]
scores_by_architecture = {}


def validation_score(architecture):
    model = MLPClassifierScratch(architecture, learning_rate=0.01, epochs=70, batch_size=32, dropout_rate=0.15, batch_normalization=True, seed=40 + sum(architecture)).fit(X_train, y_train)
    score = metrics(y_validation, model.predict(X_validation))['f1']
    scores_by_architecture[architecture] = (score, model)
    return score


search_results = bayesian_architecture_search(architectures, validation_score, initial=architectures[:2], evaluations=len(architectures), seed=40)
best_result = max(search_results, key=lambda row: row['score'])
best_architecture = best_result['architecture']
best_validation_model = scores_by_architecture[best_architecture][1]
test_rows = [{'Model': str(architecture), 'Validation F1': score} for architecture, (score, _) in scores_by_architecture.items()]
test_predictions = best_validation_model.predict(X_test)
st.subheader('Held-out test metrics for selected architecture')
st.dataframe(pd.DataFrame([{'Architecture': str(best_architecture), **metrics(y_test, test_predictions)}]).set_index('Architecture').style.format('{:.3f}'))
st.subheader('Bayesian search history')
history = pd.DataFrame([{'Evaluation': index + 1, 'Architecture': str(row['architecture']), 'Validation F1': row['score']} for index, row in enumerate(search_results)])
st.dataframe(history.set_index('Evaluation').style.format({'Validation F1': '{:.3f}'}))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(search_results) + 1), [row['score'] for row in search_results], marker='o', color='#147d64')
ax.set_xlabel('Bayesian search evaluation')
ax.set_ylabel('Validation F1')
ax.set_title('Gaussian-process expected-improvement search')
ax.set_xticks(range(1, len(search_results) + 1))
ax.grid(alpha=0.25)
st.pyplot(fig)
st.caption('A manual Gaussian-process surrogate and expected improvement select the next discrete layer/width configuration. The test set is held back until the winning validation architecture is chosen.')