import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, metrics, standardize, standardize_fit, stratified_split


def stratified_folds(labels, count=4, seed=48):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(count)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for fold_index, part in enumerate(np.array_split(indices, count)):
            folds[fold_index].extend(part.tolist())
    return [np.array(sorted(part), dtype=int) for part in folds]


st.title('Task 48: Manual Exhaustive Hyperparameter Search')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=48)

l2_values = [0.001, 0.01, 0.05]
learning_rates = [0.03, 0.06, 0.1]
iteration_values = [700, 1200]
folds = stratified_folds(y_development)
results = []
for l2 in l2_values:
    for learning_rate in learning_rates:
        for iterations in iteration_values:
            fold_scores = []
            for fold_index, valid_indices in enumerate(folds):
                valid_set = set(valid_indices.tolist())
                train_indices = np.array([i for i in range(len(y_development)) if i not in valid_set])
                X_train, X_valid = X_development[train_indices], X_development[valid_indices]
                y_train, y_valid = y_development[train_indices], y_development[valid_indices]
                means, scales = standardize_fit(X_train)
                X_train, X_valid = standardize(X_train, means, scales), standardize(X_valid, means, scales)
                model = LogisticRegressionScratch(learning_rate=learning_rate, iterations=iterations, l2=l2).fit(X_train, y_train)
                fold_scores.append(metrics(y_valid, model.predict(X_valid))['f1'])
            results.append({'l2': l2, 'learning_rate': learning_rate, 'iterations': iterations, 'mean_f1': float(np.mean(fold_scores)), 'std_f1': float(np.std(fold_scores, ddof=1))})

table = pd.DataFrame(results).sort_values(['mean_f1', 'std_f1'], ascending=[False, True]).reset_index(drop=True)
best = table.iloc[0]
model = LogisticRegressionScratch(learning_rate=float(best['learning_rate']), iterations=int(best['iterations']), l2=float(best['l2']))
means, scales = standardize_fit(X_development)
X_development_scaled = standardize(X_development, means, scales)
X_test_scaled = standardize(X_test, means, scales)
model.fit(X_development_scaled, y_development)
test_result = metrics(y_test, model.predict(X_test_scaled))

st.subheader('Exhaustive CV results')
st.dataframe(table.style.format({'l2': '{:.3g}', 'learning_rate': '{:.3f}', 'mean_f1': '{:.3f}', 'std_f1': '{:.3f}'}))
st.write(f'Selected configuration: l2={best["l2"]}, learning rate={best["learning_rate"]}, iterations={int(best["iterations"])}')
st.write('Untouched test metrics:', {key: round(value, 3) for key, value in test_result.items() if key in {'accuracy', 'precision', 'recall', 'f1'}})
st.caption(f'Manually evaluated {len(results)} parameter combinations across {len(folds)} stratified folds. No GridSearchCV or test-set tuning is used.')