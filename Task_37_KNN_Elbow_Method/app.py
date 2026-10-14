import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import KNearestNeighborsScratch, load_data, standardize, standardize_fit


def stratified_folds(y, fold_count=5, seed=37):
    import numpy as np

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(fold_count)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for fold_index, part in enumerate(np.array_split(indices, fold_count)):
            folds[fold_index].extend(part.tolist())
    return [np.array(sorted(indices), dtype=int) for indices in folds]


def vote_neighbors(labels, distances):
    import numpy as np

    values, counts = np.unique(labels, return_counts=True)
    tied = values[counts == counts.max()]
    if len(tied) == 1:
        return int(tied[0])
    distance_sums = {int(label): float(distances[labels == label].sum()) for label in tied}
    smallest_sum = min(distance_sums.values())
    tied = [label for label in tied if np.isclose(distance_sums[label], smallest_sum)]
    return int(labels[0]) if labels[0] in tied else min(tied)


st.title('Task 37: Select k with the Elbow Method')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X = frame[features].to_numpy(dtype=float)
y = frame['target'].to_numpy(dtype=int)
candidate_ks = list(range(1, 26, 2))
folds = stratified_folds(y)
fold_errors = {k: [] for k in candidate_ks}

for fold_index, test_indices in enumerate(folds):
    test_set = set(test_indices.tolist())
    train_indices = [index for index in range(len(y)) if index not in test_set]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    means, scales = standardize_fit(X_train)
    X_train = standardize(X_train, means, scales)
    X_test = standardize(X_test, means, scales)
    model = KNearestNeighborsScratch(k=max(candidate_ks)).fit(X_train, y_train)
    errors = {k: 0 for k in candidate_ks}
    for row, actual in zip(X_test, y_test):
        indices, distances = model.tree.query(row, max(candidate_ks))
        labels = model.tree.y[indices]
        for k in candidate_ks:
            prediction = vote_neighbors(labels[:k], distances[:k])
            errors[k] += int(prediction != actual)
    for k in candidate_ks:
        fold_errors[k].append(errors[k] / len(test_indices))

mean_errors = {k: sum(errors) / len(errors) for k, errors in fold_errors.items()}
standard_errors = {}
for k, errors in fold_errors.items():
    mean = mean_errors[k]
    variance = sum((error - mean) ** 2 for error in errors) / (len(errors) - 1)
    standard_errors[k] = (variance ** 0.5) / (len(errors) ** 0.5)
best_k = min(candidate_ks, key=mean_errors.get)
threshold = mean_errors[best_k] + standard_errors[best_k]
selected_k = min(k for k in candidate_ks if mean_errors[k] <= threshold)

st.subheader('Stratified 5-fold validation error')
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(candidate_ks, [mean_errors[k] for k in candidate_ks], yerr=[standard_errors[k] for k in candidate_ks], marker='o', capsize=4, color='#147d64')
ax.axvline(selected_k, color='#b34a36', linestyle='--', label=f'One-SE choice: k={selected_k}')
ax.set_xlabel('k (odd values)')
ax.set_ylabel('Mean validation error rate')
ax.set_title('Manual k-NN Elbow Curve with Standard Error')
ax.set_xticks(candidate_ks)
ax.grid(True, alpha=0.25)
ax.legend()
st.pyplot(fig)
st.write(f'Minimum mean error at k={best_k}: {mean_errors[best_k]:.4f}')
st.write(f'One-standard-error selection: k={selected_k}; threshold={threshold:.4f}')
st.dataframe(pd.DataFrame({'k': candidate_ks, 'mean_error': [mean_errors[k] for k in candidate_ks], 'standard_error': [standard_errors[k] for k in candidate_ks]}).set_index('k'))
st.caption('Error rates are averaged over stratified folds; the smallest k within one standard error of the minimum is selected to favor a simpler model.')