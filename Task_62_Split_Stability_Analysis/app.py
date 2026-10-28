import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, metrics, standardize, standardize_fit


def make_split(labels, test_fraction, seed, stratified):
    rng = np.random.default_rng(seed)
    if not stratified:
        order = rng.permutation(len(labels))
        cut = int(round(len(labels) * (1 - test_fraction)))
        return order[:cut], order[cut:]
    train_parts, test_parts = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cut = int(round(len(indices) * (1 - test_fraction)))
        train_parts.append(indices[:cut]); test_parts.append(indices[cut:])
    train = np.concatenate(train_parts); test = np.concatenate(test_parts)
    rng.shuffle(train); rng.shuffle(test)
    return train, test


st.title('Task 62: Performance Stability Across Data Splits')
st.write('Compare repeated random and stratified holdouts on unique records using one fixed scratch logistic model.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
split_count = st.slider('Number of repeat seeds', 10, 40, 20, 5)
records = []
for protocol, stratified in [('Random', False), ('Stratified', True)]:
    for seed in range(split_count):
        train_indices, test_indices = make_split(y, 0.2, 6200 + seed, stratified)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        means, scales = standardize_fit(X_train)
        X_train, X_test = standardize(X_train, means, scales), standardize(X_test, means, scales)
        model = LogisticRegressionScratch(learning_rate=0.06, iterations=1100, l2=0.02).fit(X_train, y_train)
        score = metrics(y_test, model.predict(X_test))
        records.append({'Protocol': protocol, 'Seed': seed, 'Accuracy': score['accuracy'], 'Precision': score['precision'], 'Recall': score['recall'], 'F1': score['f1'], 'Test positive rate': float(y_test.mean())})

results = pd.DataFrame(records)
summary = results.groupby('Protocol')[['Accuracy', 'Precision', 'Recall', 'F1', 'Test positive rate']].agg(['mean', 'std', 'min', 'max'])
st.subheader('Repeated-split summary')
st.dataframe(summary.style.format('{:.3f}'))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for axis, metric_name in zip(axes, ['Accuracy', 'F1']):
    random_values = results.loc[results['Protocol'] == 'Random', metric_name]
    stratified_values = results.loc[results['Protocol'] == 'Stratified', metric_name]
    axis.boxplot([random_values, stratified_values], tick_labels=['Random', 'Stratified'])
    axis.set_ylabel(metric_name); axis.set_title(f'{metric_name} across {split_count} seeds'); axis.grid(axis='y', alpha=0.2)
st.pyplot(fig)
st.subheader('Label distribution by split protocol')
st.dataframe(results.groupby('Protocol')['Test positive rate'].agg(['mean', 'std', 'min', 'max']).style.format('{:.3f}'))
st.caption('A stratified split stabilizes class composition, not all sampling variation. These repeated holdouts share records and are not independent confidence intervals; report them as sensitivity analysis, not formal significance tests.')