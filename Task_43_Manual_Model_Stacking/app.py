import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, LogisticRegressionScratch, RandomForestScratch, load_data, metrics, standardize, standardize_fit, stratified_split


def make_model(kind, setting):
    if kind == 'logistic':
        return LogisticRegressionScratch(learning_rate=0.06, iterations=1200, l2=setting)
    if kind == 'tree':
        return DecisionTreeScratch(max_depth=setting, min_samples_split=8)
    return RandomForestScratch(n_trees=setting, max_depth=4, seed=43)


def fit_predict(kind, setting, x_train, y_train, x_eval, feature_names):
    model = make_model(kind, setting)
    if isinstance(model, (DecisionTreeScratch, RandomForestScratch)):
        model.fit(x_train, y_train, feature_names)
    else:
        model.fit(x_train, y_train)
    return model.predict_proba(x_eval)


def stratified_indices(labels, fold_count, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(fold_count)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for fold_index, part in enumerate(np.array_split(indices, fold_count)):
            folds[fold_index].extend(part.tolist())
    return [np.array(sorted(part), dtype=int) for part in folds]


st.title('Task 43: Manual Leakage-Safe Model Stacking')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, feature_names, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[feature_names].to_numpy(dtype=float), frame['target'].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=43)

# Keep one stratified fold entirely outside meta-model OOF training for base tuning.
tuning_folds = stratified_indices(y_train, fold_count=5, seed=44)
tuning_indices = tuning_folds[0]
stack_indices = np.concatenate(tuning_folds[1:])
X_tuning, y_tuning = X_train[tuning_indices], y_train[tuning_indices]
X_stack, y_stack = X_train[stack_indices], y_train[stack_indices]
tuning_train, tuning_validation, y_tune, y_validation = stratified_split(X_tuning, y_tuning, 0.8, seed=45)
means, scales = standardize_fit(tuning_train)
tuning_train = standardize(tuning_train, means, scales)
tuning_validation = standardize(tuning_validation, means, scales)
settings = {'logistic': [0.001, 0.03], 'tree': [3, 5], 'forest': [5, 9]}
selected = {}
tuning_rows = []
for kind, choices in settings.items():
    scored = []
    for choice in choices:
        probabilities = fit_predict(kind, choice, tuning_train, y_tune, tuning_validation, feature_names)
        predicted = (probabilities >= 0.5).astype(int)
        result = metrics(y_validation, predicted)
        scored.append((result['f1'], choice))
        tuning_rows.append({'Base model': kind, 'Setting': str(choice), 'Tuning F1': result['f1']})
    selected[kind] = max(scored, key=lambda pair: pair[0])[1]

# Generate out-of-fold base predictions, fitting each fold's scaler on fold training rows only.
stack_features = np.zeros((len(X_stack), len(settings)))
folds = stratified_indices(y_stack, fold_count=4, seed=46)
for validation_indices in folds:
    held_out = set(validation_indices.tolist())
    fold_train_indices = np.array([index for index in range(len(y_stack)) if index not in held_out], dtype=int)
    fold_train, fold_valid = X_stack[fold_train_indices], X_stack[validation_indices]
    fold_y = y_stack[fold_train_indices]
    fold_means, fold_scales = standardize_fit(fold_train)
    fold_train = standardize(fold_train, fold_means, fold_scales)
    fold_valid = standardize(fold_valid, fold_means, fold_scales)
    for column, kind in enumerate(settings):
        stack_features[validation_indices, column] = fit_predict(kind, selected[kind], fold_train, fold_y, fold_valid, feature_names)

# Fit the meta-model only on out-of-fold predictions, never in-sample base predictions.
meta_model = LogisticRegressionScratch(learning_rate=0.08, iterations=1000, l2=0.02).fit(stack_features, y_stack)

# Refit base learners on all outer training data; test is touched only for final scoring.
all_means, all_scales = standardize_fit(X_stack)
scaled_train = standardize(X_stack, all_means, all_scales)
scaled_test = standardize(X_test, all_means, all_scales)
test_meta_features = np.column_stack([fit_predict(kind, selected[kind], scaled_train, y_stack, scaled_test, feature_names) for kind in settings])
stack_predictions = meta_model.predict(test_meta_features)
test_scores = metrics(y_test, stack_predictions)

st.subheader('Base learner settings selected on tuning partition')
st.write(selected)
st.dataframe(pd.DataFrame(tuning_rows).set_index(['Base model', 'Setting']).style.format('{:.3f}'))
st.subheader('Held-out stacked-model metrics')
st.dataframe(pd.DataFrame([test_scores])[['accuracy', 'precision', 'recall', 'f1']].style.format('{:.3f}'))
st.write(f'Tuning rows: {len(y_tuning)}; meta-training rows: {len(stack_features)}; out-of-fold partitions: {len(folds)}; untouched test rows: {len(y_test)}')
st.caption('The tuning subset is disjoint from meta-training. Meta-model inputs are generated out-of-fold, and the test set is reserved for the final report. This prevents in-sample base predictions and tuning labels from leaking into the meta-model.')