import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, LogisticRegressionScratch, RandomForestScratch, load_data, metrics, standardize, standardize_fit, stratified_split


def stratified_folds(labels, count=3, seed=50):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(count)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for fold_index, part in enumerate(np.array_split(indices, count)):
            folds[fold_index].extend(part.tolist())
    return [np.array(sorted(part), dtype=int) for part in folds]


def fit_classifier(name, setting, x, y, feature_names, seed):
    if name == 'Logistic regression':
        model = LogisticRegressionScratch(**setting)
        return model.fit(x, y)
    if name == 'Decision tree':
        model = DecisionTreeScratch(max_depth=setting['depth'], min_samples_split=8)
        return model.fit(x, y, feature_names)
    model = RandomForestScratch(n_trees=setting['trees'], max_depth=setting['depth'], seed=seed)
    return model.fit(x, y, feature_names)


def tracked_logistic(setting, x, y, x_validation, y_validation):
    model = LogisticRegressionScratch(**setting)
    model.weights = np.zeros(x.shape[1], dtype=float)
    model.bias = 0.0
    train_losses, validation_losses = [], []
    for iteration in range(model.iterations):
        train_probability = model._sigmoid(x @ model.weights + model.bias)
        error = train_probability - y
        model.weights -= model.learning_rate * (x.T @ error / len(y) + model.l2 * model.weights)
        model.bias -= model.learning_rate * float(error.mean())
        if (iteration + 1) % 20 == 0 or iteration == model.iterations - 1:
            train_probability = np.clip(model.predict_proba(x), 1e-12, 1 - 1e-12)
            valid_probability = np.clip(model.predict_proba(x_validation), 1e-12, 1 - 1e-12)
            train_losses.append(float(-np.mean(y * np.log(train_probability) + (1 - y) * np.log(1 - train_probability))))
            validation_losses.append(float(-np.mean(y_validation * np.log(valid_probability) + (1 - y_validation) * np.log(1 - valid_probability))))
    return train_losses, validation_losses


st.title('Task 50: Before/After Tuning and Generalization')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, feature_names, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[feature_names].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=50)

search_space = {
    'Logistic regression': [
        {'learning_rate': 0.06, 'iterations': 900, 'l2': 0.02},
        {'learning_rate': 0.03, 'iterations': 700, 'l2': 0.001},
        {'learning_rate': 0.06, 'iterations': 1200, 'l2': 0.01},
    ],
    'Decision tree': [{'depth': 2}, {'depth': 4}, {'depth': 6}],
    'Random forest': [{'trees': 4, 'depth': 3}, {'trees': 6, 'depth': 4}, {'trees': 10, 'depth': 4}],
}
baseline_settings = {name: candidates[0] for name, candidates in search_space.items()}
folds = stratified_folds(y_development)
cross_validation_rows = []
best_settings = {}
for name, candidates in search_space.items():
    family_results = []
    for setting in candidates:
        f1_scores, accuracy_gaps = [], []
        for fold_index, valid_indices in enumerate(folds):
            valid_set = set(valid_indices.tolist())
            train_indices = np.array([i for i in range(len(y_development)) if i not in valid_set])
            X_train, X_valid = X_development[train_indices], X_development[valid_indices]
            y_train, y_valid = y_development[train_indices], y_development[valid_indices]
            means, scales = standardize_fit(X_train)
            X_train, X_valid = standardize(X_train, means, scales), standardize(X_valid, means, scales)
            model = fit_classifier(name, setting, X_train, y_train, feature_names, 500 + fold_index)
            train_accuracy = metrics(y_train, model.predict(X_train))['accuracy']
            valid_result = metrics(y_valid, model.predict(X_valid))
            f1_scores.append(valid_result['f1'])
            accuracy_gaps.append(train_accuracy - valid_result['accuracy'])
        result = {'Model': name, 'Setting': str(setting), 'Mean CV F1': float(np.mean(f1_scores)), 'CV F1 SD': float(np.std(f1_scores, ddof=1)), 'Train-validation accuracy gap': float(np.mean(accuracy_gaps))}
        family_results.append((float(np.mean(f1_scores)), -float(np.std(f1_scores, ddof=1)), setting))
        cross_validation_rows.append(result)
    best_settings[name] = max(family_results, key=lambda row: (row[0], row[1]))[2]

test_rows = []
for name in search_space:
    for phase, setting in [('Before tuning', baseline_settings[name]), ('After tuning', best_settings[name])]:
        means, scales = standardize_fit(X_development)
        X_train, X_holdout = standardize(X_development, means, scales), standardize(X_test, means, scales)
        model = fit_classifier(name, setting, X_train, y_development, feature_names, 510)
        test_rows.append({'Model': name, 'Phase': phase, 'Setting': str(setting), **metrics(y_test, model.predict(X_holdout))})

table = pd.DataFrame(test_rows)
st.subheader('Cross-validation search and train/validation gap')
st.dataframe(pd.DataFrame(cross_validation_rows).set_index(['Model', 'Setting']).style.format('{:.3f}'))
st.subheader('Held-out performance before and after tuning')
st.dataframe(table.set_index(['Model', 'Phase'])[['accuracy', 'precision', 'recall', 'f1']].style.format('{:.3f}'))
fig, ax = plt.subplots(figsize=(8, 4))
for phase, color in [('Before tuning', '#b34a36'), ('After tuning', '#147d64')]:
    selection = table[table['Phase'] == phase]
    ax.plot(selection['Model'], selection['f1'], marker='o', label=phase, color=color)
ax.set_ylabel('Held-out F1'); ax.set_title('Generalization performance before/after tuning'); ax.legend(); ax.grid(alpha=0.25)
st.pyplot(fig)

# Convergence diagnostic: compare baseline and tuned logistic loss on a dev-only split.
X_fit, X_curve, y_fit, y_curve = stratified_split(X_development, y_development, 0.2, seed=51)
means, scales = standardize_fit(X_fit)
X_fit, X_curve = standardize(X_fit, means, scales), standardize(X_curve, means, scales)
baseline_train, baseline_valid = tracked_logistic(baseline_settings['Logistic regression'], X_fit, y_fit, X_curve, y_curve)
tuned_train, tuned_valid = tracked_logistic(best_settings['Logistic regression'], X_fit, y_fit, X_curve, y_curve)
fig, ax = plt.subplots(figsize=(8, 4))
baseline_steps = np.arange(20, len(baseline_train) * 20 + 1, 20)
tuned_steps = np.arange(20, len(tuned_train) * 20 + 1, 20)
ax.plot(baseline_steps, baseline_train, label='Baseline train', color='#147d64')
ax.plot(baseline_steps, baseline_valid, label='Baseline validation', color='#147d64', linestyle='--')
ax.plot(tuned_steps, tuned_train, label='Tuned train', color='#b34a36')
ax.plot(tuned_steps, tuned_valid, label='Tuned validation', color='#b34a36', linestyle='--')
ax.set_xlabel('Gradient updates'); ax.set_ylabel('Binary cross-entropy'); ax.set_title('Convergence on development-only holdout'); ax.legend(); ax.grid(alpha=0.25)
st.pyplot(fig)
st.caption('Tuning gains are judged on identical held-out rows. CV fold spread and train-validation gaps reveal variance and overfit risk; the convergence plot checks optimization behavior, not test-set model selection.')