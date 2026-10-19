import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, GradientBoostingScratch, KNearestNeighborsScratch, LogisticRegressionScratch, MLPClassifierScratch, RandomForestScratch, SVMLinearScratch, load_data, metrics, standardize, standardize_fit


def stratified_folds(labels, count=5, seed=45):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(count)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for fold_index, part in enumerate(np.array_split(indices, count)):
            folds[fold_index].extend(part.tolist())
    return [np.array(sorted(part), dtype=int) for part in folds]


def fit_model(name, x, y, feature_names, seed):
    if name == 'Logistic regression':
        return LogisticRegressionScratch(learning_rate=0.06, iterations=1200, l2=0.02).fit(x, y)
    if name == 'Decision tree':
        return DecisionTreeScratch(max_depth=4, min_samples_split=8).fit(x, y, feature_names)
    if name == 'Random forest':
        return RandomForestScratch(n_trees=6, max_depth=4, seed=seed).fit(x, y, feature_names)
    if name == 'Gradient boosting':
        return GradientBoostingScratch(n_estimators=16, learning_rate=0.15).fit(x, y)
    if name == 'Linear SVM':
        return SVMLinearScratch(learning_rate=0.01, epochs=700).fit(x, y)
    if name == 'k-NN':
        return KNearestNeighborsScratch(k=7).fit(x, y)
    return MLPClassifierScratch((16, 8), epochs=55, batch_size=32, dropout_rate=0.1, batch_normalization=True, seed=seed).fit(x, y)


def predict_labels(model, name, x):
    if name == 'Linear SVM':
        return (model.decision_function(x) >= 0).astype(int)
    return model.predict(x)


def f1_value(labels, predictions):
    return metrics(labels, predictions)['f1']


def paired_bootstrap_f1(labels, first, second, repeats=1500, seed=45):
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    differences = np.empty(repeats)
    for repeat in range(repeats):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in class_indices])
        differences[repeat] = f1_value(labels[sample], first[sample]) - f1_value(labels[sample], second[sample])
    return np.quantile(differences, [0.025, 0.975])


st.title('Task 45: Model Performance and Statistical Comparison')
st.write('Five-fold out-of-fold predictions compare model families on identical held-out rows.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, feature_names, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[feature_names].to_numpy(float), frame['target'].to_numpy(int)
names = ['Logistic regression', 'Decision tree', 'Random forest', 'Gradient boosting', 'Linear SVM', 'k-NN', 'Neural network']
oof_predictions = {name: np.zeros(len(y), dtype=int) for name in names}
fold_metrics = {name: [] for name in names}
fold_accuracy_gaps = {name: [] for name in names}
folds = stratified_folds(y)

for fold_index, valid_indices in enumerate(folds):
    valid_set = set(valid_indices.tolist())
    train_indices = np.array([index for index in range(len(y)) if index not in valid_set])
    X_train, X_valid = X[train_indices], X[valid_indices]
    y_train, y_valid = y[train_indices], y[valid_indices]
    means, scales = standardize_fit(X_train)
    X_train, X_valid = standardize(X_train, means, scales), standardize(X_valid, means, scales)
    for model_index, name in enumerate(names):
        model = fit_model(name, X_train, y_train, feature_names, 45 + fold_index * 10 + model_index)
        predictions = predict_labels(model, name, X_valid)
        oof_predictions[name][valid_indices] = predictions
        validation_result = metrics(y_valid, predictions)
        train_result = metrics(y_train, predict_labels(model, name, X_train))
        fold_metrics[name].append(validation_result)
        fold_accuracy_gaps[name].append(train_result['accuracy'] - validation_result['accuracy'])

rows = []
for name in names:
    result = metrics(y, oof_predictions[name])
    rows.append({'Model': name, **result, 'Fold F1 SD': float(np.std([item['f1'] for item in fold_metrics[name]], ddof=1)), 'Mean train-validation accuracy gap': float(np.mean(fold_accuracy_gaps[name]))})
performance = pd.DataFrame(rows).set_index('Model')
st.subheader('Out-of-fold metrics and variability')
st.dataframe(performance[['accuracy', 'precision', 'recall', 'f1', 'Fold F1 SD', 'Mean train-validation accuracy gap']].style.format('{:.3f}'))
st.caption('Fold F1 standard deviation summarizes split-to-split instability; train-validation accuracy gaps provide a rough overfit signal. Lower-complexity models can have higher bias but lower variance, while flexible trees/networks can fit interactions but may be less stable.')

best_name = performance['f1'].idxmax()
st.subheader(f'Paired bootstrap F1 comparisons vs. {best_name}')
comparison_rows = []
for name in names:
    if name == best_name:
        continue
    low, high = paired_bootstrap_f1(y, oof_predictions[best_name], oof_predictions[name], seed=45 + names.index(name))
    comparison_rows.append({'Comparison': f'{best_name} minus {name}', '95% paired bootstrap CI': f'[{low:.3f}, {high:.3f}]', 'Evidence of difference': 'CI excludes zero' if low > 0 or high < 0 else 'Inconclusive'})
st.dataframe(pd.DataFrame(comparison_rows).set_index('Comparison'))
st.caption('These intervals are exploratory because the top model was selected from the same cross-validation results. A final confirmatory study needs a new external test cohort; overlapping intervals do not prove equivalence.')