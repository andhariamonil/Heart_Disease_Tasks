import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, metrics, standardize, standardize_fit, stratified_split


def stratified_folds(labels, count=3, seed=49):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(count)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for fold_index, part in enumerate(np.array_split(indices, count)):
            folds[fold_index].extend(part.tolist())
    return [np.array(sorted(part), dtype=int) for part in folds]


def gaussian_process_prediction(observed_x, observed_y, query_x, length_scale=0.8, noise=1e-5):
    distances = np.sum((observed_x[:, None, :] - observed_x[None, :, :]) ** 2, axis=2)
    covariance = np.exp(-0.5 * distances / length_scale ** 2) + np.eye(len(observed_y)) * noise
    cross_distances = np.sum((query_x[:, None, :] - observed_x[None, :, :]) ** 2, axis=2)
    cross_covariance = np.exp(-0.5 * cross_distances / length_scale ** 2)
    try:
        alpha = np.linalg.solve(covariance, observed_y)
        mean = cross_covariance @ alpha
        solved_cross = np.linalg.solve(covariance, cross_covariance.T)
    except np.linalg.LinAlgError:
        covariance += np.eye(len(observed_y)) * 1e-3
        alpha = np.linalg.solve(covariance, observed_y)
        mean = cross_covariance @ alpha
        solved_cross = np.linalg.solve(covariance, cross_covariance.T)
    variance = np.maximum(1.0 - np.sum(cross_covariance * solved_cross.T, axis=1), 1e-9)
    return mean, np.sqrt(variance)


def expected_improvement(mean, standard_deviation, best_score, exploration=0.01):
    z = (mean - best_score - exploration) / standard_deviation
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / np.sqrt(2)))
    density = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return (mean - best_score - exploration) * cdf + standard_deviation * density


st.title('Task 49: Manual Bayesian Hyperparameter Optimization')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=49)
folds = stratified_folds(y_development)

# Parameters are represented in normalized coordinates and decoded to model settings.
def decode(point):
    l2 = 10 ** (-3 + point[0] * 2)
    learning_rate = 0.02 + point[1] * 0.10
    iterations = int(round((600 + point[2] * 1200) / 100) * 100)
    return {'l2': l2, 'learning_rate': learning_rate, 'iterations': iterations}


def objective(point):
    settings = decode(point)
    fold_scores = []
    for valid_indices in folds:
        valid_set = set(valid_indices.tolist())
        train_indices = np.array([i for i in range(len(y_development)) if i not in valid_set])
        X_train, X_valid = X_development[train_indices], X_development[valid_indices]
        y_train, y_valid = y_development[train_indices], y_development[valid_indices]
        means, scales = standardize_fit(X_train)
        X_train, X_valid = standardize(X_train, means, scales), standardize(X_valid, means, scales)
        model = LogisticRegressionScratch(**settings).fit(X_train, y_train)
        fold_scores.append(metrics(y_valid, model.predict(X_valid))['f1'])
    return float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1))


rng = np.random.default_rng(490)
initial_points = [np.array([0.15, 0.35, 0.35]), np.array([0.75, 0.65, 0.75]), np.array([0.45, 0.5, 0.5])]
observed_points, observed_scores, observed_stds = [], [], []
history = []
for point in initial_points:
    score, fold_sd = objective(point)
    observed_points.append(point); observed_scores.append(score); observed_stds.append(fold_sd)
    history.append({'evaluation': len(history) + 1, **decode(point), 'mean_cv_f1': score, 'fold_f1_sd': fold_sd})

for _ in range(5):
    candidate_points = rng.random((80, 3))
    means, deviations = gaussian_process_prediction(np.vstack(observed_points), np.array(observed_scores), candidate_points)
    acquisition = expected_improvement(means, deviations, max(observed_scores))
    next_point = candidate_points[int(np.argmax(acquisition))]
    score, fold_sd = objective(next_point)
    observed_points.append(next_point); observed_scores.append(score); observed_stds.append(fold_sd)
    history.append({'evaluation': len(history) + 1, **decode(next_point), 'mean_cv_f1': score, 'fold_f1_sd': fold_sd})

history_table = pd.DataFrame(history)
best_index = int(np.argmax(observed_scores))
best_parameters = decode(observed_points[best_index])
means, scales = standardize_fit(X_development)
X_development_scaled = standardize(X_development, means, scales)
X_test_scaled = standardize(X_test, means, scales)
final_model = LogisticRegressionScratch(**best_parameters).fit(X_development_scaled, y_development)
test_result = metrics(y_test, final_model.predict(X_test_scaled))

st.subheader('Gaussian-process search history')
st.dataframe(history_table.style.format({'l2': '{:.4g}', 'learning_rate': '{:.3f}', 'mean_cv_f1': '{:.3f}', 'fold_f1_sd': '{:.3f}'}))
st.write('Selected settings:', best_parameters)
st.write('Untouched test metrics:', {key: round(value, 3) for key, value in test_result.items() if key in {'accuracy', 'precision', 'recall', 'f1'}})
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history_table['evaluation'], history_table['mean_cv_f1'], marker='o', color='#147d64')
ax.set_xlabel('Objective evaluation'); ax.set_ylabel('Mean cross-validation F1'); ax.set_title('Manual Gaussian-process expected-improvement search'); ax.grid(alpha=0.25)
st.pyplot(fig)
st.caption('The RBF Gaussian-process posterior and expected-improvement acquisition are computed directly with NumPy. The test split is excluded from optimization. This educational search is not a substitute for nested validation or external deployment validation.')