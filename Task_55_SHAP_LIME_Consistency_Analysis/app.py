import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, exact_shapley_values, fit_lime_surrogate, lime_perturbations, load_data, standardize, standardize_fit, stratified_split


def local_weights(perturbed, instance, width):
    distance = np.sqrt(np.mean((perturbed - instance) ** 2, axis=1))
    return np.exp(-(distance ** 2) / width ** 2)


def weighted_r_squared(actual, predicted, weights):
    mean = np.average(actual, weights=weights)
    residual = np.sum(weights * (actual - predicted) ** 2)
    total = np.sum(weights * (actual - mean) ** 2)
    return 1 - residual / total if total > 1e-12 else 1.0


def average_ranks(values):
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def correlation(first, second):
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


st.title('Task 55: SHAP and LIME Consistency Analysis')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=55)
means, scales = standardize_fit(X_train)
X_train, X_test = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train, y_train)
background = X_train[np.linspace(0, len(X_train) - 1, min(10, len(X_train)), dtype=int)]
case_count = min(6, len(X_test))
case_indices = np.linspace(0, len(X_test) - 1, case_count, dtype=int)
shap_rows, lime_rows, consistency_rows = [], [], []
verification_scores = []

for case_order, case_index in enumerate(case_indices):
    instance = X_test[case_index]
    _, shap_values = exact_shapley_values(model.predict_proba, instance, background, batch_size=64)
    perturbations = lime_perturbations(instance, X_train, 1200, seed=5500 + case_order)
    targets = model.predict_proba(perturbations)
    weights = local_weights(perturbations, instance, 0.75)
    intercept, lime_values, _ = fit_lime_surrogate(perturbations, targets, weights, ridge=0.01)
    independent = lime_perturbations(instance, X_train, 400, seed=6500 + case_order)
    independent_targets = model.predict_proba(independent)
    independent_weights = local_weights(independent, instance, 0.75)
    independent_predictions = intercept + independent @ lime_values
    verification_scores.append(weighted_r_squared(independent_targets, independent_predictions, independent_weights))
    shap_rows.append(shap_values)
    lime_rows.append(lime_values)
    shap_rank, lime_rank = average_ranks(np.abs(shap_values)), average_ranks(np.abs(lime_values))
    top_count = min(5, len(features))
    shap_top = set(np.argsort(np.abs(shap_values))[-top_count:])
    lime_top = set(np.argsort(np.abs(lime_values))[-top_count:])
    consistency_rows.append({'Case index': int(case_index), 'Pearson (signed)': correlation(shap_values, lime_values), 'Spearman (magnitude ranks)': correlation(shap_rank, lime_rank), 'Sign agreement': float(np.mean(np.sign(shap_values) == np.sign(lime_values))), 'Top-5 overlap': len(shap_top & lime_top) / top_count, 'LIME independent weighted R2': verification_scores[-1]})

shap_array, lime_array = np.asarray(shap_rows), np.asarray(lime_rows)
consistency = pd.DataFrame(consistency_rows).set_index('Case index')
st.subheader('Per-case consistency and LIME verification')
st.dataframe(consistency.style.format('{:.3f}'))
st.write('Mean Pearson attribution correlation:', round(float(consistency['Pearson (signed)'].mean()), 3))
st.write('Mean Spearman magnitude-rank correlation:', round(float(consistency['Spearman (magnitude ranks)'].mean()), 3))
st.write('Mean top-5 overlap:', round(float(consistency['Top-5 overlap'].mean()), 3))

mean_shap, mean_lime = np.mean(np.abs(shap_array), axis=0), np.mean(np.abs(lime_array), axis=0)
variability = pd.DataFrame({'Feature': features, 'Mean |SHAP|': mean_shap, 'SD SHAP': np.std(shap_array, axis=0), 'Mean |LIME coefficient|': mean_lime, 'SD LIME': np.std(lime_array, axis=0)}).sort_values('Mean |SHAP|', ascending=False)
st.subheader('Feature-level contribution variability')
st.dataframe(variability.set_index('Feature').style.format('{:.4f}'))
fig, ax = plt.subplots(figsize=(8, 5))
order = np.argsort(mean_shap)
ax.barh(np.array(features)[order], mean_shap[order], alpha=0.8, label='Mean absolute SHAP')
ax.scatter(mean_lime[order], np.array(features)[order], color='#b34a36', label='Mean absolute LIME coefficient')
ax.set_xlabel('Mean absolute attribution magnitude'); ax.set_title('Cross-case feature impact'); ax.legend(); ax.grid(axis='x', alpha=0.2)
st.pyplot(fig)
st.markdown('**Limitations:** SHAP provides an additive attribution relative to a chosen background but exact computation is exponential and interventional replacement can break feature dependence. LIME is cheaper and locally interpretable, but coefficients depend on perturbation distribution, kernel width, and ridge penalty; independent fidelity scores test the surrogate, not the model’s correctness. Integer category codes are ordinal representations, so neither method should treat their numeric distance as intrinsically meaningful.')