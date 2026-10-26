import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, exact_shapley_values, load_data, metrics, standardize, standardize_fit, stratified_split


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


def rank_correlation(left, right):
    left, right = average_ranks(left), average_ranks(right)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


st.title('Task 56: Feature Importance Technical Report')
st.caption('Global rankings combine standardized model coefficients, held-out permutation impact, and mean absolute exact Shapley values.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=56)
X_train, X_validation, y_train, y_validation = stratified_split(X_development, y_development, 0.2, seed=57)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_validation = standardize(X_validation, means, scales)
X_test = standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train, y_train)
baseline_f1 = metrics(y_validation, model.predict(X_validation))['f1']

permutation_runs = 5
permutation_impacts = np.zeros((permutation_runs, len(features)))
rng = np.random.default_rng(560)
for repeat in range(permutation_runs):
    for feature_index in range(len(features)):
        shuffled = X_validation.copy()
        shuffled[:, feature_index] = rng.permutation(shuffled[:, feature_index])
        permuted_f1 = metrics(y_validation, model.predict(shuffled))['f1']
        permutation_impacts[repeat, feature_index] = baseline_f1 - permuted_f1

background = X_train[np.linspace(0, len(X_train) - 1, min(8, len(X_train)), dtype=int)]
explanation_indices = np.linspace(0, len(X_test) - 1, min(8, len(X_test)), dtype=int)
shapley_rows = []
for index in explanation_indices:
    _, values = exact_shapley_values(model.predict_proba, X_test[index], background, batch_size=64)
    shapley_rows.append(values)
shapley_rows = np.asarray(shapley_rows)

importance = pd.DataFrame({
    'Feature': features,
    'Coefficient |weight|': np.abs(model.weights),
    'Permutation F1 decrease': permutation_impacts.mean(axis=0),
    'Permutation SD': permutation_impacts.std(axis=0, ddof=1),
    'Mean |SHAP|': np.mean(np.abs(shapley_rows), axis=0),
    'SHAP SD across cases': np.std(shapley_rows, axis=0),
}).set_index('Feature')
for column in ['Coefficient |weight|', 'Permutation F1 decrease', 'Mean |SHAP|']:
    values = importance[column].to_numpy()
    scale = np.max(np.abs(values))
    importance[f'{column} normalized'] = values / scale if scale else values
importance['Mean rank'] = np.mean(np.vstack([average_ranks(-importance[f'{column} normalized'].to_numpy()) for column in ['Coefficient |weight|', 'Permutation F1 decrease', 'Mean |SHAP|']]), axis=0)
importance = importance.sort_values('Mean rank')

instance_index = st.slider('Local explanation case index', 0, len(explanation_indices) - 1, 0)
selected_test_index = explanation_indices[instance_index]
local_base, local_shap = exact_shapley_values(model.predict_proba, X_test[selected_test_index], background, batch_size=64)
local = pd.Series(local_shap, index=features).sort_values(key=np.abs, ascending=False)
st.subheader('Global importance rankings')
st.dataframe(importance.style.format('{:.4f}'))
st.subheader('Agreement between global methods')
method_rankings = [average_ranks(-importance[f'{column} normalized'].to_numpy()) for column in ['Coefficient |weight|', 'Permutation F1 decrease', 'Mean |SHAP|']]
rank_agreement = pd.DataFrame({
    'Coefficient vs permutation': [rank_correlation(method_rankings[0], method_rankings[1])],
    'Coefficient vs SHAP': [rank_correlation(method_rankings[0], method_rankings[2])],
    'Permutation vs SHAP': [rank_correlation(method_rankings[1], method_rankings[2])],
})
st.dataframe(rank_agreement.style.format('{:.3f}'))

fig, ax = plt.subplots(figsize=(8, 5))
for column, label in [('Coefficient |weight| normalized', 'Coefficient'), ('Permutation F1 decrease normalized', 'Permutation'), ('Mean |SHAP| normalized', 'Mean absolute SHAP')]:
    ax.plot(importance[column], importance.index, marker='o', label=label)
ax.set_xlabel('Normalized global importance'); ax.set_title('Global feature importance by method'); ax.legend(); ax.grid(axis='x', alpha=0.2)
st.pyplot(fig)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(local.index[::-1], local.values[::-1], color=np.where(local.values[::-1] >= 0, '#b34a36', '#147d64'))
ax.axvline(0, color='black', linewidth=0.8); ax.set_xlabel('Contribution to this case probability'); ax.set_title(f'Local Shapley values (base {local_base:.3f})')
st.pyplot(fig)

st.markdown('**Selection rationale:** prioritize features supported by more than one global method, then inspect local contributions and permutation uncertainty. Coefficients are conditional on the linear model and scaling; permutation importance is affected by correlated substitutes; Shapley values depend on the background and independent-feature intervention. None establishes causality or clinical relevance.')
st.download_button('Download importance table as CSV', importance.to_csv().encode('utf-8'), 'feature_importance_task56.csv', 'text/csv')