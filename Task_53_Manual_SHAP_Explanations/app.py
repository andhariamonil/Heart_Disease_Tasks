import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, exact_shapley_values, load_data, standardize, standardize_fit, stratified_split


st.title('Task 53: Manual SHAP-Style Shapley Explanations')
st.caption('Exact interventional Shapley values are computed by coalition enumeration; runtime grows exponentially with feature count.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=53)
means, scales = standardize_fit(X_train)
X_train_scaled, X_test_scaled = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_scaled, y_train)
background_count = min(12, len(X_train_scaled))
background = X_train_scaled[np.linspace(0, len(X_train_scaled) - 1, background_count, dtype=int)]
instance_index = st.slider('Held-out case index', 0, len(X_test_scaled) - 1, 0)
instance = X_test_scaled[instance_index]
baseline, shap_values = exact_shapley_values(model.predict_proba, instance, background, batch_size=64)
prediction = float(model.predict_proba(instance[None, :])[0])

st.write(f'Predicted positive probability: {prediction:.3f}')
st.write(f'Background expected probability: {baseline:.3f}')
st.write(f'Additivity check: baseline + contributions = {baseline + shap_values.sum():.6f}')
contributions = pd.DataFrame({'Feature': features, 'Standardized value': instance, 'Shapley value': shap_values}).sort_values('Shapley value', key=np.abs, ascending=False)
st.subheader('Individual feature contributions')
st.dataframe(contributions.set_index('Feature').style.format('{:.3f}'))
fig, ax = plt.subplots(figsize=(8, 5))
ordered = contributions.sort_values('Shapley value')
ax.barh(ordered['Feature'], ordered['Shapley value'], color=np.where(ordered['Shapley value'] >= 0, '#b34a36', '#147d64'))
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Contribution to predicted probability relative to background')
ax.set_title('Exact Shapley values for selected held-out case')
st.pyplot(fig)

sample_count = min(5, len(X_test_scaled))
sample_indices = np.linspace(0, len(X_test_scaled) - 1, sample_count, dtype=int)
sample_values = []
for index in sample_indices:
    _, values = exact_shapley_values(model.predict_proba, X_test_scaled[index], background, batch_size=64)
    sample_values.append(values)
variation = pd.DataFrame({'Feature': features, 'Mean absolute attribution': np.mean(np.abs(sample_values), axis=0), 'Attribution SD across cases': np.std(sample_values, axis=0)}).sort_values('Mean absolute attribution', ascending=False)
st.subheader('Contribution variability across held-out cases')
st.dataframe(variation.set_index('Feature').style.format('{:.4f}'))
st.caption('These are interventional attributions relative to a small empirical background. Independent feature replacement can create implausible combinations, especially among correlated and ordinal-coded categorical variables; attributions are not causal effects.')