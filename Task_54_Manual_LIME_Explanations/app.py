import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, fit_lime_surrogate, lime_perturbations, load_data, standardize, standardize_fit, stratified_split


def local_weights(perturbed, instance, kernel_width):
    distance = np.sqrt(np.mean((perturbed - instance) ** 2, axis=1))
    return np.exp(-(distance ** 2) / (kernel_width ** 2))


def weighted_r_squared(actual, predicted, weights):
    mean = np.average(actual, weights=weights)
    residual = np.sum(weights * (actual - predicted) ** 2)
    total = np.sum(weights * (actual - mean) ** 2)
    return 1 - residual / total if total > 1e-12 else 1.0


st.title('Task 54: Manual LIME Explanations')
st.caption('Perturbations are sampled feature-wise from observed training values; a locally weighted ridge surrogate approximates one model prediction.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=54)
means, scales = standardize_fit(X_train)
X_train, X_test = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train, y_train)
case_index = st.slider('Held-out case index', 0, len(X_test) - 1, 0)
instance = X_test[case_index]
samples = st.slider('Perturbation sample count', 500, 3000, 1500, 250)
kernel_width = st.slider('Local kernel width', 0.25, 2.0, 0.75, 0.05)

fit_samples = lime_perturbations(instance, X_train, samples, seed=540 + case_index)
fit_targets = model.predict_proba(fit_samples)
fit_weights = local_weights(fit_samples, instance, kernel_width)
intercept, coefficients, training_r2 = fit_lime_surrogate(fit_samples, fit_targets, fit_weights, ridge=0.01)

# Verify fidelity on independently generated perturbations not used to fit the surrogate.
verification_samples = lime_perturbations(instance, X_train, samples // 2, seed=1540 + case_index)
verification_targets = model.predict_proba(verification_samples)
verification_weights = local_weights(verification_samples, instance, kernel_width)
verification_prediction = intercept + verification_samples @ coefficients
verification_r2 = weighted_r_squared(verification_targets, verification_prediction, verification_weights)
model_probability = float(model.predict_proba(instance[None, :])[0])
surrogate_probability = float(intercept + instance @ coefficients)

st.write(f'Model probability: {model_probability:.3f}; local surrogate at case: {surrogate_probability:.3f}')
st.write(f'Weighted R-squared: fit perturbations {training_r2:.3f}; independent verification perturbations {verification_r2:.3f}')
importance = pd.DataFrame({'Feature': features, 'Standardized value': instance, 'Local coefficient': coefficients, 'Contribution at case': coefficients * instance}).sort_values('Contribution at case', key=np.abs, ascending=False)
st.subheader('Local surrogate explanation')
st.dataframe(importance.set_index('Feature').style.format('{:.3f}'))
fig, ax = plt.subplots(figsize=(8, 5))
ordered = importance.sort_values('Contribution at case')
ax.barh(ordered['Feature'], ordered['Contribution at case'], color=np.where(ordered['Contribution at case'] >= 0, '#b34a36', '#147d64'))
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Local linear contribution')
ax.set_title('Manual LIME-style explanation')
st.pyplot(fig)
st.caption('Independent weighted R-squared tests local surrogate fidelity, not correctness of the underlying model. Feature-wise perturbations can break correlations and produce unrealistic categorical combinations; results depend on the background, kernel width, and sample seed.')