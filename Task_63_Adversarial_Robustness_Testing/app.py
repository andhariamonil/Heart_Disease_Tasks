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


st.title('Task 63: Manual Adversarial Robustness Testing')
st.caption('Perturbations are applied in standardized feature space and clipped to the training 1st–99th percentile bounds; coded categories can still become fractional.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=63)
means, scales = standardize_fit(X_train)
X_train, X_test = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train, y_train)
epsilon = st.slider('L-infinity perturbation budget (training SD units)', 0.02, 0.75, 0.20, 0.02)
lower, upper = np.quantile(X_train, [0.01, 0.99], axis=0)
lower = np.minimum(lower, X_train.min(axis=0)); upper = np.maximum(upper, X_train.max(axis=0))

clean_probability = model.predict_proba(X_test)
clean_prediction = (clean_probability >= 0.5).astype(int)
clean_accuracy = metrics(y_test, clean_prediction)['accuracy']
probability = np.clip(clean_probability, 1e-8, 1 - 1e-8)
gradient = ((probability - y_test) * probability * (1 - probability))[:, None] * model.weights[None, :]
adversarial = np.clip(X_test + epsilon * np.sign(gradient), lower, upper)
adversarial_probability = model.predict_proba(adversarial)
adversarial_prediction = (adversarial_probability >= 0.5).astype(int)

rng = np.random.default_rng(630)
random_noise = rng.normal(0, epsilon, size=X_test.shape)
random_perturbed = np.clip(X_test + random_noise, lower, upper)
random_prediction = model.predict(random_perturbed)

# For each correctly classified case, move in the minimum L2 direction toward the linear boundary.
scores = X_test @ model.weights + model.bias
norm_squared = float(model.weights @ model.weights)
margin = np.maximum(np.abs(scores) + 1e-4, 1e-4)
boundary_worst_case = X_test - (np.sign(scores) * margin / max(norm_squared, 1e-12))[:, None] * model.weights[None, :]
boundary_worst_case = np.clip(boundary_worst_case, lower, upper)
boundary_prediction = model.predict(boundary_worst_case)

results = pd.DataFrame([
    {'Scenario': 'Clean', **metrics(y_test, clean_prediction)},
    {'Scenario': f'FGSM-style, epsilon={epsilon:.2f}', **metrics(y_test, adversarial_prediction)},
    {'Scenario': 'Random Gaussian noise', **metrics(y_test, random_prediction)},
    {'Scenario': 'Linear boundary-directed', **metrics(y_test, boundary_prediction)},
]).set_index('Scenario')
st.subheader('Performance under input perturbations')
st.dataframe(results[['accuracy', 'precision', 'recall', 'f1']].style.format('{:.3f}'))
st.write(f'Clean accuracy: {clean_accuracy:.3f}; adversarial accuracy: {metrics(y_test, adversarial_prediction)["accuracy"]:.3f}; changed predictions: {int(np.sum(adversarial_prediction != clean_prediction))} / {len(y_test)}')
case_index = st.slider('Inspect test case', 0, len(X_test) - 1, 0)
comparison = pd.DataFrame({'Feature': features, 'Clean standardized': X_test[case_index], 'Adversarial standardized': adversarial[case_index], 'Change': adversarial[case_index] - X_test[case_index]}).sort_values('Change', key=np.abs, ascending=False)
st.dataframe(comparison.set_index('Feature').style.format('{:.3f}'))
st.write(f'Case probability clean={clean_probability[case_index]:.3f}, perturbed={adversarial_probability[case_index]:.3f}, actual={y_test[case_index]}')
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(clean_probability, bins=12, alpha=0.6, label='Clean'); ax.hist(adversarial_probability, bins=12, alpha=0.6, label='Adversarial')
ax.set_xlabel('Predicted positive probability'); ax.set_ylabel('Count'); ax.set_title('Confidence shift under bounded perturbations'); ax.legend(); st.pyplot(fig)
st.caption('This is a robustness stress test, not a realistic patient-edit model. Independent feature changes ignore clinical constraints and categorical validity; never interpret generated rows as medical advice.')