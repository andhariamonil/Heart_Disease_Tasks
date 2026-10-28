import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, standardize, standardize_fit, stratified_split


st.title('Task 64: Out-of-Distribution Stress Tests')
st.caption('Synthetic edge cases have no ground-truth labels; this app inspects model confidence and distance from observed training examples, not accuracy.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=64)
means, scales = standardize_fit(X_train)
X_train_scaled, X_test_scaled = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_scaled, y_train)

continuous = [feature for feature in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] if feature in features]
categorical_codes = [feature for feature in features if feature not in continuous]
training_features = pd.DataFrame(X_train, columns=features)
medians = training_features.median()
modes = training_features[categorical_codes].mode().iloc[0] if categorical_codes else pd.Series(dtype=float)
quantiles = training_features[continuous].quantile([0.01, 0.99])
cases = []
for feature in continuous:
    low, high = float(quantiles.loc[0.01, feature]), float(quantiles.loc[0.99, feature])
    for direction, value in [('low-tail', low), ('high-tail', high)]:
        row = medians.copy()
        for category in categorical_codes:
            row[category] = modes[category]
        row[feature] = value
        cases.append({'Scenario': f'{feature} {direction}', **row.to_dict()})

composite = medians.copy()
for category in categorical_codes:
    composite[category] = modes[category]
for feature in ['age', 'trestbps', 'chol', 'oldpeak']:
    if feature in continuous:
        composite[feature] = float(quantiles.loc[0.99, feature])
if 'thalach' in continuous:
    composite['thalach'] = float(quantiles.loc[0.01, 'thalach'])
cases.append({'Scenario': 'Combined marginal tails', **composite.to_dict()})
synthetic = pd.DataFrame(cases)
synthetic_X = synthetic[features].to_numpy(float)
synthetic_scaled = standardize(synthetic_X, means, scales)
probability = model.predict_proba(synthetic_scaled)
confidence = np.maximum(probability, 1 - probability)

tail_mask = np.zeros((len(synthetic), len(continuous)), dtype=bool)
for column_index, feature in enumerate(continuous):
    low, high = float(quantiles.loc[0.01, feature]), float(quantiles.loc[0.99, feature])
    tail_width = max((high - low) * 0.01, 1e-8)
    tail_mask[:, column_index] = (synthetic[feature] <= low + tail_width) | (synthetic[feature] >= high - tail_width)
tail_fraction = tail_mask.mean(axis=1)
nearest_distance = []
for row in synthetic_scaled:
    nearest_distance.append(float(np.sqrt(np.min(np.sum((X_train_scaled - row) ** 2, axis=1)))))
synthetic['Predicted probability'] = probability
synthetic['Confidence'] = confidence
synthetic['Continuous features near 1–99% tails'] = tail_fraction
synthetic['Nearest train distance (standardized L2)'] = nearest_distance
st.subheader('Synthetic edge-case predictions')
st.dataframe(synthetic[['Scenario', *features, 'Predicted probability', 'Confidence', 'Continuous features near 1–99% tails', 'Nearest train distance (standardized L2)']].set_index('Scenario').style.format('{:.3f}'))

in_distribution_confidence = np.maximum(model.predict_proba(X_test_scaled), 1 - model.predict_proba(X_test_scaled))
st.write(f'Median in-distribution test confidence: {np.median(in_distribution_confidence):.3f}; median synthetic-case confidence: {np.median(confidence):.3f}')
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].hist(in_distribution_confidence, bins=12, alpha=0.7, label='Held-out data'); axes[0].hist(confidence, bins=12, alpha=0.7, label='Synthetic edge cases')
axes[0].set_xlabel('Maximum class probability'); axes[0].set_ylabel('Count'); axes[0].set_title('Confidence comparison'); axes[0].legend()
axes[1].scatter(nearest_distance, confidence, c=tail_fraction, cmap='viridis', s=65)
axes[1].set_xlabel('Nearest training distance'); axes[1].set_ylabel('Model confidence'); axes[1].set_title('Confidence vs distance'); axes[1].grid(alpha=0.2)
st.pyplot(fig)
st.caption('Marginal-tail cases do not necessarily represent realistic joint patients; nearest-distance is a simple diagnostic, not a formal OOD detector. High confidence far from training support is a warning to reject/escalate, not evidence that the prediction is trustworthy.')