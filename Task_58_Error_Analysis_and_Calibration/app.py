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


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def fit_platt_scaler(probabilities, labels, iterations=1500, learning_rate=0.03, l2=0.01):
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    slope, intercept = 1.0, 0.0
    for _ in range(iterations):
        calibrated = sigmoid(slope * logits + intercept)
        error = calibrated - labels
        slope -= learning_rate * (np.mean(error * logits) + l2 * slope)
        intercept -= learning_rate * float(error.mean())
    return slope, intercept


def calibration_error(probabilities, labels, bin_count=8):
    edges = np.linspace(0, 1, bin_count + 1)
    total_error = 0.0
    rows = []
    for index in range(bin_count):
        mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1] if index < bin_count - 1 else probabilities <= edges[index + 1])
        if mask.any():
            confidence, observed = float(probabilities[mask].mean()), float(labels[mask].mean())
            total_error += mask.mean() * abs(confidence - observed)
            rows.append({'Probability bin': f'{edges[index]:.2f}–{edges[index + 1]:.2f}', 'Count': int(mask.sum()), 'Mean predicted': confidence, 'Observed positive rate': observed})
    return float(total_error), pd.DataFrame(rows)


st.title('Task 58: False-Positive/False-Negative Analysis')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=58)
X_train, X_validation, y_train, y_validation = stratified_split(X_development, y_development, 0.2, seed=59)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_validation = standardize(X_validation, means, scales)
X_test = standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train, y_train)

validation_probability = model.predict_proba(X_validation)
platt_slope, platt_intercept = fit_platt_scaler(validation_probability, y_validation)
threshold_options = np.linspace(0.25, 0.75, 21)
threshold_results = [(metrics(y_validation, (sigmoid(platt_slope * np.log(np.clip(validation_probability, 1e-6, 1 - 1e-6) / np.clip(1 - validation_probability, 1e-6, 1)) + platt_intercept) >= threshold).astype(int))['f1'], threshold) for threshold in threshold_options]
selected_threshold = float(max(threshold_results, key=lambda item: item[0])[1])

raw_probability = model.predict_proba(X_test)
calibrated_probability = sigmoid(platt_slope * np.log(np.clip(raw_probability, 1e-6, 1 - 1e-6) / np.clip(1 - raw_probability, 1e-6, 1)) + platt_intercept)
raw_prediction = (raw_probability >= 0.5).astype(int)
calibrated_prediction = (calibrated_probability >= selected_threshold).astype(int)
comparison = pd.DataFrame([
    {'Policy': 'Raw probability, threshold 0.50', **metrics(y_test, raw_prediction), 'ECE': calibration_error(raw_probability, y_test)[0]},
    {'Policy': f'Platt calibrated, threshold {selected_threshold:.2f}', **metrics(y_test, calibrated_prediction), 'ECE': calibration_error(calibrated_probability, y_test)[0]},
]).set_index('Policy')

st.subheader('Validation-calibrated policy comparison')
st.write(f'Manual Platt parameters: slope={platt_slope:.3f}, intercept={platt_intercept:.3f}; threshold selected on validation data: {selected_threshold:.2f}')
st.dataframe(comparison[['accuracy', 'precision', 'recall', 'f1', 'ECE']].style.format('{:.3f}'))

false_positive = (y_test == 0) & (calibrated_prediction == 1)
false_negative = (y_test == 1) & (calibrated_prediction == 0)
# Rebuild the same held-out row positions from the seeded split to identify concrete examples.
all_indices = np.arange(len(frame))
_, test_indices, _, _ = stratified_split(all_indices.reshape(-1, 1), y, 0.2, seed=58)
test_frame = frame.iloc[test_indices.ravel()].reset_index(drop=True)
false_positive_table = test_frame.loc[false_positive, features].copy()
false_positive_table['Predicted probability'] = calibrated_probability[false_positive]
false_negative_table = test_frame.loc[false_negative, features].copy()
false_negative_table['Predicted probability'] = calibrated_probability[false_negative]
st.subheader(f'False positives ({int(false_positive.sum())})')
st.dataframe(false_positive_table.head(10).style.format('{:.3f}'))
st.subheader(f'False negatives ({int(false_negative.sum())})')
st.dataframe(false_negative_table.head(10).style.format('{:.3f}'))

error_frame = test_frame.copy()
error_frame['Actual'] = y_test
error_frame['Predicted'] = calibrated_prediction
error_frame['Error type'] = np.where(false_positive, 'False positive', np.where(false_negative, 'False negative', 'Correct'))
error_frame['Age group'] = pd.cut(error_frame['age'], bins=[0, 45, 55, 65, 120], labels=['≤45', '46–55', '56–65', '>65'])
patterns = error_frame.groupby('Age group', observed=False).agg(rows=('Actual', 'size'), false_positives=('Error type', lambda series: int((series == 'False positive').sum())), false_negatives=('Error type', lambda series: int((series == 'False negative').sum())))
st.subheader('Error patterns by age band')
st.dataframe(patterns)

raw_ece, raw_bins = calibration_error(raw_probability, y_test)
calibrated_ece, calibrated_bins = calibration_error(calibrated_probability, y_test)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect calibration')
for label, probability in [('Raw', raw_probability), ('Platt calibrated', calibrated_probability)]:
    bins = calibration_error(probability, y_test)[1]
    ax.plot(bins['Mean predicted'], bins['Observed positive rate'], marker='o', label=label)
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed positive fraction'); ax.set_title('Test-set reliability diagram'); ax.legend(); ax.grid(alpha=0.2)
st.pyplot(fig)
st.caption('Platt scaling and threshold selection use validation data only; the test split is reserved for assessment. A changed threshold trades false positives against false negatives, and group error patterns require domain review rather than automatic causal interpretation.')