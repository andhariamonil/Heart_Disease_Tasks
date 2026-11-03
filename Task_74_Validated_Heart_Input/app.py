import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, standardize, standardize_fit, stratified_split, validate_input_row


st.title('Task 74: Validated Heart-Feature Input')
st.caption('Input validation distinguishes hard errors from uncommon-but-plausible values. Suggestions are never applied silently.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, _, y_train, _ = stratified_split(X, y, 0.2, seed=74)
means, scales = standardize_fit(X_train)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(standardize(X_train, means, scales), y_train)

categorical = {'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'}
categorical_values = {name: sorted(frame[name].dropna().astype(int).unique().tolist()) for name in features if name in categorical}
hard_ranges = {'age': (18, 100), 'trestbps': (60, 260), 'chol': (80, 700), 'thalach': (40, 250), 'oldpeak': (-5, 15)}
training_frame = pd.DataFrame(X_train, columns=features)
learned_ranges = {name: (float(training_frame[name].quantile(0.01)), float(training_frame[name].quantile(0.99))) for name in features if name not in categorical}
defaults = {}
for name in features:
    if name in categorical_values:
        defaults[name] = str(int(training_frame[name].mode().iloc[0]))
    else:
        defaults[name] = f'{float(training_frame[name].median()):.3f}'

apply_suggestions = st.checkbox('Apply explicit suggestions to a temporary prediction copy', value=False)
with st.form('feature_input_form'):
    raw_values = {}
    st.write('Enter the feature values using the dataset’s numeric codes.')
    for feature in features:
        raw_values[feature] = st.text_input(feature, value=defaults[feature], help=f'Observed codes: {categorical_values[feature]}' if feature in categorical_values else 'Numeric measurement from the dataset schema.')
    submitted = st.form_submit_button('Validate and evaluate')

if submitted:
    parsed, issues, suggestions = validate_input_row(raw_values, features, categorical_values, hard_ranges, learned_ranges)
    issues_frame = pd.DataFrame(issues)
    if issues:
        st.subheader('Validation feedback')
        st.dataframe(issues_frame.set_index('feature'))
    if apply_suggestions and suggestions:
        values_for_prediction = raw_values.copy()
        correction_rows = []
        for _ in range(3):
            parsed, remaining_issues, next_suggestions = validate_input_row(values_for_prediction, features, categorical_values, hard_ranges, learned_ranges)
            if not next_suggestions:
                break
            corrections = {name: value for name, value in next_suggestions.items() if values_for_prediction.get(name) != value}
            if not corrections:
                break
            for name, value in corrections.items():
                correction_rows.append({'Feature': name, 'Entered value': str(values_for_prediction.get(name)), 'Suggested value': str(value)})
                values_for_prediction[name] = value
        parsed, remaining_issues, _ = validate_input_row(values_for_prediction, features, categorical_values, hard_ranges, learned_ranges)
        if correction_rows:
            st.write('Corrections applied to temporary prediction row:')
            st.dataframe(pd.DataFrame(correction_rows).set_index('Feature'))
        st.info('Suggestions were applied only to a temporary row; entered text and source data were not changed.')
        hard_errors = [issue for issue in remaining_issues if issue['level'] == 'error']
        if hard_errors or any(name not in parsed for name in features):
            st.error('The suggested row still has blocking input issues; no prediction was produced.')
            st.stop()
    elif any(issue['level'] == 'error' for issue in issues):
        st.error('Fix blocking errors or explicitly apply available suggestions before prediction.')
        st.stop()
    if not parsed:
        st.error('No valid feature values were available for evaluation.')
        st.stop()
    row = np.array([[parsed[name] for name in features]], dtype=float)
    probability = float(model.predict_proba(standardize(row, means, scales))[0])
    st.subheader('Model output')
    st.write(f'Positive-class probability: {probability:.3f}')
    if any(issue['level'] == 'warning' for issue in issues) and not apply_suggestions:
        st.warning('This row is valid under configured checks but differs from the training distribution. Treat the output cautiously.')
    else:
        st.success('Input passed configured checks. This educational output is not a diagnosis.')

st.caption('Codes and hard ranges are based on the supplied data schema, not a clinical guideline. Confirm units, coding, and deployment requirements before using real inputs.')