import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, GradientBoostingScratch, LogisticRegressionScratch, RandomForestScratch, class_balanced_indices, load_data, manual_voting, metrics, standardize, standardize_fit, stratified_split


def train_models(x, y, feature_names, seed):
    models = {
        'Logistic regression': LogisticRegressionScratch(learning_rate=0.06, iterations=1300, l2=0.02),
        'Decision tree': DecisionTreeScratch(max_depth=4, min_samples_split=8),
        'Random forest': RandomForestScratch(n_trees=8, max_depth=4, seed=seed),
        'Gradient boosting': GradientBoostingScratch(n_estimators=18, learning_rate=0.15),
    }
    for model in models.values():
        if isinstance(model, DecisionTreeScratch):
            model.fit(x, y, feature_names)
        elif isinstance(model, RandomForestScratch):
            model.fit(x, y, feature_names)
        else:
            model.fit(x, y)
    return models


st.title('Task 44: Manual Hard and Soft Voting')
st.write('Model weights are learned from validation performance; the test split is used only for final comparison.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, feature_names, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[feature_names].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=44)
X_train, X_validation, y_train, y_validation = stratified_split(X_development, y_development, 0.25, seed=45)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_validation = standardize(X_validation, means, scales)
X_test = standardize(X_test, means, scales)

resample = st.checkbox('Use class-balanced resampling for base learners', value=False)
if resample:
    sampled = class_balanced_indices(y_train, seed=46)
    X_train, y_train = X_train[sampled], y_train[sampled]

models = train_models(X_train, y_train, feature_names, seed=44)
model_names = list(models)
validation_probabilities = pd.DataFrame({name: model.predict_proba(X_validation) for name, model in models.items()})
validation_metrics = {name: metrics(y_validation, (validation_probabilities[name].to_numpy() >= 0.5).astype(int)) for name in model_names}
weight_metric = st.selectbox('Validation metric used for dynamic weights', ['f1', 'recall', 'precision'], index=0)
raw_weights = pd.Series({name: max(validation_metrics[name][weight_metric], 1e-6) for name in model_names})
weights = raw_weights / raw_weights.sum()

test_probabilities = pd.DataFrame({name: model.predict_proba(X_test) for name, model in models.items()})
probability_matrix = test_probabilities.to_numpy().T
equal_weights = [1.0] * len(model_names)
predictions = {
    'Hard vote, equal weights': manual_voting(probability_matrix, equal_weights, 'hard')[0],
    'Soft vote, equal weights': manual_voting(probability_matrix, equal_weights, 'soft')[0],
    'Hard vote, validation-weighted': manual_voting(probability_matrix, weights.to_numpy(), 'hard')[0],
    'Soft vote, validation-weighted': manual_voting(probability_matrix, weights.to_numpy(), 'soft')[0],
}

st.subheader('Validation-derived model weights')
st.dataframe(pd.DataFrame({'Validation precision': [validation_metrics[name]['precision'] for name in model_names], 'Validation recall': [validation_metrics[name]['recall'] for name in model_names], 'Validation F1': [validation_metrics[name]['f1'] for name in model_names], 'Normalized weight': weights.to_numpy()}, index=model_names).style.format('{:.3f}'))
rows = []
for name in model_names:
    rows.append({'Prediction rule': name, **metrics(y_test, (test_probabilities[name].to_numpy() >= 0.5).astype(int))})
for name, predicted in predictions.items():
    rows.append({'Prediction rule': name, **metrics(y_test, predicted)})
st.subheader('Held-out base and ensemble performance')
st.dataframe(pd.DataFrame(rows).set_index('Prediction rule')[['accuracy', 'precision', 'recall', 'f1']].style.format('{:.3f}'))
st.caption('Hard voting combines thresholded class votes; soft voting averages probabilities. A hard-vote tie is resolved with the weighted mean probability. Validation metrics set the weights, avoiding test-set weight selection.')