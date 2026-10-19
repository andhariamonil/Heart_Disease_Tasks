import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, GradientBoostingScratch, KernelSVMScratch, LogisticRegressionScratch, MLPClassifierScratch, RandomForestScratch, class_balanced_indices, load_data, metrics, standardize, standardize_fit, stratified_split


def fit_model(name, x, y, feature_names, seed):
    if name == 'Logistic regression':
        return LogisticRegressionScratch(learning_rate=0.06, iterations=1300, l2=0.02).fit(x, y)
    if name == 'Decision tree':
        return DecisionTreeScratch(max_depth=4, min_samples_split=8).fit(x, y, feature_names)
    if name == 'Random forest':
        return RandomForestScratch(n_trees=8, max_depth=4, seed=seed).fit(x, y, feature_names)
    if name == 'Gradient boosting':
        return GradientBoostingScratch(n_estimators=18, learning_rate=0.15).fit(x, y)
    if name == 'Kernel SVM':
        return KernelSVMScratch('rbf', gamma=1 / x.shape[1], epochs=10).fit(x, y)
    return MLPClassifierScratch((16, 8), epochs=75, batch_size=32, dropout_rate=0.15, batch_normalization=True, seed=seed).fit(x, y)


def probabilities(model, name, x):
    if name == 'Kernel SVM':
        return 1 / (1 + np.exp(-np.clip(model.decision_function(x), -30, 30)))
    return model.predict_proba(x)


st.title('Task 46: Cost-Sensitive Model Selection')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, feature_names, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[feature_names].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=46)
X_train, X_validation, y_train, y_validation = stratified_split(X_development, y_development, 0.25, seed=47)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_validation = standardize(X_validation, means, scales)
X_test = standardize(X_test, means, scales)

class_counts = pd.Series(y_train).value_counts().sort_index()
balanced_indices = class_balanced_indices(y_train, seed=48)
X_balanced, y_balanced = X_train[balanced_indices], y_train[balanced_indices]
st.write('Original training class counts:', class_counts.to_dict())
st.write('Custom inverse-frequency resampling counts:', pd.Series(y_balanced).value_counts().sort_index().to_dict())

precision_weight = st.slider('Precision weight', 0.0, 1.0, 0.2, 0.05)
recall_weight = st.slider('Recall weight', 0.0, 1.0, 0.3, 0.05)
f1_weight = st.slider('F1 weight', 0.0, 1.0, 0.5, 0.05)
weight_total = precision_weight + recall_weight + f1_weight
if weight_total == 0:
    st.error('At least one metric weight must be positive.')
    st.stop()
metric_weights = [precision_weight / weight_total, recall_weight / weight_total, f1_weight / weight_total]

names = ['Logistic regression', 'Decision tree', 'Random forest', 'Gradient boosting', 'Kernel SVM', 'Neural network']
thresholds = [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]
validation_rows, trained = [], {}
for index, name in enumerate(names):
    model = fit_model(name, X_balanced, y_balanced, feature_names, 46 + index)
    trained[name] = model
    validation_probabilities = probabilities(model, name, X_validation)
    best = None
    for threshold in thresholds:
        score = metrics(y_validation, (validation_probabilities >= threshold).astype(int))
        utility = metric_weights[0] * score['precision'] + metric_weights[1] * score['recall'] + metric_weights[2] * score['f1']
        candidate = (utility, threshold, score)
        if best is None or candidate[0] > best[0]:
            best = candidate
    validation_rows.append({'Model': name, 'Threshold': best[1], 'Validation precision': best[2]['precision'], 'Validation recall': best[2]['recall'], 'Validation F1': best[2]['f1'], 'Weighted utility': best[0]})

validation_table = pd.DataFrame(validation_rows).set_index('Model')
selected_name = validation_table['Weighted utility'].idxmax()
selected_threshold = float(validation_table.loc[selected_name, 'Threshold'])
test_probabilities = probabilities(trained[selected_name], selected_name, X_test)
test_result = metrics(y_test, (test_probabilities >= selected_threshold).astype(int))
st.subheader('Validation-based selection')
st.dataframe(validation_table.style.format('{:.3f}'))
st.write(f'Selected model: **{selected_name}** at threshold **{selected_threshold:.2f}** using normalized precision/recall/F1 weights {metric_weights}.')
st.subheader('Untouched test performance of selected model')
st.dataframe(pd.DataFrame([test_result])[['accuracy', 'precision', 'recall', 'f1']].style.format('{:.3f}'))
st.caption('Balanced resampling changes the training distribution; threshold and model are chosen on validation data only. This can improve minority-class sensitivity but may reduce precision. Test data is reserved for one final report, and class weighting should be reconsidered when the deployment prevalence or error costs differ.')