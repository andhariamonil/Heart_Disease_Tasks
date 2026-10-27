import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, metrics, standardize, standardize_fit, stratified_split


def reweighing_weights(labels, groups):
    weights = np.zeros(len(labels), dtype=float)
    for group in np.unique(groups):
        for label in np.unique(labels):
            cell = (groups == group) & (labels == label)
            if cell.any():
                group_probability = np.mean(groups == group)
                label_probability = np.mean(labels == label)
                joint_probability = np.mean(cell)
                weights[cell] = group_probability * label_probability / joint_probability
    return weights / weights.mean()


def fairness_scores(actual, predicted, groups):
    by_group = {}
    for group in sorted(np.unique(groups)):
        selected = groups == group
        truth, output = actual[selected], predicted[selected]
        tp = int(((truth == 1) & (output == 1)).sum())
        fp = int(((truth == 0) & (output == 1)).sum())
        tn = int(((truth == 0) & (output == 0)).sum())
        fn = int(((truth == 1) & (output == 0)).sum())
        by_group[group] = {'n': int(selected.sum()), 'selection_rate': float(output.mean()), 'TPR': tp / (tp + fn) if tp + fn else np.nan, 'FPR': fp / (fp + tn) if fp + tn else np.nan}
    return by_group


def differences(by_group, measure):
    values = [row[measure] for row in by_group.values() if pd.notna(row[measure])]
    return max(values) - min(values) if len(values) >= 2 else np.nan


st.title('Task 61: Bias-Mitigation Strategy Comparison')
st.caption('Compare two manual training interventions: group/outcome reweighting and dropping the sensitive field. Neither guarantees fairness or removes proxy information.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y, group = frame[features].to_numpy(float), frame['target'].to_numpy(int), frame['sex'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=61)
_, group_test, _, _ = stratified_split(group.reshape(-1, 1), y, 0.2, seed=61)
group_test = group_test.ravel()
means, scales = standardize_fit(X_train)
X_train_scaled = standardize(X_train, means, scales)
X_test_scaled = standardize(X_test, means, scales)
threshold = st.slider('Common decision threshold', 0.1, 0.9, 0.5, 0.05)

baseline = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_scaled, y_train)
weights = reweighing_weights(y_train, X_train[:, features.index('sex')].astype(int))
reweighted = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_scaled, y_train, sample_weight=weights)
sex_index = features.index('sex')
retained = [index for index in range(len(features)) if index != sex_index]
drop_sensitive = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_scaled[:, retained], y_train)

models = {
    'Baseline': baseline.predict(X_test_scaled, threshold),
    'Reweighted': reweighted.predict(X_test_scaled, threshold),
    'Sensitive feature removed': drop_sensitive.predict(X_test_scaled[:, retained], threshold),
}
rows = []
group_tables = {}
for name, predicted in models.items():
    result = metrics(y_test, predicted)
    groups = fairness_scores(y_test, predicted, group_test)
    group_tables[name] = groups
    selection_rates = [row['selection_rate'] for row in groups.values()]
    disparate_impact = min(selection_rates) / max(selection_rates) if max(selection_rates) else np.nan
    rows.append({'Strategy': name, **{key: result[key] for key in ['accuracy', 'precision', 'recall', 'f1']}, 'Disparate impact': disparate_impact, 'DP gap': differences(groups, 'selection_rate'), 'TPR gap': differences(groups, 'TPR'), 'FPR gap': differences(groups, 'FPR')})

st.subheader('Performance and fairness comparison')
st.dataframe(pd.DataFrame(rows).set_index('Strategy').style.format('{:.3f}'))
st.subheader('Per-group outcomes')
strategy = st.selectbox('Inspect strategy', list(group_tables))
st.dataframe(pd.DataFrame.from_dict(group_tables[strategy], orient='index').rename_axis('sex group').style.format('{:.3f}'))
st.write('Reweighing weight summary:', {'minimum': round(float(weights.min()), 3), 'maximum': round(float(weights.max()), 3), 'mean': round(float(weights.mean()), 3)})
st.caption('A mitigation can improve a measured group gap while harming F1 or another fairness criterion. Dropping sex can leave correlated proxies; reweighting changes the training objective. Review uncertainty, subgroup sizes, intersectional groups, and task-specific costs before deployment.')