import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, metrics, standardize, standardize_fit, stratified_split


def group_fairness(y_true, y_pred, groups):
    results = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        actual, predicted = y_true[mask], y_pred[mask]
        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        selection_rate = float(predicted.mean()) if len(predicted) else 0.0
        tpr = tp / (tp + fn) if tp + fn else float('nan')
        fpr = fp / (fp + tn) if fp + tn else float('nan')
        results[group] = {'n': int(mask.sum()), 'selection_rate': selection_rate, 'TPR': tpr, 'FPR': fpr, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    return results


def gap(group_results, metric):
    values = [row[metric] for row in group_results.values() if pd.notna(row[metric])]
    return max(values) - min(values) if len(values) >= 2 else float('nan')


st.title('Task 60: Manual Fairness Metrics')
st.caption('The dataset’s `sex` field is used as the available binary group attribute; it is not a comprehensive demographic fairness audit.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y, groups = frame[features].to_numpy(float), frame['target'].to_numpy(int), frame['sex'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=60)
_, groups_test, _, _ = stratified_split(groups.reshape(-1, 1), y, 0.2, seed=60)
groups_test = groups_test.ravel()
means, scales = standardize_fit(X_train)
X_train, X_test = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train, y_train)
threshold = st.slider('Decision threshold', 0.1, 0.9, 0.5, 0.05)
prediction = model.predict(X_test, threshold)
group_results = group_fairness(y_test, prediction, groups_test)
group_table = pd.DataFrame.from_dict(group_results, orient='index').rename_axis('sex group')
st.subheader('Group confusion and rate metrics')
st.dataframe(group_table.style.format('{:.3f}'))
selection_rates = [row['selection_rate'] for row in group_results.values()]
max_rate = max(selection_rates) if selection_rates else 0.0
min_rate = min(selection_rates) if selection_rates else 0.0
disparate_impact = min_rate / max_rate if max_rate > 0 else float('nan')
st.subheader('Manual group comparisons')
st.write(f'Disparate impact (lower selection rate / higher selection rate): {disparate_impact:.3f}')
st.write(f'Demographic-parity selection-rate gap: {gap(group_results, "selection_rate"):.3f}')
st.write(f'Equalized-odds TPR gap: {gap(group_results, "TPR"):.3f}')
st.write(f'Equalized-odds FPR gap: {gap(group_results, "FPR"):.3f}')
st.write('Overall performance:', {key: round(value, 3) for key, value in metrics(y_test, prediction).items() if key in {'accuracy', 'precision', 'recall', 'f1'}})
st.caption('Ratios/gaps are descriptive estimates, sensitive to subgroup size and the chosen threshold. The 0.8 disparate-impact heuristic is not a universal legal or clinical standard; equalized odds can conflict with calibration and requires domain-specific review.')