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


def manual_kmeans(x, cluster_count=3, seed=59, iterations=50):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=cluster_count, replace=False)].copy()
    assignments = np.zeros(len(x), dtype=int)
    for _ in range(iterations):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated_assignments = np.argmin(distances, axis=1)
        updated_centers = centers.copy()
        for cluster in range(cluster_count):
            members = x[updated_assignments == cluster]
            if len(members):
                updated_centers[cluster] = members.mean(axis=0)
        if np.array_equal(updated_assignments, assignments) and np.allclose(updated_centers, centers):
            break
        assignments, centers = updated_assignments, updated_centers
    distances = np.sum((x - centers[assignments]) ** 2, axis=1)
    return assignments, centers, distances


st.title('Task 59: Misclassification and Error-Cluster Analysis')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=59)
means, scales = standardize_fit(X_train)
X_train_scaled, X_test_scaled = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_scaled, y_train)
probabilities = model.predict_proba(X_test_scaled)
threshold = st.slider('Classification threshold', 0.1, 0.9, 0.5, 0.05)
predictions = (probabilities >= threshold).astype(int)
false_positive = (y_test == 0) & (predictions == 1)
false_negative = (y_test == 1) & (predictions == 0)
st.write('Held-out metrics:', {key: round(value, 3) for key, value in metrics(y_test, predictions).items() if key in {'accuracy', 'precision', 'recall', 'f1'}})

all_indices = np.arange(len(frame))
_, test_indices, _, _ = stratified_split(all_indices.reshape(-1, 1), y, 0.2, seed=59)
test_frame = frame.iloc[test_indices.ravel()].reset_index(drop=True)
error_mask = false_positive | false_negative
error_rows = test_frame.loc[error_mask, features].copy().reset_index(drop=True)
error_rows['Actual'] = y_test[error_mask]
error_rows['Predicted'] = predictions[error_mask]
error_rows['Probability'] = probabilities[error_mask]
error_rows['Error type'] = np.where(false_positive[error_mask], 'False positive', 'False negative')
st.subheader('Misclassified examples')
st.dataframe(error_rows.head(20).style.format({'Probability': '{:.3f}'}))

if len(error_rows) >= 2:
    error_values = (error_rows[features].to_numpy(float) - means) / scales
    cluster_count = min(3, len(error_values))
    clusters, centers, distances = manual_kmeans(error_values, cluster_count)
    error_rows['Cluster'] = clusters
    st.subheader('Manual k-means clusters of errors')
    cluster_summary = error_rows.groupby('Cluster').agg(errors=('Error type', 'size'), false_positives=('Error type', lambda values: int((values == 'False positive').sum())), false_negatives=('Error type', lambda values: int((values == 'False negative').sum())))
    st.dataframe(cluster_summary)
    st.write('Cluster centers, in standardized feature space:')
    st.dataframe(pd.DataFrame(centers, columns=features).round(2))
    st.write(f'Within-cluster squared distance: {distances.sum():.3f}')
else:
    st.info('Too few errors for a meaningful multi-cluster comparison on this split.')

error_feature_values = test_frame.loc[error_mask, features].to_numpy(float)
z_scores = np.abs((error_feature_values - means) / scales)
anomaly_counts = pd.Series((z_scores > 2.5).sum(axis=0), index=features, name='Error examples beyond 2.5 training SD').sort_values(ascending=False)
st.subheader('Feature-distribution anomaly scan')
st.dataframe(anomaly_counts.to_frame())
error_frame = test_frame.copy()
error_frame['Actual'] = y_test
error_frame['Predicted'] = predictions
error_frame['Error type'] = np.where(false_positive, 'False positive', np.where(false_negative, 'False negative', 'Correct'))
error_frame['Age band'] = pd.cut(error_frame['age'], [0, 45, 55, 65, 120], labels=['≤45', '46–55', '56–65', '>65'])


def subgroup_error_rates(frame, group_column):
    rows = []
    for group_value, subset in frame.groupby(group_column, observed=False):
        actual = subset['Actual'].to_numpy()
        predicted = subset['Predicted'].to_numpy()
        negatives = actual == 0
        positives = actual == 1
        fp = int(np.sum(negatives & (predicted == 1)))
        fn = int(np.sum(positives & (predicted == 0)))
        rows.append({group_column: group_value, 'rows': len(subset), 'FP count': fp, 'actual negatives': int(negatives.sum()), 'FPR': fp / negatives.sum() if negatives.any() else np.nan, 'FN count': fn, 'actual positives': int(positives.sum()), 'FNR': fn / positives.sum() if positives.any() else np.nan})
    return pd.DataFrame(rows).set_index(group_column)


st.subheader('Error rates by subgroup')
st.write('FPR is FP divided by actual negatives; FNR is FN divided by actual positives.')
st.dataframe(subgroup_error_rates(error_frame, 'sex').style.format('{:.3f}'))
st.dataframe(subgroup_error_rates(error_frame, 'Age band').style.format('{:.3f}'))

if len(error_rows) >= 2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(error_rows['age'], error_rows['thalach'], c=error_rows['Cluster'], cmap='tab10', s=65)
    ax.set_xlabel('Age'); ax.set_ylabel('Maximum heart rate'); ax.set_title('Error clusters on two selected features'); ax.grid(alpha=0.2)
    st.pyplot(fig)
st.caption('Clusters and standardized outlier flags are exploratory diagnostics. Small test subgroups are unstable; review records and measurement provenance before attributing errors to a population or changing policy.')