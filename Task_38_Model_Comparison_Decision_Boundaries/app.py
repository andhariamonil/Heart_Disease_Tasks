import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, KNearestNeighborsScratch, SVMLinearScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 38: SVM, k-NN, and Decision Tree Comparison')
st.write('The two-feature view makes each model boundary inspectable; metrics use a separate held-out split.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, _, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
features = ['age', 'thalach']
X = frame[features].to_numpy(dtype=float)
y = frame['target'].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.25, seed=38)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_test = standardize(X_test, means, scales)

models = {
    'Linear SVM': SVMLinearScratch(learning_rate=0.01, epochs=1000).fit(X_train, y_train),
    'k-NN (k=7)': KNearestNeighborsScratch(k=7).fit(X_train, y_train),
    'Decision Tree': DecisionTreeScratch(max_depth=4, min_samples_split=8).fit(X_train, y_train, features),
}
metric_rows = []
for name, model in models.items():
    metric_rows.append({'Model': name, **metrics(y_test, model.predict(X_test))})
st.subheader('Held-out metrics')
st.dataframe(pd.DataFrame(metric_rows).set_index('Model')[['accuracy', 'precision', 'recall', 'f1']].style.format('{:.3f}'))

st.subheader('Decision regions')
axis = np.linspace(min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.3, max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.3, 100)
vertical = np.linspace(min(X_train[:, 1].min(), X_test[:, 1].min()) - 0.3, max(X_train[:, 1].max(), X_test[:, 1].max()) + 0.3, 100)
xx, yy = np.meshgrid(axis, vertical)
grid = np.column_stack([xx.ravel(), yy.ravel()])
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for plot_axis, (name, model) in zip(axes, models.items()):
    regions = model.predict(grid).reshape(xx.shape)
    plot_axis.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5], alpha=0.24, cmap='RdYlBu')
    plot_axis.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black', s=20)
    plot_axis.set_title(name)
    plot_axis.set_xlabel('Standardized age')
    plot_axis.set_ylabel('Standardized maximum heart rate')
st.pyplot(fig)
st.markdown('**Trade-offs:** the linear SVM is compact and can generalize well when a margin is meaningful; k-NN is intuitive but prediction cost and sensitivity to scaling grow with data size; a shallow tree is interpretable but its axis-aligned regions can be unstable. This comparison uses only age and maximum heart rate, so it is educational, not a clinical decision tool.')