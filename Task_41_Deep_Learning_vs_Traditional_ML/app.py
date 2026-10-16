import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, LogisticRegressionScratch, MLPClassifierScratch, RandomForestScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 41: Deep Learning vs Traditional Machine Learning')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(dtype=float), frame['target'].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.25, seed=41)
means, scales = standardize_fit(X_train)
X_train, X_test = standardize(X_train, means, scales), standardize(X_test, means, scales)

models = {
    'Logistic regression': LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02),
    'Decision tree': DecisionTreeScratch(max_depth=4, min_samples_split=8),
    'Random forest': RandomForestScratch(n_trees=8, max_depth=4, seed=41),
    'Scratch neural network': MLPClassifierScratch((16, 8), learning_rate=0.01, epochs=100, batch_size=32, dropout_rate=0.15, batch_normalization=True, seed=41),
}
rows = []
for name, model in models.items():
    if isinstance(model, (DecisionTreeScratch, RandomForestScratch)):
        model.fit(X_train, y_train, features)
    elif isinstance(model, MLPClassifierScratch):
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    train_scores = metrics(y_train, model.predict(X_train))
    test_scores = metrics(y_test, model.predict(X_test))
    rows.append({'Model': name, 'Train accuracy': train_scores['accuracy'], 'Test accuracy': test_scores['accuracy'], 'Accuracy gap': train_scores['accuracy'] - test_scores['accuracy'], 'Test precision': test_scores['precision'], 'Test recall': test_scores['recall'], 'Test F1': test_scores['f1']})

results = pd.DataFrame(rows).set_index('Model')
st.subheader('Train/test metrics and generalization gap')
st.dataframe(results.style.format('{:.3f}'))
fig, ax = plt.subplots(figsize=(9, 4))
positions = range(len(results))
ax.bar([p - 0.18 for p in positions], results['Train accuracy'], width=0.36, label='Train')
ax.bar([p + 0.18 for p in positions], results['Test accuracy'], width=0.36, label='Test')
ax.set_xticks(list(positions), results.index, rotation=15, ha='right')
ax.set_ylabel('Accuracy')
ax.set_title('Training vs held-out accuracy')
ax.legend()
ax.grid(axis='y', alpha=0.2)
st.pyplot(fig)
st.markdown('**Interpretation:** classical methods often train quickly and can be easier to explain or deploy on small tabular data; trees expose readable rules, while logistic regression gives compact coefficients. Deep networks can learn nonlinear feature interactions and benefit from larger, richer datasets, but require more data and tuning and can overfit or be harder to explain. Accuracy alone is incomplete: precision and recall matter differently depending on the cost of false alarms and missed cases. This educational dataset is not a clinical tool.')