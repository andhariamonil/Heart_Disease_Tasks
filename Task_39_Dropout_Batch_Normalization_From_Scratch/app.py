import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import MLPClassifierScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 39: Dropout and Batch Normalization From Scratch')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(dtype=float), frame['target'].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=39)
means, scales = standardize_fit(X_train)
X_train, X_test = standardize(X_train, means, scales), standardize(X_test, means, scales)

variants = {
    'Plain MLP': {'dropout_rate': 0.0, 'batch_normalization': False},
    'Dropout + Batch Norm': {'dropout_rate': 0.25, 'batch_normalization': True},
}
trained = {}
rows = []
for index, (name, options) in enumerate(variants.items()):
    model = MLPClassifierScratch((16, 8), learning_rate=0.01, epochs=100, batch_size=32, seed=39 + index, **options).fit(X_train, y_train)
    trained[name] = model
    rows.append({'Model': name, **metrics(y_test, model.predict(X_test)), 'Final training loss': model.loss_history[-1]})

st.subheader('Held-out metrics')
st.dataframe(pd.DataFrame(rows).set_index('Model')[['accuracy', 'precision', 'recall', 'f1', 'Final training loss']].style.format('{:.3f}'))
fig, ax = plt.subplots(figsize=(8, 4))
for name, model in trained.items():
    ax.plot(model.loss_history, label=name)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training cross-entropy')
ax.set_title('Training curves')
ax.legend()
ax.grid(alpha=0.25)
st.pyplot(fig)
batch_norm_model = trained['Dropout + Batch Norm']
st.write(f'Batch-normalization layers: {len(batch_norm_model.running_mean)}')
st.write(f'Mean observed mini-batch activation variance: {np.mean(batch_norm_model.batch_variance_history):.4f}')
st.caption('Dropout uses inverted Bernoulli masks during training. Batch-normalization mean, variance, gamma, beta, and gradients are updated manually; inference uses running statistics.')