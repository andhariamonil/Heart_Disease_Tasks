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


st.title('Task 51: Manual Neural-Network Learning-Rate Scheduler')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=51)
X_train, X_validation, y_train, y_validation = stratified_split(X_development, y_development, 0.2, seed=52)
means, scales = standardize_fit(X_train)
X_train, X_validation, X_test = standardize(X_train, means, scales), standardize(X_validation, means, scales), standardize(X_test, means, scales)

baseline = MLPClassifierScratch((24, 12), learning_rate=0.01, epochs=120, batch_size=32, dropout_rate=0.15, batch_normalization=True, seed=51)
baseline.fit(X_train, y_train, validation_data=(X_validation, y_validation))
scheduled = MLPClassifierScratch((24, 12), learning_rate=0.02, epochs=120, batch_size=32, dropout_rate=0.15, batch_normalization=True, seed=51)
scheduled.fit(X_train, y_train, validation_data=(X_validation, y_validation), scheduler={'factor': 0.5, 'patience': 7, 'min_delta': 0.0005, 'min_learning_rate': 0.0005})

rows = []
for name, model in [('Constant rate', baseline), ('Manual reduce-on-plateau', scheduled)]:
    rows.append({'Model': name, **metrics(y_test, model.predict(X_test)), 'Minimum validation loss': min(model.validation_loss_history), 'Final training loss': model.loss_history[-1]})
st.subheader('Final held-out metrics')
st.dataframe(pd.DataFrame(rows).set_index('Model')[['accuracy', 'precision', 'recall', 'f1', 'Minimum validation loss', 'Final training loss']].style.format('{:.3f}'))

fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
for name, model in [('Constant rate', baseline), ('Scheduled rate', scheduled)]:
    epochs = np.arange(1, len(model.loss_history) + 1)
    axes[0].plot(epochs, model.loss_history, label=f'{name} train')
    axes[0].plot(epochs, model.validation_loss_history, linestyle='--', label=f'{name} validation')
    axes[1].step(epochs, model.learning_rate_history, where='post', label=name)
    axes[2].plot(epochs, model.update_norm_history, label=name)
axes[0].set_ylabel('Cross-entropy'); axes[0].set_title('Training and validation convergence'); axes[0].legend(); axes[0].grid(alpha=0.2)
axes[1].set_ylabel('Learning rate'); axes[1].set_title('Manual rate schedule'); axes[1].legend(); axes[1].grid(alpha=0.2)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Cumulative update norm'); axes[2].set_title('Weight-update stability diagnostic'); axes[2].legend(); axes[2].grid(alpha=0.2)
st.pyplot(fig)
st.write('Scheduler reductions:', scheduled.scheduler_events)
st.caption('The rate changes only after validation loss fails to improve by min_delta for the configured patience. Smaller update norms after reductions indicate damped weight movement; they do not alone prove better generalization.')