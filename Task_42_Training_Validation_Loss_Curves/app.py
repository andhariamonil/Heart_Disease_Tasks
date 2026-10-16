import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import MLPClassifierScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 42: Training and Validation Loss Curves')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(dtype=float), frame['target'].to_numpy(dtype=int)
X_train_full, X_test, y_train_full, y_test = stratified_split(X, y, 0.2, seed=42)
X_train, X_validation, y_train, y_validation = stratified_split(X_train_full, y_train_full, 0.2, seed=43)
means, scales = standardize_fit(X_train)
X_train = standardize(X_train, means, scales)
X_validation = standardize(X_validation, means, scales)
X_test = standardize(X_test, means, scales)

model = MLPClassifierScratch((24, 12), learning_rate=0.008, epochs=140, batch_size=32, dropout_rate=0.15, batch_normalization=True, seed=42)
model.fit(X_train, y_train, validation_data=(X_validation, y_validation))
test_metrics = metrics(y_test, model.predict(X_test))

st.subheader('Manual loss histories')
fig, ax = plt.subplots(figsize=(9, 5))
epochs = range(1, len(model.loss_history) + 1)
ax.plot(epochs, model.loss_history, color='#147d64', linewidth=1.8, label='Training loss')
ax.plot(epochs, model.validation_loss_history, color='#b34a36', linewidth=1.8, label='Validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Binary cross-entropy')
ax.set_title('Training and validation learning trends')
ax.legend()
ax.grid(True, alpha=0.25)
st.pyplot(fig)
st.write(f'Epochs recorded: {len(model.loss_history)}; minimum validation loss: {min(model.validation_loss_history):.4f} at epoch {model.validation_loss_history.index(min(model.validation_loss_history)) + 1}')
st.subheader('Final held-out test metrics')
st.dataframe(pd.DataFrame([test_metrics])[['accuracy', 'precision', 'recall', 'f1']].style.format('{:.3f}'))
st.markdown('**Reading the curves:** both losses falling suggests learning; training loss falling while validation loss rises suggests overfitting; flat or jointly high curves suggest underfitting or optimization trouble. The held-out test set is not used to select epochs.')
st.caption('Loss values are recorded in the training loop and plotted using Matplotlib axes directly; no Keras visualization utilities are used.')