import gc
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import DecisionTreeScratch, KNearestNeighborsScratch, LogisticRegressionScratch, MLPClassifierScratch, RandomForestScratch, load_data, metrics, standardize, standardize_fit, stratified_split


st.title('Task 52: Runtime, Memory, and Performance Benchmark')
st.write('Measurements are collected on one fixed split in this local process; they are practical comparisons, not hardware-independent guarantees.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=52)
means, scales = standardize_fit(X_train)
X_train, X_test = standardize(X_train, means, scales), standardize(X_test, means, scales)

model_factories = {
    'Logistic regression': lambda: LogisticRegressionScratch(learning_rate=0.06, iterations=1000, l2=0.02),
    'Decision tree': lambda: DecisionTreeScratch(max_depth=4, min_samples_split=8),
    'Random forest (6 trees)': lambda: RandomForestScratch(n_trees=6, max_depth=4, seed=52),
    'k-NN (k=7)': lambda: KNearestNeighborsScratch(k=7),
    'Neural network': lambda: MLPClassifierScratch((16, 8), epochs=60, batch_size=32, dropout_rate=0.1, batch_normalization=True, seed=52),
}
rows = []
for name, factory in model_factories.items():
    gc.collect()
    model = factory()
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    if isinstance(model, (DecisionTreeScratch, RandomForestScratch)):
        model.fit(X_train, y_train, features)
    else:
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    peak_extra_mb = max(0, peak_bytes - baseline_current) / (1024 ** 2)
    tracemalloc.stop()
    started = time.perf_counter()
    predicted = model.predict(X_test)
    predict_seconds = time.perf_counter() - started
    result = metrics(y_test, predicted)
    rows.append({'Model': name, 'Fit seconds': fit_seconds, 'Prediction seconds': predict_seconds, 'Peak traced extra MB': peak_extra_mb, 'Test accuracy': result['accuracy'], 'Test F1': result['f1'], 'Train rows': len(y_train), 'Test rows': len(y_test)})

table = pd.DataFrame(rows).set_index('Model').sort_values('Fit seconds')
st.subheader('Measured cost and held-out quality')
st.dataframe(table.style.format('{:.3f}'))
fig, ax = plt.subplots(figsize=(8, 5))
for name, row in table.iterrows():
    ax.scatter(row['Fit seconds'], row['Test F1'], s=60 + row['Peak traced extra MB'] * 8, alpha=0.7)
    ax.annotate(name, (row['Fit seconds'], row['Test F1']), xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Fit time (seconds)'); ax.set_ylabel('Held-out F1'); ax.set_title('Compute cost vs predictive quality (marker size ~ traced memory)'); ax.grid(alpha=0.25)
st.pyplot(fig)
fastest = table.index[0]
st.write(f'Fastest measured fit: {fastest}. A practical choice should balance F1, prediction latency, memory, interpretability, and retraining budget.')
st.caption('Memory is the additional peak traced by Python tracemalloc during fit; allocator/native-library behavior and process baseline affect it. Timings vary by CPU, load, and library version. Benchmark on deployment hardware before choosing.')