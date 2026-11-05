import json
import statistics
import sys
import time
import tracemalloc
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, standardize, standardize_fit, validate_input_row


@lru_cache(maxsize=1)
def load_model_artifact(path_string, modified_time_ns):
    return joblib.load(path_string)


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -35, 35)))


st.title('Task 80: API Latency and Inference Profiling')
st.caption('Local microbenchmarks isolate API stages and quantization error. They are hardware-specific and do not replace measurements through the deployed ALB.')
artifact_path = ROOT / 'Task_47_Versioned_Model_Export' / 'artifacts' / 'heart_logistic_v1.0.0.joblib'
frame_path = Path(__file__).with_name('heart.csv')
if not frame_path.exists():
    frame_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
artifact_stat = artifact_path.stat()
load_start = time.perf_counter_ns()
artifact = load_model_artifact(str(artifact_path), artifact_stat.st_mtime_ns)
cold_load_ms = (time.perf_counter_ns() - load_start) / 1e6
uncached_loads = 20
load_start = time.perf_counter_ns()
for _ in range(uncached_loads):
    joblib.load(artifact_path)
uncached_average_ms = (time.perf_counter_ns() - load_start) / 1e6 / uncached_loads
cached_loads = 5000
load_start = time.perf_counter_ns()
for _ in range(cached_loads):
    load_model_artifact(str(artifact_path), artifact_stat.st_mtime_ns)
cached_average_us = (time.perf_counter_ns() - load_start) / 1e3 / cached_loads
_, _, features, frame = load_data(frame_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = __import__('manual_ml').stratified_split(X, y, 0.2, seed=80)
means, scales = np.asarray(artifact['preprocessing']['means']), np.asarray(artifact['preprocessing']['scales'])
baseline_model = LogisticRegressionScratch(**artifact['parameters'])
baseline_model.weights = np.asarray(artifact['state']['weights'], dtype=np.float32)
baseline_model.bias = float(artifact['state']['bias'])
means, scales = means.astype(np.float32), scales.astype(np.float32)
test_scaled = ((X_test - means) / scales).astype(np.float32)

weight_fp32 = baseline_model.weights
weight_fp16 = weight_fp32.astype(np.float16)
weight_scale = max(float(np.max(np.abs(weight_fp32))) / 127, 1e-12)
weight_int8 = np.clip(np.round(weight_fp32 / weight_scale), -127, 127).astype(np.int8)
input_scale = 1 / 32


def probability_fp32(scaled):
    return sigmoid(scaled @ weight_fp32 + baseline_model.bias)


def probability_fp16(scaled):
    logits = (scaled.astype(np.float16) @ weight_fp16).astype(np.float32) + np.float32(baseline_model.bias)
    return sigmoid(logits)


def probability_int8(scaled):
    quantized = np.clip(np.round(scaled / input_scale), -127, 127).astype(np.int8)
    logits = (quantized.astype(np.int32) @ weight_int8.astype(np.int32)) * input_scale * weight_scale + baseline_model.bias
    return sigmoid(logits)


quantization_rows = []
reference_probability = probability_fp32(test_scaled)
for name, function, weight_bytes in [
    ('FP32', probability_fp32, weight_fp32.nbytes),
    ('FP16', probability_fp16, weight_fp16.nbytes),
    ('INT8', probability_int8, weight_int8.nbytes),
]:
    start = time.perf_counter_ns()
    probabilities = function(test_scaled)
    batch_ms = (time.perf_counter_ns() - start) / 1e6
    quantization_rows.append({'Mode': name, 'Weight bytes': weight_bytes, 'Batch inference ms': batch_ms, 'Mean absolute probability error vs FP32': float(np.mean(np.abs(probabilities - reference_probability))), 'Maximum error': float(np.max(np.abs(probabilities - reference_probability)))})

raw_example = {name: float(value) for name, value in zip(features, X_test[0])}
categorical_codes = {'sex': [0, 1], 'cp': [0, 1, 2, 3], 'fbs': [0, 1], 'restecg': [0, 1, 2], 'exang': [0, 1], 'slope': [0, 1, 2], 'ca': [0, 1, 2, 3, 4], 'thal': [0, 1, 2, 3]}
hard_ranges = {'age': (18, 100), 'trestbps': (60, 260), 'chol': (80, 700), 'thalach': (40, 250), 'oldpeak': (-5, 15)}
parsed_input, issues, _ = validate_input_row(raw_example, features, categorical_codes, hard_ranges, {})
if any(issue['level'] == 'error' for issue in issues):
    st.error('Selected local test example failed the serving validator.')
    st.stop()
input_vector = np.array([parsed_input[name] for name in features], dtype=np.float32)


def request_stages():
    timings = {}
    start = time.perf_counter_ns()
    parsed, issues, _ = validate_input_row(raw_example, features, categorical_codes, hard_ranges, {})
    timings['validation_ms'] = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    scaled = ((np.array([parsed[name] for name in features], dtype=np.float32) - means) / scales)
    timings['scaling_ms'] = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    probability = float(probability_fp32(scaled[None, :])[0])
    timings['inference_ms'] = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    contributions = weight_fp32 * scaled
    top = np.argsort(np.abs(contributions))[::-1][:3]
    explanation = [{'feature': features[index], 'logit_contribution': float(contributions[index])} for index in top]
    timings['explanation_ms'] = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    json.dumps({'probability': probability, 'explanation': explanation})
    timings['serialization_ms'] = (time.perf_counter_ns() - start) / 1e6
    timings['total_ms'] = sum(timings.values())
    return timings


iterations = st.slider('Local pipeline repetitions', 100, 3000, 1000, 100)
if st.button('Profile local inference'):
    tracemalloc.start()
    stage_runs = [request_stages() for _ in range(iterations)]
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stage_frame = pd.DataFrame(stage_runs)
    profile = pd.DataFrame({'Median ms': stage_frame.median(), 'P95 ms': stage_frame.quantile(0.95), 'P99 ms': stage_frame.quantile(0.99)})
    st.subheader('Request-stage latency')
    st.dataframe(profile.style.format('{:.4f}'))
    st.write(f'Peak traced Python allocation during profile: {peak_bytes / (1024 ** 2):.3f} MiB')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(profile.index.drop('total_ms'), profile.loc[profile.index.drop('total_ms'), 'Median ms'], color='#176b68')
    ax.tick_params(axis='x', rotation=20); ax.set_ylabel('Median milliseconds'); ax.set_title('Per-stage median latency'); st.pyplot(fig)

st.subheader('FP32 / FP16 / INT8 weight comparison')
st.dataframe(pd.DataFrame(quantization_rows).set_index('Mode').style.format({'Batch inference ms': '{:.4f}', 'Mean absolute probability error vs FP32': '{:.6f}', 'Maximum error': '{:.6f}'}))
st.write(f'Cold model load: {cold_load_ms:.3f} ms; repeated uncached file load: {uncached_average_ms:.3f} ms/read; cached metadata lookup: {cached_average_us:.3f} µs/access.')
st.caption('Only the immutable model artifact/metadata is cached for the worker lifetime. User feature payloads and predictions are not cached to avoid retaining sensitive inputs. Try FP16 only if its validation error is acceptable; INT8 uses fixed activation scale 1/32 and can saturate outside the normal range.')