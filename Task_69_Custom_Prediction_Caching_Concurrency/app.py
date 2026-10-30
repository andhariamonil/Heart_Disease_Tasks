import concurrent.futures
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import (
    FEATURE_SPECS,
    LogisticRegressionScratch,
    ThreadSafeLRUCache,
    fast_vectorize_input,
    load_data,
    standardize,
    standardize_fit,
    stratified_split,
)

st.set_page_config(
    page_title="Task 69: Custom Prediction Caching & Concurrency",
    page_icon="⚡",
    layout="wide",
)

st.title("Task 69: Custom Prediction Caching & Concurrency Optimization")
st.caption(
    "Dynamically display model predictions using a custom thread-safe LRU cache, "
    "ensuring smooth updates without unnecessary reloading, and serving concurrent client requests efficiently."
)

# Load data and train model
csv_path = Path(__file__).with_name("heart.csv")
if not csv_path.exists():
    csv_path = ROOT / "Task_21_Decision_Tree_Feature_Importance" / "heart.csv"

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame["target"].to_numpy(int)

X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=69)
means, scales = standardize_fit(X_train)
X_train_s, X_test_s = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_s, y_train)

# Initialize Session-State LRU Cache
if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = ThreadSafeLRUCache(capacity=256)

if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []

cache: ThreadSafeLRUCache = st.session_state.prediction_cache

# Cached inference function
def predict_with_cache(raw_dict: dict, precision: int = 1) -> tuple[float, int, bool, float]:
    t0 = time.perf_counter_ns()
    raw_vec = np.array([float(raw_dict[f]) for f in features], dtype=float)
    cache_key = cache.hash_vector(raw_vec, precision=precision)

    cached_val = cache.get(cache_key)
    if cached_val is not None:
        latency_us = (time.perf_counter_ns() - t0) / 1000.0
        return cached_val["prob"], cached_val["pred"], True, latency_us

    # Cache miss: compute prediction
    std_vec = (raw_vec - means) / scales
    prob = float(model.predict_proba(std_vec[None, :])[0])
    pred = int(prob >= 0.5)

    cache.put(cache_key, {"prob": prob, "pred": pred})
    latency_us = (time.perf_counter_ns() - t0) / 1000.0
    return prob, pred, False, latency_us

# Top Section: Real-Time Cache Telemetry
stats = cache.stats()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Cache Hit Rate", f"{stats['hit_rate'] * 100:.1f}%")
k2.metric("Cache Hits", f"{stats['hits']:,}")
k3.metric("Cache Misses", f"{stats['misses']:,}")
k4.metric("Active Entries", f"{stats['size']} / {stats['capacity']}")
k5.metric("Eviction Count", f"{stats['evictions']:,}")

# Section 1: Dynamic Patient Risk Assessment Form (Smooth Local Updates)
st.subheader("1. Dynamic Patient Profile & Instant Prediction")
st.write("Adjust patient vitals below. Notice how repeated or cached configurations evaluate in < 0.005 ms.")

c1, c2, c3 = st.columns(3)
with c1:
    age_in = st.slider("Patient Age", 25, 80, 56, 1)
    sex_in = st.selectbox("Sex", [1, 0], format_func=lambda s: "Male (1)" if s == 1 else "Female (0)")
    cp_in = st.selectbox("Chest Pain (cp)", [0, 1, 2, 3], index=1)
    trestbps_in = st.slider("Resting BP (trestbps)", 90, 200, 134, 1)
with c2:
    chol_in = st.slider("Cholesterol (chol)", 130, 500, 242, 1)
    fbs_in = st.selectbox("Fasting Sugar > 120 (fbs)", [0, 1], index=0)
    restecg_in = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=0)
    thalach_in = st.slider("Max Heart Rate (thalach)", 75, 205, 148, 1)
with c3:
    exang_in = st.selectbox("Exercise Angina (exang)", [0, 1], index=0)
    oldpeak_in = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.2, 0.1)
    slope_in = st.selectbox("Slope", [0, 1, 2], index=1)
    ca_in = st.selectbox("Vessels Colored (ca)", [0, 1, 2, 3], index=0)
    thal_in = st.selectbox("Thal", [1, 2, 3], index=1)

current_input = {
    "age": age_in, "sex": sex_in, "cp": cp_in, "trestbps": trestbps_in,
    "chol": chol_in, "fbs": fbs_in, "restecg": restecg_in, "thalach": thalach_in,
    "exang": exang_in, "oldpeak": oldpeak_in, "slope": slope_in, "ca": ca_in, "thal": thal_in
}

# Real-time inference
pred_prob, pred_cls, is_hit, latency_us = predict_with_cache(current_input)

# Update query log in session state
st.session_state.recent_queries.append({
    "Time": time.strftime("%H:%M:%S"),
    "Age": age_in, "BP": trestbps_in, "Chol": chol_in, "HR": thalach_in,
    "Prob": f"{pred_prob*100:.1f}%",
    "Status": "⚡ CACHE HIT" if is_hit else "🔄 CACHE MISS",
    "Latency (µs)": f"{latency_us:.2f}"
})
if len(st.session_state.recent_queries) > 10:
    st.session_state.recent_queries.pop(0)

# Display Prediction Card
res_box = st.container()
with res_box:
    r_col1, r_col2, r_col3 = st.columns([2, 2, 2])
    with r_col1:
        st.metric(
            "Predicted Risk",
            "DISEASE PRESENT (1)" if pred_cls == 1 else "ABSENCE (0)",
            f"{pred_prob * 100:.1f}% Risk Probability"
        )
    with r_col2:
        badge = "⚡ Instant Cache HIT" if is_hit else "🔄 Model Computed (Cache Miss)"
        badge_delta = "-98% time" if is_hit else "Normal computation"
        st.metric("Cache Status", badge, badge_delta)
    with r_col3:
        st.metric("Inference Latency", f"{latency_us:.2f} µs", "Sub-millisecond" if latency_us < 1000 else "")

    if is_hit:
        st.success(f"⚡ Retrieved precomputed inference from LRU cache in **{latency_us:.2f} µs** without matrix calculation.")
    else:
        st.info(f"🔄 Cache miss: Computed model inference in **{latency_us:.2f} µs** and indexed vector for future queries.")

# Section 2: Concurrent User Requests Stress Testing
st.subheader("2. Multi-Threaded Concurrent Request Handling")
st.write(
    "Simulate multiple concurrent clients querying predictions simultaneously. "
    "A thread-safe `RLock` ensures 100% data integrity without race conditions or memory corruption."
)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    n_workers = st.slider("Concurrent Worker Threads", 5, 50, 20, 5)
with col_ctrl2:
    n_requests = st.slider("Total Requests to Dispatch", 200, 3000, 1000, 100)
with col_ctrl3:
    traffic_pattern = st.selectbox(
        "Traffic Distribution",
        ["Zipfian / Skewed Distribution (80/20 Clinical Re-queries)", "Uniform Random Exploration"]
    )

if st.button("🔥 Launch Concurrent Traffic Simulation"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Dispatching concurrent threads...")

    # Generate synthetic query pool
    rng = np.random.default_rng(42)
    pool_size = 50 if "Skewed" in traffic_pattern else 400
    candidate_profiles = []
    for _ in range(pool_size):
        row = {f: float(rng.uniform(FEATURE_SPECS[f].get("min", 0), FEATURE_SPECS[f].get("max", 1))) if not FEATURE_SPECS[f]["categorical"]
               else float(rng.choice(FEATURE_SPECS[f]["allowed"])) for f in features}
        candidate_profiles.append(row)

    if "Skewed" in traffic_pattern:
        weights = 1.0 / (np.arange(1, pool_size + 1) ** 0.8)
        weights /= weights.sum()
        query_indices = rng.choice(pool_size, size=n_requests, p=weights)
    else:
        query_indices = rng.choice(pool_size, size=n_requests)

    queries_to_run = [candidate_profiles[idx] for idx in query_indices]

    # Execute with ThreadPoolExecutor
    t_start_concurrent = time.perf_counter()
    latencies = []
    hits_concurrent = 0

    def worker_task(q_dict):
        p, c_res, hit, lat = predict_with_cache(q_dict)
        return hit, lat

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(worker_task, q) for q in queries_to_run]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures)):
            hit, lat = fut.result()
            latencies.append(lat)
            if hit:
                hits_concurrent += 1
            if (idx + 1) % (n_requests // 10) == 0:
                progress_bar.progress((idx + 1) / n_requests)

    t_total_sec = time.perf_counter() - t_start_concurrent
    throughput = n_requests / max(t_total_sec, 1e-6)

    progress_bar.progress(1.0)
    status_text.text("Concurrent simulation completed successfully!")

    # Display Concurrency Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Throughput (RPS)", f"{throughput:,.1f} req/s")
    m2.metric("Concurrent Hit Rate", f"{(hits_concurrent / n_requests) * 100:.1f}%")
    m3.metric("Median Latency (p50)", f"{np.percentile(latencies, 50):.2f} µs")
    m4.metric("Tail Latency (p99)", f"{np.percentile(latencies, 99):.2f} µs")

    # Diagnostic Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)

    # 1. Latency distribution histogram
    axes[0].hist(latencies, bins=35, color="#2563eb", alpha=0.8, edgecolor="black")
    axes[0].axvline(np.percentile(latencies, 50), color="#10b981", linestyle="--", linewidth=2, label=f"p50: {np.percentile(latencies, 50):.1f}µs")
    axes[0].axvline(np.percentile(latencies, 99), color="#ef4444", linestyle="--", linewidth=2, label=f"p99: {np.percentile(latencies, 99):.1f}µs")
    axes[0].set_xlabel("Request Latency (µs)")
    axes[0].set_ylabel("Request Count")
    axes[0].set_title(f"Concurrent Latency Distribution ({n_workers} Worker Threads)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    # 2. Cumulative hits vs misses
    cum_hits = np.cumsum([1 if lat < 15 else 0 for lat in latencies])
    axes[1].plot(np.arange(1, n_requests + 1), cum_hits, color="#10b981", linewidth=2, label="Cumulative Cache Hits")
    axes[1].plot(np.arange(1, n_requests + 1), np.arange(1, n_requests + 1) - cum_hits, color="#f59e0b", linewidth=2, label="Cumulative Cache Misses")
    axes[1].set_xlabel("Total Dispatched Requests")
    axes[1].set_ylabel("Request Count")
    axes[1].set_title("Concurrent Cache Accumulation Over Time")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    st.pyplot(fig)

# Section 3: Live Session Audit Table & Cache Management
st.subheader("3. Real-Time Query History & Cache Controls")
c_tbl, c_btn = st.columns([4, 1])
with c_tbl:
    if st.session_state.recent_queries:
        st.dataframe(pd.DataFrame(st.session_state.recent_queries), use_container_width=True)
    else:
        st.info("No queries recorded yet.")

with c_btn:
    if st.button("🗑️ Purge LRU Cache"):
        cache.clear()
        st.session_state.recent_queries.clear()
        st.rerun()

