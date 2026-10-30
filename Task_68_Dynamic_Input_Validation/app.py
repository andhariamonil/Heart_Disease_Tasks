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
    fast_vectorize_input,
    load_data,
    metrics,
    standardize,
    standardize_fit,
    stratified_split,
    validate_and_impute_input,
)

st.set_page_config(
    page_title="Task 68: Dynamic Input Validation & Real-Time Processing",
    page_icon="🛡️",
    layout="wide",
)

st.title("Task 68: Dynamic User Input Validation & Real-Time Processing")
st.caption(
    "Dynamically validate user inputs, handle missing/corrupted edge cases with automated fallback imputation, "
    "and optimize feature processing for microsecond real-time model inference."
)

# Load data and fit baseline model
csv_path = Path(__file__).with_name("heart.csv")
if not csv_path.exists():
    csv_path = ROOT / "Task_21_Decision_Tree_Feature_Importance" / "heart.csv"

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame["target"].to_numpy(int)

# Extract empirical reference medians / modes
reference_stats = {}
for col in features:
    if FEATURE_SPECS[col]["categorical"]:
        reference_stats[col] = float(frame[col].mode().iloc[0])
    else:
        reference_stats[col] = float(frame[col].median())

X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=68)
means, scales = standardize_fit(X_train)
X_train_s, X_test_s = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_s, y_train)

# Section 1: Presets & Input Mode
st.subheader("1. Patient Input Configuration & Edge Case Presets")
preset_choice = st.selectbox(
    "Select a pre-configured test scenario or create custom input:",
    [
        "Clean Typical Patient (All Valid Inputs)",
        "Edge Case 1: Partial Input (Missing 7 Attributes)",
        "Edge Case 2: Extreme Out-of-Bounds Values (Age=145, BP=310, CP=7)",
        "Edge Case 3: Malformed Strings & Type Mismatches (chol='elevated', fbs='yes')",
        "Edge Case 4: Physiological Conflict (Age 78 with Max Heart Rate 215 bpm)",
        "Custom Manual Input",
    ],
)

# Preset definitions
if preset_choice == "Clean Typical Patient (All Valid Inputs)":
    raw_input = {"age": 58, "sex": 1, "cp": 1, "trestbps": 135, "chol": 240, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2}
elif preset_choice == "Edge Case 1: Partial Input (Missing 7 Attributes)":
    raw_input = {"age": 62, "trestbps": 145, "chol": 280, "cp": 2, "thalach": 138, "oldpeak": 1.8}
elif preset_choice == "Edge Case 2: Extreme Out-of-Bounds Values (Age=145, BP=310, CP=7)":
    raw_input = {"age": 145, "sex": 1, "cp": 7, "trestbps": 310, "chol": 850, "fbs": 0, "restecg": 0, "thalach": 280, "exang": 0, "oldpeak": 12.5, "slope": 5, "ca": 9, "thal": 8}
elif preset_choice == "Edge Case 3: Malformed Strings & Type Mismatches (chol='elevated', fbs='yes')":
    raw_input = {"age": "60", "sex": "1", "cp": "0", "trestbps": "140", "chol": "elevated", "fbs": "yes", "restecg": "normal", "thalach": "130", "exang": "0", "oldpeak": "2.1", "slope": "1", "ca": "None", "thal": "3"}
elif preset_choice == "Edge Case 4: Physiological Conflict (Age 78 with Max Heart Rate 215 bpm)":
    raw_input = {"age": 78, "sex": 1, "cp": 0, "trestbps": 150, "chol": 230, "fbs": 0, "restecg": 1, "thalach": 215, "exang": 1, "oldpeak": 2.4, "slope": 1, "ca": 2, "thal": 2}
else:
    # Custom interactive sliders
    c1, c2, c3 = st.columns(3)
    with c1:
        age_in = st.number_input("Age", value=55, step=1)
        sex_in = st.selectbox("Sex (1=M, 0=F)", [1, 0])
        cp_in = st.number_input("Chest Pain (cp)", value=1, step=1)
        trestbps_in = st.number_input("Resting BP (trestbps)", value=130, step=1)
        chol_in = st.number_input("Cholesterol (chol)", value=240, step=1)
    with c2:
        fbs_in = st.number_input("FBS > 120", value=0, step=1)
        restecg_in = st.number_input("Rest ECG", value=0, step=1)
        thalach_in = st.number_input("Max HR (thalach)", value=150, step=1)
        exang_in = st.number_input("Exang (0/1)", value=0, step=1)
    with c3:
        oldpeak_in = st.number_input("Oldpeak", value=1.0, step=0.1)
        slope_in = st.number_input("Slope (0,1,2)", value=1, step=1)
        ca_in = st.number_input("Major Vessels (ca)", value=0, step=1)
        thal_in = st.number_input("Thal (0,1,2,3)", value=2, step=1)

    raw_input = {
        "age": age_in, "sex": sex_in, "cp": cp_in, "trestbps": trestbps_in,
        "chol": chol_in, "fbs": fbs_in, "restecg": restecg_in, "thalach": thalach_in,
        "exang": exang_in, "oldpeak": oldpeak_in, "slope": slope_in, "ca": ca_in, "thal": thal_in
    }

st.write("**Raw Input Dictionary Received**:")
st.json(raw_input)

# Section 2: Dynamic Validation & Edge Case Resolution
st.subheader("2. Dynamic Input Validation Diagnostics")
t_start = time.perf_counter_ns()
clean_dict, warnings_list, errors_list = validate_and_impute_input(raw_input, FEATURE_SPECS)
t_val_us = (time.perf_counter_ns() - t_start) / 1000.0  # microseconds

col_status, col_time = st.columns([3, 1])
with col_status:
    if not warnings_list and not errors_list:
        st.success("✅ **Validation Status: 100% Valid Clean Input**. All attributes strictly adhere to physiological specifications.")
    elif errors_list:
        st.error(f"❌ **Validation Status: Handled {len(errors_list)} Critical Error(s)** with fallback defaults and {len(warnings_list)} warning(s).")
    else:
        st.warning(f"⚠️ **Validation Status: Valid with {len(warnings_list)} Alert(s) / Automatic Imputation(s)**.")

with col_time:
    st.metric("Validation Latency", f"{t_val_us:.2f} µs")

if errors_list:
    st.write("**Resolved Fatal Errors (Fallback applied):**")
    for err in errors_list:
        st.markdown(f"- 🔴 `{err}`")

if warnings_list:
    st.write("**Warning & Imputation Audit Log:**")
    for warn in warnings_list:
        st.markdown(f"- 🟡 `{warn}`")

# Section 3: Real-Time Vectorized Inference
st.subheader("3. Real-Time Vectorized Inference Pipeline")
t_inf_start = time.perf_counter_ns()
vec_std = fast_vectorize_input(clean_dict, features, means, scales)
prob = float(model.predict_proba(vec_std[None, :])[0])
pred_class = int(prob >= 0.5)
t_inf_us = (time.perf_counter_ns() - t_inf_start) / 1000.0

col_res1, col_res2, col_res3 = st.columns(3)
with col_res1:
    risk_text = "HIGH RISK (Disease Indicated)" if pred_class == 1 else "LOW RISK (Healthy Profile)"
    delta_color = "inverse" if pred_class == 1 else "normal"
    st.metric("Model Classification", risk_text)

with col_res2:
    st.metric("Predicted Disease Probability", f"{prob * 100:.2f}%")

with col_res3:
    total_latency_us = t_val_us + t_inf_us
    st.metric("End-to-End Processing Latency", f"{total_latency_us:.2f} µs", f"Inf: {t_inf_us:.2f} µs")

# Parameter comparison table
comp_data = []
for f in features:
    raw_val = raw_input.get(f, "MISSING")
    clean_val = clean_dict[f]
    status = "Preserved" if raw_val == clean_val else ("Imputed" if raw_val == "MISSING" else "Clipped / Converted")
    comp_data.append({
        "Attribute": f,
        "Raw Input": str(raw_val),
        "Validated & Sanitized Value": f"{clean_val:.2f}" if isinstance(clean_val, float) else str(clean_val),
        "Validation Status": status
    })
st.dataframe(pd.DataFrame(comp_data).set_index("Attribute"), use_container_width=True)

# Section 4: Performance Benchmarking (DataFrame vs Vectorized Array)
st.subheader("4. Input Processing Optimization Benchmark")
st.write(
    "Benchmark 2,000 real-time inference cycles comparing legacy **Pandas DataFrame pipeline** "
    "versus optimized **Pre-allocated NumPy Vectorization**."
)

if st.button("🚀 Run Latency Benchmark (2,000 Iterations)"):
    n_iters = 2000
    
    # 1. Pandas DataFrame conversion
    t0 = time.perf_counter()
    for _ in range(n_iters):
        df_row = pd.DataFrame([clean_dict])[features]
        arr_pd = (df_row.to_numpy(dtype=float) - means) / scales
        _ = model.predict_proba(arr_pd)
    t_pd_total = (time.perf_counter() - t0) * 1000.0  # ms
    t_pd_per_call = (t_pd_total / n_iters) * 1000.0   # us

    # 2. Optimized NumPy Vectorizer
    t0 = time.perf_counter()
    for _ in range(n_iters):
        arr_np = fast_vectorize_input(clean_dict, features, means, scales)
        _ = model.predict_proba(arr_np[None, :])
    t_np_total = (time.perf_counter() - t0) * 1000.0  # ms
    t_np_per_call = (t_np_total / n_iters) * 1000.0   # us

    speedup = t_pd_per_call / max(t_np_per_call, 1e-6)

    m1, m2, m3 = st.columns(3)
    m1.metric("Pandas Latency", f"{t_pd_per_call:.2f} µs/req")
    m2.metric("Optimized NumPy Latency", f"{t_np_per_call:.2f} µs/req")
    m3.metric("Throughput Speedup", f"{speedup:.1f}x Faster! ⚡")

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    bars = ax.bar(["Pandas Pipeline", "Optimized Vectorized"], [t_pd_per_call, t_np_per_call], color=["#ef4444", "#10b981"], width=0.45)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2.0, h + max(t_pd_per_call * 0.02, 1), f"{h:.2f} µs", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Latency per Prediction (microseconds)")
    ax.set_title("Inference Latency: Legacy Pandas vs Vectorized NumPy")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)

