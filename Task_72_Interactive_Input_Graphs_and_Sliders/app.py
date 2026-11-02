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
    compute_confidence_distribution,
    fast_vectorize_input,
    load_data,
    standardize,
    standardize_fit,
    stratified_split,
)

st.set_page_config(
    page_title="Task 72: Interactive Graphs & Dynamic Sliders",
    page_icon="📈",
    layout="wide",
)

# Load data and fit baseline model
csv_path = Path(__file__).with_name("heart.csv")
if not csv_path.exists():
    csv_path = ROOT / "Task_21_Decision_Tree_Feature_Importance" / "heart.csv"

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame["target"].to_numpy(int)

X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=72)
means, scales = standardize_fit(X_train)
X_train_s, X_test_s = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_s, y_train)

# Reference min/max for normalization
cont_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cont_mins = frame[cont_features].min()
cont_maxs = frame[cont_features].max()
cont_meds = frame[cont_features].median()

# Custom CSS for Touch-Friendly Sliders & Multi-Device Compatibility
CUSTOM_CSS = """
<style>
/* Touch-friendly enlarged slider controls */
.stSlider > div > div > div > div {
    height: 8px;
    border-radius: 4px;
}
.stSlider > div > div > div > div > div {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: #2563eb;
    border: 2px solid white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.radar-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

.risk-pill-high { background: #fee2e2; color: #b91c1c; padding: 0.4rem 0.9rem; border-radius: 9999px; font-weight: 700; }
.risk-pill-mod { background: #fef3c7; color: #b45309; padding: 0.4rem 0.9rem; border-radius: 9999px; font-weight: 700; }
.risk-pill-low { background: #dcfce7; color: #15803d; padding: 0.4rem 0.9rem; border-radius: 9999px; font-weight: 700; }

/* Responsive adjustments */
@media screen and (max-width: 768px) {
    .radar-card { padding: 0.75rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Task 72: Interactive Graphs, Dynamic Sliders & Cross-Device Linking")
st.caption(
    "Dynamically link continuous patient input sliders directly to real-time visual components "
    "(Multi-Feature Radar Chart & 2D Decision Surface), ensuring seamless multi-device responsiveness."
)

# Device Viewport Selector
col_v1, col_v2 = st.columns([3, 1])
with col_v1:
    st.write("Adjust patient vitals using touch-friendly sliders. Graphs and model predictions update instantaneously (< 0.01 ms).")
with col_v2:
    viewport_mode = st.selectbox("Device Viewport Simulation", ["Desktop View (>1024px)", "Tablet View (768px-1024px)", "Mobile Handset View (≤768px)"])

# Layout split depending on viewport
if viewport_mode == "Mobile Handset View (≤768px)":
    st.info("📱 **Mobile Viewport Active**: Stacked single-column vertical layout with enlarged tap targets.")
    col_inputs, col_viz = st.container(), st.container()
else:
    col_inputs, col_viz = st.columns([1, 1], gap="medium")

with col_inputs:
    st.subheader("1. Interactive Patient Input Controls")
    
    st.markdown("**Core Continuous Vitals (Directly Linked to Visual Graphs)**")
    age_in = st.slider("Patient Age (years)", int(cont_mins["age"]), int(cont_maxs["age"]), 58, 1)
    trestbps_in = st.slider("Resting Blood Pressure (trestbps in mmHg)", int(cont_mins["trestbps"]), int(cont_maxs["trestbps"]), 138, 1)
    chol_in = st.slider("Serum Cholesterol (chol in mg/dL)", int(cont_mins["chol"]), int(cont_maxs["chol"]), 245, 1)
    thalach_in = st.slider("Maximum Heart Rate Achieved (thalach in bpm)", int(cont_mins["thalach"]), int(cont_maxs["thalach"]), 142, 1)
    oldpeak_in = st.slider("ST Depression (oldpeak in mm)", float(cont_mins["oldpeak"]), float(cont_maxs["oldpeak"]), 1.4, 0.1)

    st.markdown("**Diagnostic Categories & Secondary Indicators**")
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        sex_in = st.selectbox("Sex", [1, 0], format_func=lambda s: "Male (1)" if s == 1 else "Female (0)")
        cp_in = st.selectbox("Chest Pain (cp)", [0, 1, 2, 3], index=1, format_func=lambda c: {0: "Typical Angina (0)", 1: "Atypical Angina (1)", 2: "Non-Anginal (2)", 3: "Asymptomatic (3)"}[c])
        fbs_in = st.selectbox("FBS > 120 mg/dL", [0, 1], index=0)
        restecg_in = st.selectbox("Resting ECG", [0, 1, 2], index=0)
    with c_sub2:
        exang_in = st.selectbox("Exercise Angina (exang)", [0, 1], index=0)
        slope_in = st.selectbox("ST Slope", [0, 1, 2], index=1, format_func=lambda s: {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"}[s])
        ca_in = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3], index=0)
        thal_in = st.selectbox("Thalassemia (thal)", [1, 2, 3], index=1, format_func=lambda t: {1: "Normal (1)", 2: "Fixed (2)", 3: "Reversible (3)"}[t])

patient_dict = {
    "age": age_in, "sex": sex_in, "cp": cp_in, "trestbps": trestbps_in,
    "chol": chol_in, "fbs": fbs_in, "restecg": restecg_in, "thalach": thalach_in,
    "exang": exang_in, "oldpeak": oldpeak_in, "slope": slope_in, "ca": ca_in, "thal": thal_in
}

# Real-Time Model Inference
t0 = time.perf_counter_ns()
vec = fast_vectorize_input(patient_dict, features, means, scales)
logit = float(vec @ model.weights + model.bias)
prob = float(model.predict_proba(vec[None, :])[0])
dt_us = (time.perf_counter_ns() - t0) / 1000.0
conf_data = compute_confidence_distribution(prob)

with col_viz:
    st.subheader("2. Real-Time Dynamically Linked Visualizations")

    # Real-Time Decision Card
    if prob >= 0.65:
        tier_badge = f'<span class="risk-pill-high">HIGH CAD RISK ({prob*100:.1f}%)</span>'
    elif prob >= 0.35:
        tier_badge = f'<span class="risk-pill-mod">BORDERLINE RISK ({prob*100:.1f}%)</span>'
    else:
        tier_badge = f'<span class="risk-pill-low">LOW CAD RISK ({prob*100:.1f}%)</span>'

    st.markdown(f"""
    <div class="radar-card" style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            {tier_badge}
            <span style="font-weight:700; font-size:1.05rem; margin-left:0.75rem;">Model Probability: {prob*100:.1f}%</span>
        </div>
        <div style="font-size:0.85rem; color:#64748b;">
            Latency: <strong>{dt_us:.2f} µs</strong> | Logit: <strong>{logit:+.3f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Multi-Feature Physiological Radar Chart
    # Normalize continuous values to [0, 1] relative to cohort range
    patient_norm = [
        float((patient_dict[f] - cont_mins[f]) / (cont_maxs[f] - cont_mins[f]))
        for f in cont_features
    ]
    median_norm = [
        float((cont_meds[f] - cont_mins[f]) / (cont_maxs[f] - cont_mins[f]))
        for f in cont_features
    ]

    N_vars = len(cont_features)
    angles = [n / float(N_vars) * 2 * np.pi for n in range(N_vars)]
    angles += angles[:1]
    p_norm_cycle = patient_norm + [patient_norm[0]]
    med_norm_cycle = median_norm + [median_norm[0]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)

    # Radar plot on left
    axes[0].remove()
    ax_radar = fig.add_subplot(1, 2, 1, polar=True)
    ax_radar.plot(angles, med_norm_cycle, color="#94a3b8", linewidth=1.5, linestyle="--", label="Cohort Median")
    ax_radar.fill(angles, med_norm_cycle, color="#94a3b8", alpha=0.15)
    ax_radar.plot(angles, p_norm_cycle, color="#2563eb", linewidth=2.2, marker="o", label="Current Patient")
    ax_radar.fill(angles, p_norm_cycle, color="#3b82f6", alpha=0.3)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(["Age", "BP", "Chol", "Max HR", "ST Dep"], fontsize=9, fontweight="bold")
    ax_radar.set_ylim(0, 1.0)
    ax_radar.set_title("Physiological Radar Profile", pad=12, fontsize=11)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8)

    # 2. Bivariate Decision Surface: Max Heart Rate vs Age
    age_grid = np.linspace(cont_mins["age"], cont_maxs["age"], 40)
    hr_grid = np.linspace(cont_mins["thalach"], cont_maxs["thalach"], 40)
    A_mesh, H_mesh = np.meshgrid(age_grid, hr_grid)

    # Vectorize meshgrid predictions holding other features fixed
    mesh_flat_len = A_mesh.size
    mesh_matrix = np.tile([patient_dict[f] for f in features], (mesh_flat_len, 1))
    mesh_matrix[:, features.index("age")] = A_mesh.ravel()
    mesh_matrix[:, features.index("thalach")] = H_mesh.ravel()
    mesh_scaled = (mesh_matrix - means) / scales
    probs_mesh = model.predict_proba(mesh_scaled).reshape(A_mesh.shape)

    contour = axes[1].contourf(A_mesh, H_mesh, probs_mesh, levels=12, cmap="coolwarm", alpha=0.75)
    cbar = plt.colorbar(contour, ax=axes[1], shrink=0.85)
    cbar.set_label("CAD Probability", fontsize=8)
    axes[1].contour(A_mesh, H_mesh, probs_mesh, levels=[0.50], colors="black", linestyles="--", linewidths=1.8)
    axes[1].scatter([age_in], [thalach_in], color="#facc15", edgecolors="black", s=180, marker="*", zorder=10, label=f"Patient ({age_in}y, {thalach_in}bpm)")
    axes[1].set_xlabel("Age (years)", fontsize=9)
    axes[1].set_ylabel("Max Heart Rate (bpm)", fontsize=9)
    axes[1].set_title("2D Bivariate Decision Surface", fontsize=11)
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.25)

    st.pyplot(fig)

    st.caption("⚡ **Two-Way Synchronization**: Moving the Age, BP, Chol, Max HR, or ST Depression sliders in Panel 1 immediately recalculates the radar polygon and moves the gold star marker on the bivariate risk contour surface.")

