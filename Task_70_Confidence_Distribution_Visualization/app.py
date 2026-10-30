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
    page_title="Task 70: Confidence Distribution & Interactive Visualization",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for Animations & Visual Confidence Bars
ANIMATION_CSS = """
<style>
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); }
    100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}

@keyframes barGrow {
    from { width: 0%; }
    to { width: 100%; }
}

.prob-container {
    background: #f1f5f9;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}

.dual-bar-wrap {
    height: 38px;
    border-radius: 9999px;
    background: #e2e8f0;
    overflow: hidden;
    display: flex;
    margin: 1rem 0;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
}

.bar-healthy {
    background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    color: white;
    font-weight: 700;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
}

.bar-disease {
    background: linear-gradient(90deg, #f87171 0%, #ef4444 100%);
    color: white;
    font-weight: 700;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
}

.tier-badge {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.tier-high { background: #fee2e2; color: #b91c1c; }
.tier-mod { background: #fef3c7; color: #b45309; }
.tier-low { background: #dcfce7; color: #15803d; }
.tier-borderline { background: #f3e8ff; color: #7e22ce; animation: pulseGlow 2s infinite; }

.uncertainty-box {
    background: #faf5ff;
    border: 1px solid #d8b4fe;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #581c87;
    margin-top: 0.75rem;
}
</style>
"""
st.markdown(ANIMATION_CSS, unsafe_allow_html=True)

st.title("Task 70: Prediction Confidence Distribution & Interactive Exploration")
st.caption(
    "Manually compute probability distributions, visualize prediction confidence using animated dual-class bars, "
    "and interactively explore classification thresholds and clinical abstention margins."
)

# Load data and fit model
csv_path = Path(__file__).with_name("heart.csv")
if not csv_path.exists():
    csv_path = ROOT / "Task_21_Decision_Tree_Feature_Importance" / "heart.csv"

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame["target"].to_numpy(int)

X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=70)
means, scales = standardize_fit(X_train)
X_train_s, X_test_s = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_s, y_train)

# Section 1: Patient Attribute Input Controls
st.subheader("1. Interactive Patient Profile Tuning")
st.write("Tune continuous and discrete patient metrics to observe immediate probability shifts.")

c1, c2, c3 = st.columns(3)
with c1:
    age_val = st.slider("Age (years)", 29, 77, 57, 1)
    sex_val = st.selectbox("Sex", [1, 0], format_func=lambda s: "Male (1)" if s == 1 else "Female (0)")
    cp_val = st.selectbox("Chest Pain (cp)", [0, 1, 2, 3], index=1)
    trestbps_val = st.slider("Resting Blood Pressure (trestbps in mmHg)", 90, 200, 138, 1)
with c2:
    chol_val = st.slider("Serum Cholesterol (chol in mg/dL)", 130, 500, 246, 1)
    fbs_val = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1], index=0)
    restecg_val = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=0)
    thalach_val = st.slider("Max Heart Rate (thalach in bpm)", 70, 205, 142, 1)
with c3:
    exang_val = st.selectbox("Exercise-Induced Angina (exang)", [0, 1], index=0)
    oldpeak_val = st.slider("ST Depression (oldpeak in mm)", 0.0, 6.0, 1.4, 0.1)
    slope_val = st.selectbox("Slope of ST Segment", [0, 1, 2], index=1)
    ca_val = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3], index=0)
    thal_val = st.selectbox("Thalassemia (thal)", [1, 2, 3], index=1)

patient_dict = {
    "age": age_val, "sex": sex_val, "cp": cp_val, "trestbps": trestbps_val,
    "chol": chol_val, "fbs": fbs_val, "restecg": restecg_val, "thalach": thalach_val,
    "exang": exang_val, "oldpeak": oldpeak_val, "slope": slope_val, "ca": ca_val, "thal": thal_val
}

# Manual Probability & Distribution Computation
vec = fast_vectorize_input(patient_dict, features, means, scales)
logit = float(vec @ model.weights + model.bias)
prob_presence = 1.0 / (1.0 + np.exp(-np.clip(logit, -35.0, 35.0)))
conf_data = compute_confidence_distribution(prob_presence)

# Section 2: Interactive Decision Threshold & Abstention Zone Controls
st.subheader("2. Decision Threshold & Clinical Abstention Zone")
t_col1, t_col2 = st.columns(2)
with t_col1:
    threshold = st.slider("Classification Decision Threshold (τ)", 0.10, 0.90, 0.50, 0.05)
    st.caption("Standard default is 0.50. Lower thresholds increase sensitivity (detecting more cases); higher thresholds increase specificity.")

with t_col2:
    abstention_margin = st.slider("Clinical Abstention Band (±δ around 0.50)", 0.00, 0.25, 0.10, 0.02)
    st.caption("Uncertainty band where the model recommends physician review rather than autonomous prediction.")

# Evaluate classification vs threshold & abstention
p1 = conf_data["p1"]
p0 = conf_data["p0"]
is_in_gray_zone = abs(p1 - 0.50) <= abstention_margin

if is_in_gray_zone:
    decision_text = "⚠️ ABSTAIN (Physician Review Mandated)"
    decision_color = "#7e22ce"
    tier_class = "tier-borderline"
elif p1 >= threshold:
    decision_text = "DISEASE INDICATED (1)"
    decision_color = "#dc2626"
    tier_class = "tier-high"
else:
    decision_text = "HEALTHY PROFILE (0)"
    decision_color = "#16a34a"
    tier_class = "tier-low"

# Section 3: Animated Probability Distribution Bar & Metric Dashboard
st.subheader("3. Real-Time Probability Distribution & Animated Visual Bar")

p0_pct = p0 * 100.0
p1_pct = p1 * 100.0

HTML_DUAL_BAR = f"""
<div class="prob-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <span class="tier-badge {tier_class}">{conf_data['tier']}</span>
            <span style="font-weight: 700; font-size: 1.1rem; margin-left: 0.75rem;">Model Decision: <span style="color: {decision_color};">{decision_text}</span></span>
        </div>
        <div style="font-size: 0.9rem; color: #475569;">
            Raw Logit (z): <strong>{logit:+.3f}</strong> | Margin Δ: <strong>{conf_data['margin']:.3f}</strong> | Entropy H: <strong>{conf_data['entropy']:.3f} bits</strong>
        </div>
    </div>

    <!-- Animated Dual Probability Bar -->
    <div class="dual-bar-wrap">
        <div class="bar-healthy" style="width: {p0_pct:.1f}%;">
            {f"Healthy {p0_pct:.1f}%" if p0_pct >= 15 else ""}
        </div>
        <div class="bar-disease" style="width: {p1_pct:.1f}%;">
            {f"Disease {p1_pct:.1f}%" if p1_pct >= 15 else ""}
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #64748b;">
        <span>Absence P(Y=0): <strong>{p0:.4f}</strong></span>
        <span>Decision Threshold τ: <strong>{threshold:.2f}</strong></span>
        <span>Presence P(Y=1): <strong>{p1:.4f}</strong></span>
    </div>

    {"<div class='uncertainty-box'><strong>⚠️ Clinical Abstention Notice:</strong> Model probability is within the ±" + f"{abstention_margin:.2f}" + " uncertainty band of the decision boundary. High epistemic entropy (" + f"{conf_data['entropy']:.3f}" + " bits). Automated diagnosis suspended.</div>" if is_in_gray_zone else ""}
</div>
"""
st.markdown(HTML_DUAL_BAR, unsafe_allow_html=True)

# Side-by-side Matplotlib visualization
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)

# Plot 1: Discrete Probability Distribution Bar Chart
classes = ["Absence (0)\nHealthy", "Presence (1)\nDisease"]
probs = [p0, p1]
colors = ["#10b981", "#ef4444"]
bars = axes[0].bar(classes, probs, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
for b, p in zip(bars, probs):
    axes[0].text(b.get_x() + b.get_width() / 2.0, p + 0.02, f"{p * 100:.1f}%\n({p:.4f})", ha="center", va="bottom", fontweight="bold")
axes[0].axhline(threshold, color="#2563eb", linestyle="--", linewidth=1.5, label=f"Threshold τ = {threshold:.2f}")
if abstention_margin > 0:
    axes[0].axhspan(0.50 - abstention_margin, 0.50 + abstention_margin, color="#a855f7", alpha=0.15, label=f"Abstention Band (±{abstention_margin:.2f})")
axes[0].set_ylim(0, 1.18)
axes[0].set_ylabel("Posterior Class Probability")
axes[0].set_title("Manually Computed Probability Distribution")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(axis="y", alpha=0.25)

# Plot 2: Shannon Entropy & Uncertainty Curve
prob_axis = np.linspace(0.001, 0.999, 200)
entropy_curve = -(prob_axis * np.log2(prob_axis) + (1 - prob_axis) * np.log2(1 - prob_axis))
axes[1].plot(prob_axis, entropy_curve, color="#6366f1", linewidth=2.5, label="Shannon Entropy H(P)")
axes[1].scatter([p1], [conf_data["entropy"]], color="#dc2626", s=100, zorder=5, label=f"Current Patient (H={conf_data['entropy']:.2f}b)")
axes[1].axvline(0.50, color="#64748b", linestyle=":", label="Boundary (P=0.50, Max Entropy)")
axes[1].set_xlabel("Disease Probability P(Y=1)")
axes[1].set_ylabel("Entropy H(P) in bits")
axes[1].set_title("Epistemic Uncertainty vs Decision Margin")
axes[1].legend(loc="lower center", fontsize=8)
axes[1].grid(alpha=0.25)

st.pyplot(fig)

# Section 4: Live Stepped Clinical Risk Trajectory Simulator
st.subheader("4. Clinical Risk Trajectory Simulator (Dynamic Stepped Animation)")
st.write(
    "Simulate an escalating clinical stress episode (e.g. onset of acute ischemia during a treadmill test: "
    "rising systolic BP, heart rate elevation, followed by ST depression and severe angina)."
)

if st.button("▶️ Run Clinical Stress Trajectory Animation"):
    anim_placeholder = st.empty()
    steps = [
        {"desc": "Resting Baseline", "bp": 120, "hr": 75, "oldpeak": 0.0, "exang": 0, "cp": 1},
        {"desc": "Stage 1 Exercise Warmup", "bp": 138, "hr": 110, "oldpeak": 0.4, "exang": 0, "cp": 1},
        {"desc": "Stage 2 Moderate Exertion", "bp": 155, "hr": 140, "oldpeak": 1.1, "exang": 0, "cp": 1},
        {"desc": "Stage 3 Peak Exertion with Early Ischemia", "bp": 172, "hr": 162, "oldpeak": 2.2, "exang": 1, "cp": 0},
        {"desc": "Stage 4 Severe Angina & Deep ST Depression", "bp": 190, "hr": 175, "oldpeak": 3.8, "exang": 1, "cp": 0},
    ]

    for step_idx, st_data in enumerate(steps, start=1):
        temp_dict = patient_dict.copy()
        temp_dict["trestbps"] = st_data["bp"]
        temp_dict["thalach"] = st_data["hr"]
        temp_dict["oldpeak"] = st_data["oldpeak"]
        temp_dict["exang"] = st_data["exang"]
        temp_dict["cp"] = st_data["cp"]

        t_vec = fast_vectorize_input(temp_dict, features, means, scales)
        t_prob = float(model.predict_proba(t_vec[None, :])[0])
        t_conf = compute_confidence_distribution(t_prob)

        anim_html = f"""
        <div class="prob-container" style="border-left: 6px solid #2563eb;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>Step {step_idx}/5: {st_data['desc']}</h4>
                <span class="tier-badge {'tier-high' if t_prob >= 0.5 else 'tier-low'}">{t_conf['tier']}</span>
            </div>
            <p>Vitals: Blood Pressure <strong>{st_data['bp']} mmHg</strong> | Max HR <strong>{st_data['hr']} bpm</strong> | ST Depression <strong>{st_data['oldpeak']} mm</strong> | Angina: <strong>{'Yes' if st_data['exang'] else 'No'}</strong></p>
            <div class="dual-bar-wrap">
                <div class="bar-healthy" style="width: {t_conf['p0']*100:.1f}%;">Healthy {t_conf['p0']*100:.1f}%</div>
                <div class="bar-disease" style="width: {t_conf['p1']*100:.1f}%;">Disease {t_conf['p1']*100:.1f}%</div>
            </div>
            <p style="font-size: 0.85rem; color: #64748b; margin: 0;">Calculated Disease Risk: <strong>{t_prob*100:.1f}%</strong> | Uncertainty Entropy: <strong>{t_conf['entropy']:.3f} bits</strong></p>
        </div>
        """
        anim_placeholder.markdown(anim_html, unsafe_allow_html=True)
        time.sleep(0.4)

