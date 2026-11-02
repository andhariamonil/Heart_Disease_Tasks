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
    compute_linear_shap,
    fast_vectorize_input,
    load_data,
    standardize,
    standardize_fit,
    stratified_split,
)

st.set_page_config(
    page_title="Task 73: SHAP Interpretability UI & Attribution",
    page_icon="🔍",
    layout="wide",
)

# Load data and fit baseline model
csv_path = Path(__file__).with_name("heart.csv")
if not csv_path.exists():
    csv_path = ROOT / "Task_21_Decision_Tree_Feature_Importance" / "heart.csv"

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame["target"].to_numpy(int)

X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=73)
means, scales = standardize_fit(X_train)
X_train_s, X_test_s = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_s, y_train)

# Feature display names dictionary
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1=M, 0=F)",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar > 120",
    "restecg": "Resting ECG Results",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (mm)",
    "slope": "ST Segment Slope",
    "ca": "Major Fluoroscopy Vessels",
    "thal": "Thalassemia Defect",
}

# Custom CSS
CUSTOM_CSS = """
<style>
.shap-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    margin-bottom: 1.25rem;
}
.pill-pos { background: #fee2e2; color: #b91c1c; font-weight: 700; padding: 0.25rem 0.6rem; border-radius: 4px; }
.pill-neg { background: #dcfce7; color: #15803d; font-weight: 700; padding: 0.25rem 0.6rem; border-radius: 4px; }
.pill-base { background: #f1f5f9; color: #475569; font-weight: 600; padding: 0.25rem 0.6rem; border-radius: 4px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Task 73: SHAP Interpretability UI & Feature Attribution Exploration")
st.caption(
    "Manually compute exact Shapley Additive Explanations (SHAP), visualize positive and negative feature attributions "
    "via interactive waterfall and force charts, and evaluate transparent clinical decision explanations."
)

# Section 1: Patient Configuration & Presets
st.subheader("1. Patient Clinical Profile Configuration")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

p_init = None
if col_p1.button("🚨 Case 1: High-Risk CAD Profile"):
    p_init = {"age": 64, "sex": 1, "cp": 0, "trestbps": 160, "chol": 295, "fbs": 1, "restecg": 2, "thalach": 122, "exang": 1, "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 3}
if col_p2.button("🌿 Case 2: Healthy Low-Risk Profile"):
    p_init = {"age": 39, "sex": 0, "cp": 2, "trestbps": 115, "chol": 178, "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2}
if col_p3.button("⚖️ Case 3: Borderline Ambiguous Case"):
    p_init = {"age": 57, "sex": 1, "cp": 1, "trestbps": 136, "chol": 238, "fbs": 0, "restecg": 1, "thalach": 146, "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 2}
if col_p4.button("🔄 Reset to Default"):
    p_init = {"age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 240, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}

if p_init is not None:
    st.session_state.p_shap = p_init

if "p_shap" not in st.session_state:
    st.session_state.p_shap = {"age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 240, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}

pt = st.session_state.p_shap

c1, c2, c3 = st.columns(3)
with c1:
    age_in = st.slider("Age (years)", 25, 80, int(pt["age"]))
    sex_in = st.selectbox("Sex", [1, 0], index=0 if pt["sex"] == 1 else 1, format_func=lambda s: "Male (1)" if s == 1 else "Female (0)")
    cp_in = st.selectbox("Chest Pain (cp)", [0, 1, 2, 3], index=int(pt["cp"]), format_func=lambda c: {0: "Typical Angina (0)", 1: "Atypical Angina (1)", 2: "Non-Anginal (2)", 3: "Asymptomatic (3)"}[c])
    trestbps_in = st.slider("Resting BP (trestbps in mmHg)", 90, 200, int(pt["trestbps"]))
    chol_in = st.slider("Cholesterol (chol in mg/dL)", 120, 500, int(pt["chol"]))
with c2:
    fbs_in = st.selectbox("FBS > 120 mg/dL", [0, 1], index=int(pt["fbs"]))
    restecg_in = st.selectbox("Resting ECG", [0, 1, 2], index=int(pt["restecg"]))
    thalach_in = st.slider("Max Heart Rate (thalach in bpm)", 70, 210, int(pt["thalach"]))
    exang_in = st.selectbox("Exercise Angina (exang)", [0, 1], index=int(pt["exang"]), format_func=lambda e: "No (0)" if e == 0 else "Yes (1)")
with c3:
    oldpeak_in = st.slider("ST Depression (oldpeak in mm)", 0.0, 6.0, float(pt["oldpeak"]), 0.1)
    slope_in = st.selectbox("ST Slope", [0, 1, 2], index=int(pt["slope"]), format_func=lambda s: {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"}[s])
    ca_in = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3], index=min(int(pt["ca"]), 3))
    thal_in = st.selectbox("Thalassemia (thal)", [1, 2, 3], index=max(int(pt["thal"]) - 1, 0), format_func=lambda t: {1: "Normal (1)", 2: "Fixed (2)", 3: "Reversible (3)"}[t])

curr_patient = {
    "age": age_in, "sex": sex_in, "cp": cp_in, "trestbps": trestbps_in,
    "chol": chol_in, "fbs": fbs_in, "restecg": restecg_in, "thalach": thalach_in,
    "exang": exang_in, "oldpeak": oldpeak_in, "slope": slope_in, "ca": ca_in, "thal": thal_in
}

# Manual SHAP Computation
raw_vec = np.array([curr_patient[f] for f in features], dtype=float)
scaled_vec = (raw_vec - means) / scales
shap_res = compute_linear_shap(model, scaled_vec, features)

base_logit = shap_res["base_logit"]
base_prob = shap_res["base_prob"]
total_logit = shap_res["total_logit"]
pred_prob = shap_res["pred_prob"]
ranked = shap_res["ranked_features"]

# Section 2: Model Decision & Transparency Overview
st.subheader("2. Model Decision & Additive Decomposition Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Base Rate E[f(X)]", f"{base_prob * 100:.1f}%", f"Logit: {base_logit:+.3f}")
k2.metric("Final Predicted Risk", f"{pred_prob * 100:.1f}%", f"Logit: {total_logit:+.3f}")
k3.metric("Net SHAP Shift (∑ φ)", f"{total_logit - base_logit:+.3f}", "Sum of all 13 features")
k4.metric("Model Classification", "DISEASE (1)" if pred_prob >= 0.5 else "HEALTHY (0)", f"Margin: {abs(pred_prob - 0.5)*2:.2f}")

# Additivity Axiom Verification
sum_attributions = sum(r["attribution"] for r in ranked)
additivity_verified = np.isclose(base_logit + sum_attributions, total_logit)

if additivity_verified:
    st.success(f"✅ **Efficiency / Additivity Axiom Verified**: Base Logit ({base_logit:.4f}) + ∑ Attributions ({sum_attributions:+.4f}) = Final Logit ({total_logit:.4f}). Total transparency guaranteed.")

# Section 3: Interactive SHAP Waterfall Plot
st.subheader("3. Interactive SHAP Waterfall Visualization")
st.write(
    "Traces step-by-step how each clinical feature shifts the patient's logit score from the population baseline $E[f(X)]$ "
    "to the final prediction. Crimson bars push toward elevated risk; emerald bars pull toward health."
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# 1. Waterfall Plot (Cumulative accumulation on logit scale)
top_k = 9
top_ranked = ranked[:top_k]
other_sum = sum(r["attribution"] for r in ranked[top_k:])

waterfall_steps = [{"label": "Baseline E[f(X)]", "delta": base_logit, "cumulative": base_logit, "color": "#64748b"}]
running = base_logit
for r in top_ranked:
    running += r["attribution"]
    col = "#ef4444" if r["attribution"] > 0 else "#10b981"
    waterfall_steps.append({"label": f"{r['feature']} ({curr_patient[r['feature']]})", "delta": r["attribution"], "cumulative": running, "color": col})

if ranked[top_k:]:
    running += other_sum
    waterfall_steps.append({"label": f"4 Other Features", "delta": other_sum, "cumulative": running, "color": "#a855f7"})

waterfall_steps.append({"label": "Final Output f(x)", "delta": 0, "cumulative": running, "color": "#1e293b"})

labels = [s["label"] for s in waterfall_steps]
y_pos = np.arange(len(labels))[::-1]

# Plot horizontal waterfall bars
for idx, step in enumerate(waterfall_steps):
    y_idx = y_pos[idx]
    if idx == 0 or idx == len(waterfall_steps) - 1:
        # Baseline & Final output bars from 0
        axes[0].barh(y_idx, step["cumulative"], color=step["color"], height=0.55, align="center")
        axes[0].text(step["cumulative"] + (0.05 if step["cumulative"] >= 0 else -0.2), y_idx, f"{step['cumulative']:+.2f}", va="center", fontsize=8, fontweight="bold")
    else:
        # Floating delta bar
        start_x = step["cumulative"] - step["delta"]
        axes[0].barh(y_idx, step["delta"], left=start_x, color=step["color"], height=0.55, align="center")
        axes[0].text(step["cumulative"] + (0.05 if step["delta"] >= 0 else -0.2), y_idx, f"{step['delta']:+.2f}", va="center", fontsize=8)

axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(labels, fontsize=8.5)
axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
axes[0].set_xlabel("Log-Odds (Logit Scale)")
axes[0].set_title(f"SHAP Waterfall: Cumulative Risk Trajectory\n(Base: {base_prob*100:.1f}% -> Final: {pred_prob*100:.1f}%)", fontsize=10.5)
axes[0].grid(axis="x", alpha=0.25)

# 2. Diverging Feature Attribution Force Bar Chart (All 13 features)
all_feat_names = [f"{r['feature']} = {curr_patient[r['feature']]}" for r in ranked][::-1]
all_attr_vals = [r["attribution"] for r in ranked][::-1]
bar_colors = ["#ef4444" if v > 0 else "#10b981" for v in all_attr_vals]

axes[1].barh(np.arange(len(all_feat_names)), all_attr_vals, color=bar_colors, height=0.6)
axes[1].set_yticks(np.arange(len(all_feat_names)))
axes[1].set_yticklabels(all_feat_names, fontsize=8)
axes[1].axvline(0, color="black", linestyle="-", linewidth=1.2)
for idx, v in enumerate(all_attr_vals):
    axes[1].text(v + (0.02 if v >= 0 else -0.15), idx, f"{v:+.2f}", va="center", fontsize=7.5, fontweight="bold")
axes[1].set_xlabel("SHAP Value φ (Logit Impact)")
axes[1].set_title("Ranked Feature Attributions (Local Force Chart)", fontsize=10.5)
axes[1].grid(axis="x", alpha=0.25)

st.pyplot(fig)

# Section 4: "What-If" Counterfactual Intervention Explorer
st.subheader("4. Interactive 'What-If' Counterfactual Intervention Explorer")
st.write(
    "Simulate a targeted clinical treatment or lifestyle modification. "
    "Select an actionable risk factor to observe how intervention shrinks its SHAP contribution and lowers overall patient risk."
)

col_w1, col_w2 = st.columns([1, 2])
with col_w1:
    mod_feat = st.selectbox(
        "Select Actionable Risk Factor to Treat:",
        ["trestbps", "chol", "thalach", "oldpeak"],
        format_func=lambda f: FEATURE_LABELS[f]
    )
    current_val = float(curr_patient[mod_feat])
    spec = FEATURE_SPECS[mod_feat]
    target_val = st.slider(
        f"Simulate Target Clinical Value for {FEATURE_LABELS[mod_feat]}:",
        float(spec["min"]), float(spec["max"]), current_val,
        step=1.0 if spec["type"] is int else 0.1
    )

with col_w2:
    # Counterfactual computation
    cf_patient = curr_patient.copy()
    cf_patient[mod_feat] = target_val
    cf_vec = (np.array([cf_patient[f] for f in features], dtype=float) - means) / scales
    cf_shap = compute_linear_shap(model, cf_vec, features)

    orig_attr = shap_res["attributions"][mod_feat]
    new_attr = cf_shap["attributions"][mod_feat]
    attr_delta = new_attr - orig_attr
    prob_delta = (cf_shap["pred_prob"] - pred_prob) * 100.0

    st.markdown(f"""
    <div class="shap-card">
        <h4>Therapeutic Intervention Impact on {FEATURE_LABELS[mod_feat]}:</h4>
        <ul>
            <li>Original Baseline: <strong>{current_val}</strong> (SHAP φ = <code>{orig_attr:+.3f}</code>)</li>
            <li>Post-Treatment Target: <strong>{target_val}</strong> (SHAP φ = <code>{new_attr:+.3f}</code>)</li>
            <li>Attribution Shift (Δφ): <strong style="color:{'#15803d' if attr_delta < 0 else '#b91c1c'};">{attr_delta:+.3f}</strong></li>
            <li>Net Probability Shift: <strong style="color:{'#15803d' if prob_delta < 0 else '#b91c1c'};">{prob_delta:+.1f}%</strong> (New Risk: <strong>{cf_shap['pred_prob']*100:.1f}%</strong>)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Section 5: Automated Natural Language Clinical Explanation
st.subheader("5. Plain-Language Clinical Decision Explanation")

top_positive = [r for r in ranked if r["attribution"] > 0][:3]
top_negative = [r for r in ranked if r["attribution"] < 0][:2]

pos_text = ", ".join([f"**{FEATURE_LABELS[r['feature']]}** ({curr_patient[r['feature']]}, φ={r['attribution']:+.2f})" for r in top_positive]) or "none"
neg_text = ", ".join([f"**{FEATURE_LABELS[r['feature']]}** ({curr_patient[r['feature']]}, φ={r['attribution']:+.2f})" for r in top_negative]) or "none"

audit_summary = f"""
> **Automated Clinical Explanation for Patient Profile:**  
> The model predicts a **{pred_prob * 100:.1f}% probability of cardiovascular disease** (compared to the population base expectation of **{base_prob * 100:.1f}%**).  
> - **Primary Risk-Elevating Drivers:** Elevated risk is predominantly driven by {pos_text}.  
> - **Primary Protective Factors:** Counterbalancing protective influence is provided by {neg_text}.  
> - **Clinical Transparency Note:** All local Shapley attributions strictly satisfy the additivity axiom: $\\phi_0 + \\sum \\phi_i = f(x)$. Model decisions are fully explainable with no black-box residual error.
"""
st.markdown(audit_summary)

# Feature Attribution Data Table
st.write("##### Complete 13-Feature SHAP Breakdown Table:")
table_data = []
for r in ranked:
    table_data.append({
        "Attribute": FEATURE_LABELS[r["feature"]],
        "Patient Value": str(curr_patient[r["feature"]]),
        "Standardized (z-score)": f"{r['scaled_val']:+.2f}",
        "Linear Weight (w)": f"{r['weight']:+.3f}",
        "SHAP Attribution (φ)": f"{r['attribution']:+.3f}",
        "Impact Direction": r["direction"]
    })
st.dataframe(pd.DataFrame(table_data).set_index("Attribute"), use_container_width=True)

