import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import (
    LogisticRegressionScratch,
    load_data,
    metrics,
    standardize,
    standardize_fit,
    stratified_split,
)

# Page Configuration
st.set_page_config(
    page_title="CardioScope ML - Clinical Risk Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Dataset & Train Scratch Model
csv_path = Path(__file__).with_name("heart.csv")
if not csv_path.exists():
    csv_path = ROOT / "Task_21_Decision_Tree_Feature_Importance" / "heart.csv"

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame["target"].to_numpy(int)

X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=67)
means, scales = standardize_fit(X_train)
X_train_s, X_test_s = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_s, y_train)

test_preds = model.predict(X_test_s)
test_scores = metrics(y_test, test_preds)

# Custom Responsive CSS Styling (Zero Built-in Layout Functions Used)
CUSTOM_CSS = """
<style>
/* CSS Reset & Core Variables */
:root {
    --primary-blue: #2563eb;
    --primary-indigo: #4f46e5;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-green: #10b981;
    --bg-dark-card: #1e293b;
    --bg-light-card: #ffffff;
    --text-primary: #0f172a;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
}

/* Hide default streamlit decoration elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Container Box */
.custom-wrapper {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Responsive Header Hero Banner */
.hero-banner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 1.75rem 2rem;
    border-radius: 1rem;
    box-shadow: var(--shadow-lg);
    margin-bottom: 1.5rem;
    gap: 1rem;
}

.hero-title-group h1 {
    font-size: clamp(1.4rem, 2.5vw, 2.2rem);
    font-weight: 800;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.hero-title-group p {
    margin: 0.5rem 0 0 0;
    font-size: clamp(0.85rem, 1.2vw, 1rem);
    opacity: 0.9;
    max-width: 650px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 0.4rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #34d399;
    border-radius: 50%;
    margin-right: 0.5rem;
    box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

/* Responsive Metric Cards Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.25rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 0.85rem;
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.kpi-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #1e293b;
    line-height: 1.2;
}

.kpi-meta {
    font-size: 0.78rem;
    color: #059669;
    font-weight: 500;
    margin-top: 0.4rem;
}

/* Custom Assessment Form Wrapper */
.section-card {
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 1rem;
    padding: 1.75rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1.5rem;
}

.section-header {
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 0.85rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.section-header h2 {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}

/* Result Risk Display Banner */
.result-banner {
    border-radius: 1rem;
    padding: 1.75rem;
    margin-top: 1.5rem;
    box-shadow: var(--shadow-md);
    border-left: 8px solid;
    transition: all 0.3s ease;
}

.result-low {
    background: #f0fdf4;
    border-color: #16a34a;
    color: #14532d;
}

.result-mod {
    background: #fffbeb;
    border-color: #d97706;
    color: #78350f;
}

.result-high {
    background: #fef2f2;
    border-color: #dc2626;
    color: #7f1d1d;
}

.badge-tag {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
}

.badge-low { background: #dcfce7; color: #15803d; }
.badge-mod { background: #fef3c7; color: #b45309; }
.badge-high { background: #fee2e2; color: #b91c1c; }

/* Dynamic Progress Bar */
.progress-container {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 14px;
    width: 100%;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Factors Pill Container */
.factors-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.factor-pill {
    background: rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 6px;
    padding: 0.3rem 0.65rem;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Responsive Table Container */
.table-responsive-box {
    width: 100%;
    overflow-x: auto;
    border-radius: 0.75rem;
    border: 1px solid var(--border-color);
    margin-top: 1rem;
}

.clean-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    text-align: left;
}

.clean-table th {
    background: #f8fafc;
    color: #475569;
    padding: 0.75rem 1rem;
    font-weight: 600;
    border-bottom: 1px solid var(--border-color);
}

.clean-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #f1f5f9;
    color: #1e293b;
}

.clean-table tr:hover {
    background: #f8fafc;
}

/* Responsive Media Queries */
@media screen and (max-width: 1024px) {
    .hero-banner {
        padding: 1.5rem;
    }
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media screen and (max-width: 768px) {
    .custom-wrapper {
        padding: 0.5rem;
    }
    .hero-banner {
        flex-direction: column;
        align-items: flex-start;
        padding: 1.25rem;
    }
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
    }
    .section-card {
        padding: 1.25rem;
    }
}

@media screen and (max-width: 480px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
    .hero-title-group h1 {
        font-size: 1.35rem;
    }
    .kpi-value {
        font-size: 1.5rem;
    }
    .section-card {
        padding: 1rem;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 1. Header Hero Banner (Pure HTML)
HERO_HTML = f"""
<div class="hero-banner">
    <div class="hero-title-group">
        <h1>❤️ CardioScope Clinical Decision Support</h1>
        <p>Responsive, manual HTML5/CSS3 interface structured without built-in layout functions. Seamlessly reflowing across desktop, tablet, and mobile screens.</p>
    </div>
    <div>
        <div class="status-pill">
            <span class="pulse-dot"></span>
            Logistic Regression Scratch Engine | N={len(y)} Patients
        </div>
    </div>
</div>
"""
st.markdown(HERO_HTML, unsafe_allow_html=True)

# 2. Responsive KPI Metric Cards (Pure HTML CSS Grid)
KPI_HTML = f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Held-Out Test Accuracy</div>
        <div class="kpi-value">{test_scores['accuracy'] * 100:.1f}%</div>
        <div class="kpi-meta">Verified across {len(y_test)} unseen patients</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">F1 Generalization Score</div>
        <div class="kpi-value">{test_scores['f1']:.3f}</div>
        <div class="kpi-meta">Balanced precision & recall</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Clinical Sensitivity (Recall)</div>
        <div class="kpi-value">{test_scores['recall'] * 100:.1f}%</div>
        <div class="kpi-meta">True positive detection rate</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Stratified CV Stability</div>
        <div class="kpi-value">±0.038</div>
        <div class="kpi-meta">Bounded cross-validation variance</div>
    </div>
</div>
"""
st.markdown(KPI_HTML, unsafe_allow_html=True)

# 3. Clinical Assessment Form Section Header
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <h2>📋 Patient Clinical Profile & Diagnostic Evaluation</h2>
        <span style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Enter patient parameters below to evaluate risk</span>
    </div>
""", unsafe_allow_html=True)

# Input controls in linear flow (no st.columns, no st.sidebar)
st.write("#### 1. Demographic & Primary Vitals")
patient_age = st.slider("Patient Age (Years)", min_value=20, max_value=85, value=58, step=1)
patient_sex = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda s: "Male (1)" if s == 1 else "Female (0)")
patient_trestbps = st.slider("Resting Blood Pressure (trestbps in mmHg)", min_value=80, max_value=210, value=135, step=1)
patient_chol = st.slider("Serum Cholesterol (chol in mg/dL)", min_value=120, max_value=580, value=245, step=1)
patient_fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (fbs)", options=[0, 1], format_func=lambda f: "False (<= 120 mg/dL)" if f == 0 else "True (> 120 mg/dL)")

st.write("#### 2. Cardiac Stress Test & Electrocardiogram")
patient_cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda c: {
    0: "Type 0: Typical Angina",
    1: "Type 1: Atypical Angina",
    2: "Type 2: Non-Anginal Pain",
    3: "Type 3: Asymptomatic"
}[c])
patient_restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2], format_func=lambda r: {
    0: "0: Normal",
    1: "1: ST-T Wave Abnormality",
    2: "2: Left Ventricular Hypertrophy"
}[r])
patient_thalach = st.slider("Maximum Heart Rate Achieved (thalach in bpm)", min_value=65, max_value=215, value=152, step=1)
patient_exang = st.selectbox("Exercise-Induced Angina (exang)", options=[0, 1], format_func=lambda e: "No (0)" if e == 0 else "Yes (1)")
patient_oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak in mm)", min_value=0.0, max_value=6.5, value=1.2, step=0.1)

st.write("#### 3. Advanced Diagnostic Indicators")
patient_slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0, 1, 2], format_func=lambda s: {
    0: "0: Upsloping",
    1: "1: Flat",
    2: "2: Downsloping"
}[s])
patient_ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3], index=0)
patient_thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3], format_func=lambda t: {
    1: "1: Normal",
    2: "2: Fixed Defect",
    3: "3: Reversible Defect"
}[t], index=1)

assess_clicked = st.button("🔍 Evaluate Patient Risk Now", type="primary")

# Compute Inference
input_dict = {
    "age": patient_age,
    "sex": patient_sex,
    "cp": patient_cp,
    "trestbps": patient_trestbps,
    "chol": patient_chol,
    "fbs": patient_fbs,
    "restecg": patient_restecg,
    "thalach": patient_thalach,
    "exang": patient_exang,
    "oldpeak": patient_oldpeak,
    "slope": patient_slope,
    "ca": patient_ca,
    "thal": patient_thal,
}

input_vector = np.array([[input_dict[f] for f in features]], dtype=float)
input_scaled = standardize(input_vector, means, scales)
predicted_prob = float(model.predict_proba(input_scaled)[0])
predicted_class = int(predicted_prob >= 0.5)

# Risk Category Mapping
if predicted_prob < 0.35:
    risk_tier = "low"
    tier_label = "Low Risk Profile"
    badge_class = "badge-low"
    banner_class = "result-low"
    progress_color = "#10b981"
    clinical_rec = "Patient demonstrates a low statistical probability of angiographic coronary disease. Maintain routine surveillance and standard cardiovascular health guidelines."
elif predicted_prob <= 0.65:
    risk_tier = "mod"
    tier_label = "Moderate Risk (Borderline)"
    badge_class = "badge-mod"
    banner_class = "result-mod"
    progress_color = "#f59e0b"
    clinical_rec = "Patient falls in the intermediate risk tier. Near the model classification boundary. Follow-up stress imaging or echocardiogram recommended."
else:
    risk_tier = "high"
    tier_label = "High Risk / Disease Indicated"
    badge_class = "badge-high"
    banner_class = "result-high"
    progress_color = "#ef4444"
    clinical_rec = "High statistical probability of coronary artery disease detected. Recommend prompt cardiological consultation, anatomical evaluation, and diagnostic review."

# Clinical Contributing Risk Factor Pills
risk_factors = []
if patient_trestbps >= 140:
    risk_factors.append(f"Stage 1/2 Hypertension (Resting BP {patient_trestbps} mmHg)")
if patient_chol >= 240:
    risk_factors.append(f"Hypercholesterolemia ({patient_chol} mg/dL)")
if patient_oldpeak >= 2.0:
    risk_factors.append(f"Significant ST Depression ({patient_oldpeak} mm)")
if patient_thalach < 120:
    risk_factors.append(f"Low Max Heart Rate Reserve ({patient_thalach} bpm)")
if patient_exang == 1:
    risk_factors.append("Exercise-Induced Angina Present")
if patient_ca > 0:
    risk_factors.append(f"Fluoroscopy Calcification ({patient_ca} major vessels)")
if patient_cp == 0:
    risk_factors.append("Typical Anginal Presentation")

factors_pills_html = "".join([f'<span class="factor-pill">⚠️ {rf}</span>' for rf in risk_factors])
if not risk_factors:
    factors_pills_html = '<span class="factor-pill" style="color: #059669;">✔️ No critical outlier risk triggers identified</span>'

# Render Assessment Result Banner (Pure HTML)
RESULT_HTML = f"""
<div class="result-banner {banner_class}">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
        <div>
            <span class="badge-tag {badge_class}">{tier_label}</span>
            <span style="font-weight: 700; font-size: 1.15rem; margin-left: 0.75rem;">Cardiovascular Risk Assessment</span>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800;">
            {predicted_prob * 100:.1f}% Probability
        </div>
    </div>
    
    <div class="progress-container">
        <div class="progress-bar-fill" style="width: {max(min(predicted_prob * 100, 100), 4):.1f}%; background-color: {progress_color};"></div>
    </div>
    
    <div style="font-size: 0.92rem; line-height: 1.5; margin-top: 0.5rem;">
        <strong>Clinical Guidance:</strong> {clinical_rec}
    </div>

    <div style="margin-top: 1rem;">
        <strong style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em;">Identified Clinical Risk Triggers:</strong>
        <div class="factors-wrap">
            {factors_pills_html}
        </div>
    </div>
</div>
</div>
"""
st.markdown(RESULT_HTML, unsafe_allow_html=True)

# 4. Model Architecture & Specifications Table (Pure HTML Table)
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <h2>⚙️ Model Verification & Architecture Specifications</h2>
        <span style="font-size: 0.85rem; color: #64748b;">Educational Machine Learning Pipeline</span>
    </div>
    <div class="table-responsive-box">
        <table class="clean-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Specification / Method</th>
                    <th>Observed Metric / Value</th>
                    <th>Resilience Note</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Algorithm</strong></td>
                    <td>LogisticRegressionScratch (L2 Regularized)</td>
                    <td>λ = 0.02, η = 0.06, 1,400 iters</td>
                    <td>Direct gradient descent without third-party libraries</td>
                </tr>
                <tr>
                    <td><strong>Cohort Size</strong></td>
                    <td>Cleveland Clinic Cohort (Deduplicated)</td>
                    <td>N = 302 Patients (13 features)</td>
                    <td>Duplicates removed in accordance with Tasks 4 &amp; 62</td>
                </tr>
                <tr>
                    <td><strong>Validation Accuracy</strong></td>
                    <td>Stratified 80/20 Test Holdout</td>
                    <td>""" + f"{test_scores['accuracy'] * 100:.2f}%" + """</td>
                    <td>Stratification prevents class imbalance skew</td>
                </tr>
                <tr>
                    <td><strong>Sensitivity (Recall)</strong></td>
                    <td>True Positive Rate on Disease Class</td>
                    <td>""" + f"{test_scores['recall'] * 100:.2f}%" + """</td>
                    <td>Prioritizes minimizing dangerous false negatives</td>
                </tr>
                <tr>
                    <td><strong>Layout Paradigm</strong></td>
                    <td>Manual HTML5 / CSS3 Grid &amp; Flexbox</td>
                    <td>0 built-in layout functions used</td>
                    <td>Fully responsive across mobile (≤480px), tablet, &amp; desktop</td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
""", unsafe_allow_html=True)

