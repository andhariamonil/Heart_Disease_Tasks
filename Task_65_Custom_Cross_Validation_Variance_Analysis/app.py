import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import (
    DecisionTreeScratch,
    LogisticRegressionScratch,
    RandomForestScratch,
    custom_kfold_split,
    custom_stratified_kfold_split,
    load_data,
    metrics,
    standardize,
    standardize_fit,
)

st.title("Task 65: Custom Cross-Validation & Variance Analysis")
st.caption(
    "Evaluate custom k-fold and stratified split mechanisms from scratch, ensuring balanced class "
    "representation and analyzing model performance variance across iterations and fold counts."
)

csv_path = Path(__file__).with_name("heart.csv")
if not csv_path.exists():
    csv_path = ROOT / "Task_21_Decision_Tree_Feature_Importance" / "heart.csv"

_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame["target"].to_numpy(int)
n_samples = len(y)
dataset_pos_rate = float(y.mean())

st.write(
    f"**Dataset support**: {n_samples} deduplicated patient records (13 features). "
    f"Overall positive class rate: **{dataset_pos_rate * 100:.2f}%** ({int(y.sum())} positive / {n_samples - int(y.sum())} negative)."
)

# Controls
k_folds = st.slider("Number of folds (k)", min_value=3, max_value=20, value=5, step=1)
seed = st.slider("Random seed", min_value=1, max_value=100, value=42, step=1)
model_choice = st.selectbox("Classifier model", ["Logistic Regression (Scratch)", "Decision Tree (Scratch)", "Random Forest (Scratch)"])

def run_evaluation(split_type: str, k: int, current_seed: int):
    if split_type == "Stratified K-Fold":
        splits = custom_stratified_kfold_split(y, k=k, shuffle=True, seed=current_seed)
    else:
        splits = custom_kfold_split(n_samples, k=k, shuffle=True, seed=current_seed)

    records = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if model_choice == "Logistic Regression (Scratch)":
            means, scales = standardize_fit(X_tr)
            X_tr_s = standardize(X_tr, means, scales)
            X_va_s = standardize(X_va, means, scales)
            clf = LogisticRegressionScratch(learning_rate=0.06, iterations=1200, l2=0.02).fit(X_tr_s, y_tr)
            preds = clf.predict(X_va_s)
        elif model_choice == "Decision Tree (Scratch)":
            clf = DecisionTreeScratch(max_depth=4, min_samples_split=6).fit(X_tr, y_tr, features)
            preds = clf.predict(X_va)
        else:
            clf = RandomForestScratch(tree_count=15, max_depth=4, sample_ratio=0.8, seed=current_seed + fold_idx).fit(X_tr, y_tr, features)
            preds = clf.predict(X_va)

        score = metrics(y_va, preds)
        val_pos_pct = float(y_va.mean()) * 100.0

        records.append({
            "Fold": fold_idx,
            "Train Size": len(train_idx),
            "Val Size": len(val_idx),
            "Val Pos Rate (%)": val_pos_pct,
            "Accuracy": score["accuracy"],
            "Precision": score["precision"],
            "Recall": score["recall"],
            "F1": score["f1"],
        })
    return pd.DataFrame(records)

df_strat = run_evaluation("Stratified K-Fold", k_folds, seed)
df_unstrat = run_evaluation("Standard K-Fold", k_folds, seed)

st.subheader(f"Fold-Level Results (Stratified k={k_folds})")
st.dataframe(df_strat.style.format({
    "Val Pos Rate (%)": "{:.2f}%",
    "Accuracy": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}",
    "F1": "{:.4f}",
}))

# Statistical Variance Analysis
def compute_stats(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    k = len(df)
    for col in metric_cols:
        vals = df[col].to_numpy()
        mean_val = float(np.mean(vals))
        var_val = float(np.var(vals, ddof=1)) if k > 1 else 0.0
        std_val = float(np.std(vals, ddof=1)) if k > 1 else 0.0
        se_val = std_val / np.sqrt(k) if k > 0 else 0.0
        ci_lower = mean_val - 1.96 * se_val
        ci_upper = mean_val + 1.96 * se_val
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        rows.append({
            "Metric": col,
            "Mean": mean_val,
            "Variance (σ²)": var_val,
            "Std Dev (σ)": std_val,
            "Std Error (SE)": se_val,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper,
            "Min": min_val,
            "Max": max_val,
            "Range": max_val - min_val,
        })
    return pd.DataFrame(rows).set_index("Metric")

metric_cols = ["Accuracy", "Precision", "Recall", "F1", "Val Pos Rate (%)"]
stats_strat = compute_stats(df_strat, metric_cols)
stats_unstrat = compute_stats(df_unstrat, metric_cols)

st.subheader(f"Performance Variance Across Iterations (k={k_folds})")
col_tabs = ["Stratified K-Fold", "Standard (Unstratified) K-Fold"]
selected_tab = st.radio("Select Split Protocol for Detailed Statistics:", col_tabs, horizontal=True)

if selected_tab == "Stratified K-Fold":
    st.dataframe(stats_strat.style.format("{:.4f}"))
else:
    st.dataframe(stats_unstrat.style.format("{:.4f}"))

# Comparison & Diagnostic Plots
st.subheader("Visual Diagnostics: Class Balance & Fold Stability")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

# 1. Class representation across folds
fold_ids = df_strat["Fold"].to_numpy()
axes[0].bar(fold_ids - 0.2, df_strat["Val Pos Rate (%)"], width=0.4, label="Stratified K-Fold", color="#2563eb", alpha=0.85)
axes[0].bar(fold_ids + 0.2, df_unstrat["Val Pos Rate (%)"], width=0.4, label="Standard K-Fold", color="#f97316", alpha=0.85)
axes[0].axhline(dataset_pos_rate * 100, color="crimson", linestyle="--", linewidth=1.5, label=f"True Class Prior ({dataset_pos_rate*100:.1f}%)")
axes[0].set_xlabel("Fold Index")
axes[0].set_ylabel("Positive Class % in Validation")
axes[0].set_title("Fold Class Balance Representation")
axes[0].set_ylim(0, 100)
axes[0].legend(loc="lower right", fontsize=8)
axes[0].grid(alpha=0.25)

# 2. Metric Variance Boxplots (Stratified vs Unstratified)
box_data = [
    df_strat["F1"].to_numpy(),
    df_unstrat["F1"].to_numpy(),
    df_strat["Accuracy"].to_numpy(),
    df_unstrat["Accuracy"].to_numpy(),
]
axes[1].boxplot(box_data, tick_labels=["Strat F1", "Unstrat F1", "Strat Acc", "Unstrat Acc"], patch_artist=True,
                boxprops=dict(facecolor="#e0e7ff", color="#3730a3"),
                medianprops=dict(color="#b91c1c", linewidth=1.5))
axes[1].set_ylabel("Metric Value")
axes[1].set_title("Performance Dispersion Across Folds")
axes[1].grid(alpha=0.25)

# 3. Multi-k Variance Exploration (k=3, 5, 10, 20)
k_values = [3, 5, 10, 20]
k_f1_means = []
k_f1_vars = []
for k_val in k_values:
    res = run_evaluation("Stratified K-Fold", k_val, seed)
    k_f1_means.append(res["F1"].mean())
    k_f1_vars.append(res["F1"].var(ddof=1) if k_val > 1 else 0.0)

ax3_twin = axes[2].twinx()
p1 = axes[2].plot(k_values, k_f1_means, marker="o", color="#059669", linewidth=2, label="Mean F1")
p2 = ax3_twin.plot(k_values, k_f1_vars, marker="s", color="#dc2626", linestyle="--", linewidth=2, label="Fold Variance (σ²)")
axes[2].set_xlabel("Fold Count (k)")
axes[2].set_ylabel("Mean F1-Score", color="#059669")
ax3_twin.set_ylabel("Fold Variance σ²", color="#dc2626")
axes[2].set_title("Bias-Variance Tradeoff across k")
lines = p1 + p2
labels = [l.get_label() for l in lines]
axes[2].legend(lines, labels, loc="upper right", fontsize=8)
axes[2].grid(alpha=0.25)

st.pyplot(fig)

st.info(
    f"**Statistical Findings Summary**:\n"
    f"- **Class Balance Impact**: Standard K-Fold generates validation positive rates ranging from "
    f"{df_unstrat['Val Pos Rate (%)'].min():.1f}% to {df_unstrat['Val Pos Rate (%)'].max():.1f}% "
    f"(Range: {df_unstrat['Val Pos Rate (%)'].max() - df_unstrat['Val Pos Rate (%)'].min():.1f}%), "
    f"distorting fold evaluation. Stratified K-Fold strictly bounds positive rate variation to "
    f"[{df_strat['Val Pos Rate (%)'].min():.1f}%, {df_strat['Val Pos Rate (%)'].max():.1f}%].\n"
    f"- **Variance Across Iterations**: For k={k_folds}, Stratified F1 has mean {stats_strat.loc['F1', 'Mean']:.4f} "
    f"with variance {stats_strat.loc['F1', 'Variance (σ²)']:.6f} and 95% CI [{stats_strat.loc['F1', '95% CI Lower']:.4f}, {stats_strat.loc['F1', '95% CI Upper']:.4f}].\n"
    f"- **Fold Count Selection**: Increasing k reduces training sample bias (each fold trains on {(k_folds-1)/k_folds*100:.1f}% data) "
    f"while smaller k exhibits slightly lower fold-to-fold correlation but higher pessimistic bias."
)

