# Heart Disease Prediction Using Machine Learning

## Task 73: SHAP Interpretability UI & Attribution Exploration

### Objective

Display Shapley Additive Explanations (SHAP) directly within the user interface for transparent machine learning decision support. Manually compute exact feature attributions from first principles, build interactive waterfall and diverging force visualizations allowing clinicians to explore positive and negative risk contributors, and provide automated plain-language decision explanations for total clinical transparency.

Key Interpretability Capabilities:
1. **Manual Computation of Exact Feature Attributions**:
   - Implements exact additive feature decompositions on the logit scale:
     $$z(x) = \phi_0 + \sum_{j=1}^{13} \phi_j(x) = b + \sum_{j=1}^{13} w_j \tilde{x}_j$$
     where $\phi_0 = b$ is the baseline model expectation, and $\phi_j(x) = w_j \tilde{x}_j$ is the exact local attribution for attribute $j$.
   - Verifies the fundamental **Efficiency / Additivity Axiom**:
     $$\phi_0 + \sum_{j=1}^{13} \phi_j(x) = f(x)$$
2. **Interactive Interpretability Visualizations**:
   - **Interactive SHAP Waterfall Plot**: Traces cumulative risk step-by-step from base rate $\mathbb{E}[f(X)]$ to final patient risk probability, color-coded by directional impact (crimson red for risk-elevating contributions, emerald green for protective contributions).
   - **Diverging Attribution Force Bar Chart**: Ranks all 13 attributes by absolute magnitude $|\phi_j|$, presenting clear relative weights.
   - **Interactive "What-If" Counterfactual Intervention Explorer**: Clinicians can perturb individual high-impact features (e.g., lowering systolic BP from 170 to 120 mmHg) and observe how attribution bars shrink and overall predicted probability drops in real time.
3. **Clinical Decision Transparency & Plain-Language Explanations**:
   - Generates automated natural language audit summaries detailing the top risk-increasing and protective factors.
   - Compares local patient feature attributions against global population feature importance (from Task 56).

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_73_SHAP_Interpretability_UI/app.py
```

Run all cells in `Task_73_SHAP_Interpretability_UI/Task_73_SHAP_Interpretability_UI.ipynb` for the standalone notebook.

