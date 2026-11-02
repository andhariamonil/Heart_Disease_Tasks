# Heart Disease Prediction Using Machine Learning

## Task 72: Interactive Input Graphs, Dynamic Sliders, and Cross-Device Synchronization

### Objective

Implement custom interactive visualization components and dynamically linked input sliders that bridge patient vital controls with real-time machine learning predictions, ensuring fluid cross-device compatibility across desktop, tablet, and mobile displays.

Key Architectural Capabilities:
1. **Custom Interactive Input Visualizations**:
   - **Multi-Feature Physiological Radar / Spider Chart**: Maps 5 core continuous vitals (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) normalized to cohort percentile coordinates $[0, 100\%]$, overlaying the active patient polygon against the population reference baseline.
   - **Bivariate Clinical Decision Surface**: Renders a 2D risk contour plane (e.g. Max Heart Rate `thalach` vs. Age `age`), plotting patient location relative to the model's non-linear decision boundary in real time.
   - **Dynamic Risk Gauge Needle**: Displays class probability $P(Y=1 \mid x)$ on a multi-tier risk arc (Low Risk, Borderline, High CAD Risk).
2. **Real-Time Dynamic Two-Way Linking**:
   - Every slider adjustment instantly recalculates model logit $z = w^T \tilde{x} + b$, sigmoid probability $\sigma(z)$, radar polygon vertices, and bivariate coordinate markers with sub-millisecond execution (< 0.01 ms).
3. **Multi-Device Responsive Architecture**:
   - Touch-optimized slider controls with enlarged tap targets and CSS media queries supporting:
     - **Desktop (>1024px)**: Side-by-side dual-pane view (interactive sliders left, twin visualization charts right).
     - **Tablet (768px - 1024px)**: 2-column wrapping grid.
     - **Mobile (≤768px)**: Stacked single-column vertical layout with responsive SVG/matplotlib resizing.
   - In-app interactive **Viewport Device Simulator** (Desktop, Tablet, Mobile) to verify responsive behavior dynamically.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_72_Interactive_Input_Graphs_and_Sliders/app.py
```

Run all cells in `Task_72_Interactive_Input_Graphs_and_Sliders/Task_72_Interactive_Input_Graphs_and_Sliders.ipynb` for the standalone notebook.

