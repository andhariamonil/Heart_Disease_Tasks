# Heart Disease Prediction Using Machine Learning

## Task 70: Interactive Prediction Confidence Distribution & Visualization

### Objective

Visualize and analyze machine learning model prediction confidence using manually computed probability distributions, provide real-time updates with animated visual components, and enable interactive exploration of decision thresholds, epistemic uncertainty, and clinical confidence zones.

Key Capabilities:
1. **Manual Probability Distribution & Uncertainty Calculations**:
   - Computes exact class probabilities $P(Y=0 \mid x)$ and $P(Y=1 \mid x)$ via logistic sigmoid activations without external wrapper libraries.
   - Computes normalized classification margins $\Delta = 2 \cdot |P(Y=1) - 0.5| \in [0, 1]$.
   - Measures epistemic uncertainty via Shannon entropy:
     $$H(P) = - \sum_{c \in \{0, 1\}} P(Y=c) \log_2 P(Y=c) \quad (\text{bits})$$
2. **Real-Time Updates with Dynamic CSS & SVG Animations**:
   - Renders animated distribution bars with pulse transitions, glowing confidence badges, and live progress gauges.
   - Includes an automated "Clinical Stress Trajectory Animation" stepping through evolving patient vital signs (e.g., escalating ischemia) with live animated metric updates.
3. **Interactive Confidence & Decision Threshold Exploration**:
   - **Decision Cutoff Slider ($\tau \in [0.10, 0.90]$)**: Real-time exploration of sensitivity vs. specificity clinical tradeoffs.
   - **Clinical Abstention Zone ($\pm \delta$)**: User-configurable uncertainty margin around $0.5$ where the model abstains and mandates human physician review.
   - **Attribute Sensitivity Exploration**: Live manipulation of continuous clinical vitals (systolic blood pressure, cholesterol, max heart rate, and ST depression) to observe immediate probability shifts.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_70_Confidence_Distribution_Visualization/app.py
```

Run all cells in `Task_70_Confidence_Distribution_Visualization/Task_70_Confidence_Distribution_Visualization.ipynb` for the standalone notebook.

