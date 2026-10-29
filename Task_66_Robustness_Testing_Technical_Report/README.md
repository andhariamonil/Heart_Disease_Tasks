# Heart Disease Prediction Using Machine Learning

## Task 66: Robustness Testing Technical Report

### Objective

Consolidate, benchmark, and document findings on model robustness testing through a structured technical report. The analysis evaluates experimental setups, empirical performance degradation trends under systematic input stress (adversarial FGSM, isotropic Gaussian jitter, decision boundary projection, and out-of-distribution edge cases), diagnoses four core architectural failure modes, and establishes actionable engineering recommendations for model resilience.

Detailed contents are documented in [ROBUSTNESS_TESTING_REPORT.md](ROBUSTNESS_TESTING_REPORT.md):
- **Executive Summary**: Core resilience findings and performance impacts.
- **Experimental Setups**: Mathematical formulations for adversarial $\ell_\infty$ attacks, boundary projection, and OOD metrics on deduplicated Cleveland data ($N = 302$).
- **Quantitative Trends**: Benchmark performance table across perturbation levels ($\epsilon \in [0.00, 0.50]$) comparing directed attacks vs. random noise.
- **Failure Modes**: Root-cause analysis of linear boundary brittleness, silent overconfidence, categorical distortion, and finite-sample partition variance.
- **Resilience Architecture**: 5 concrete engineering strategies (adversarial data augmentation, distance-based abstention gating, temperature scaling, ensemble soft voting, and semantic integer projection).

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_66_Robustness_Testing_Technical_Report/app.py
```

Run all cells in `Task_66_Robustness_Testing_Technical_Report/Task_66_Robustness_Testing_Technical_Report.ipynb` for the standalone notebook.

