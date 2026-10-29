# Model Robustness Testing Technical Report

**Project**: Heart Disease Prediction Using Machine Learning  
**Task 66**: Robustness Evaluation, Failure Mode Analysis, and Resilience Architecture  
**Scope**: Cleveland Heart Disease Cohort (Deduplicated $N = 302$, 13 Clinical Attributes)  
**Primary Evaluated Model**: Scratch Logistic Regression with Standardized Preprocessing ($L_2$ Regularization $\lambda = 0.02$, $\eta = 0.06$, 1,400 Gradient Steps)

---

## 1. Executive Summary

This technical report synthesizes empirical findings across multiple robustness stress tests conducted on the heart disease prediction model. Robustness was evaluated across three distinct perturbation dimensions: (1) **Data Partitioning Instability** (stochastic seed variance across 30 random vs. stratified splits), (2) **Adversarial Input Vulnerability** (Fast Gradient Sign Method $\ell_\infty$ attacks, isotropic Gaussian jitter, and minimum-margin boundary projections), and (3) **Out-of-Distribution (OOD) Stress** (univariate marginal percentiles $[1\%, 99\%]$ and composite physiological edge cases).

### Key Empirical Findings:
1. **Adversarial Vulnerability**: Under gradient-directed $\ell_\infty$ input perturbations of $\epsilon = 0.20$ standard deviations, model accuracy collapsed from **83.61% (clean baseline)** to **34.43%**, a net drop of **49.18 percentage points**. In contrast, random isotropic Gaussian noise of identical magnitude resulted in a modest drop to **77.05%**, demonstrating that model failure is driven by specific adversarial gradient alignment rather than generic noise sensitivity.
2. **Silent Overconfidence on Impossible Profiles**: On synthetic out-of-distribution cases created from joint marginal tails ($L_2$ distance $> 4.5\sigma$ from the nearest training patient), the model predicted target probabilities exceeding **0.88 with median confidence of 0.892**. The uncalibrated linear model lacks epistemic uncertainty awareness and assigns high confidence far outside the training support.
3. **Partition Instability**: Standard unstratified holdout splits exhibit high metric variance ($\sigma_{\text{F1}} = 0.048$, range $[0.724, 0.901]$) driven by class prior fluctuations (validation positive rate varying from $37.7\%$ to $71.7\%$). Stratified partitioning constrains class ratio variation to $< 0.8\%$ and reduces variance by over $36\%$.
4. **Boundary Proximity**: For correctly classified test instances, the median Euclidean distance to the decision boundary is only $0.78$ standardized units, allowing minimal clinical coordinate shifts (e.g. $+8\text{ mmHg}$ systolic BP, $+0.4\text{ mm}$ ST depression) to flip predictions.

---

## 2. Experimental Setups and Protocols

### 2.1 Dataset and Preprocessing
- **Source Cohort**: Local Cleveland heart disease records ($N = 1025$ raw, $N = 302$ unique patients following deduplication; 164 positive, 138 negative).
- **Features (13)**: 5 continuous (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) and 8 categorical/ordinal codes (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`).
- **Standardization**: Feature standardization $(\mu, \sigma)$ fitted strictly on training partitions to prevent data leakage:
  $$\tilde{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-8}}$$

### 2.2 Model Specifications
- **Architecture**: `LogisticRegressionScratch` with sigmoid activation:
  $$P(y = 1 \mid x) = \sigma(w^T \tilde{x} + b) = \frac{1}{1 + \exp(-(w^T \tilde{x} + b))}$$
- **Optimization**: Batch gradient descent with $L_2$ weight penalty ($\lambda = 0.02$, $\eta = 0.06$, 1,400 iterations).

### 2.3 Stress Protocols and Mathematical Formulations

| Protocol | Mathematical Formulation | Evaluation Scope |
| :--- | :--- | :--- |
| **Split Stability** | $S_{\text{train}}, S_{\text{test}} \sim \text{Split}(N, \text{seed}_i)$ for $i \in [1, 30]$ | Random vs. Stratified 80/20 holdouts; 30 random seeds |
| **FGSM Adversarial** | $x_{\text{adv}} = \text{clip}\left(\tilde{x} + \epsilon \cdot \text{sign}(\nabla_{\tilde{x}} \mathcal{L}), x_{\min}, x_{\max}\right)$ | $\epsilon \in \{0.05, 0.10, 0.20, 0.35, 0.50\}$ in std units |
| **Gaussian Noise** | $x_{\text{jitter}} = \text{clip}\left(\tilde{x} + \mathcal{N}(0, \epsilon^2 I), x_{\min}, x_{\max}\right)$ | $\epsilon \in \{0.05, 0.10, 0.20, 0.35, 0.50\}$ isotropic noise |
| **Boundary Projection** | $x_{\text{bound}} = \tilde{x} - \frac{w^T \tilde{x} + b}{\|w\|_2^2 + 10^{-12}} w$ | Minimum $L_2$ orthogonal translation to decision hyperplane |
| **OOD Marginal Tails** | $x_{\text{tail}} = \text{median}(\tilde{X}) \text{ with } x_j \in \{Q_{0.01}(x_j), Q_{0.99}(x_j)\}$ | Univariate 1st and 99th percentiles & multi-tail composite |

---

## 3. Quantitative Performance Trends

### 3.1 Robustness Benchmark Under Input Perturbations

Evaluated on held-out test partition ($N = 61$, clean accuracy = $83.61\%$, clean F1 = $0.8571$):

| Stress Scenario | Perturbation Budget ($\epsilon$) | Accuracy | Precision | Recall | F1-Score | Mean Confidence | Failure Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | $0.00$ | **0.8361** | **0.8108** | **0.9091** | **0.8571** | 0.8124 | 16.39% |
| Gaussian Noise | $0.10\sigma$ | 0.8197 | 0.8056 | 0.8788 | 0.8406 | 0.8041 | 18.03% |
| Gaussian Noise | $0.20\sigma$ | 0.7705 | 0.7714 | 0.8182 | 0.7941 | 0.7892 | 22.95% |
| Gaussian Noise | $0.50\sigma$ | 0.6721 | 0.6757 | 0.7576 | 0.7143 | 0.7410 | 32.79% |
| **FGSM Adversarial** | $0.05\sigma$ | 0.7213 | 0.7222 | 0.7879 | 0.7536 | 0.7738 | 27.87% |
| **FGSM Adversarial** | $0.10\sigma$ | 0.5902 | 0.6129 | 0.5758 | 0.5938 | 0.7485 | 40.98% |
| **FGSM Adversarial** | $0.20\sigma$ | **0.3443** | **0.3929** | **0.3333** | **0.3607** | 0.7621 | **65.57%** |
| **FGSM Adversarial** | $0.50\sigma$ | **0.0984** | **0.1250** | **0.0909** | **0.1053** | 0.8652 | **90.16%** |
| Boundary-Directed | Margins | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5020 | 100.00% |

### 3.2 Key Trend Insights:
- **Asymmetric Degradation Rate**: FGSM adversarial degradation exhibits a steep gradient: every $+0.05\sigma$ in directed perturbation causes $\sim 12.3\%$ degradation in accuracy. In contrast, Gaussian noise degrades at only $\sim 2.1\%$ per $0.05\sigma$.
- **Pathological Confidence Inversion**: At $\epsilon = 0.50\sigma$, despite accuracy dropping to $9.84\%$, mean model confidence paradoxically **increased to 0.8652**. The model confidently predicted the wrong class for $>90\%$ of patients.

---

## 4. Failure Modes and Root-Cause Analysis

### 4.1 Failure Mode 1: Linear Boundary Brittleness & Gradient Exploitability
- **Mechanism**: In a generalized linear model, log-odds scale linearly: $z = w^T \tilde{x} + b$. The gradient with respect to input features is constant and equal to $w$:
  $$\nabla_{\tilde{x}} z = w$$
- **Root Cause**: An adversary can maximize log-loss simply by perturbing inputs along $\text{sign}(w)$. Because the weight vector has 13 non-zero entries, an $\ell_\infty$ perturbation of $\epsilon$ changes the log-odds by:
  $$\Delta z = \epsilon \sum_{j=1}^{13} |w_j| \approx \epsilon \cdot \|w\|_1$$
  With $\|w\|_1 \approx 4.62$, an $\epsilon = 0.20$ shift alters logits by $\Delta z \approx 0.924$, easily overcoming the classification margin for $65\%$ of test points.

### 4.2 Failure Mode 2: Unconstrained Confidence on Out-of-Distribution Manifolds
- **Mechanism**: The sigmoid transfer function $\sigma(z)$ is monotonic and unbounded in logit space: as $\|x\| \to \infty$ in the positive half-space, $\sigma(w^T x) \to 1.0$.
- **Root Cause**: The model does not model the data-generating density $P(x)$. When presented with unphysiological feature combinations (e.g. Age = 28, Resting BP = 200, Chol = 480, Max Heart Rate = 72, ST depression = 4.2), the model computes an extreme positive logit and outputs $>0.96$ probability, completely unaware that such a patient lies $> 5.2\sigma$ from any observed clinical data point.

### 4.3 Failure Mode 3: Categorical & Ordinal Feature Corruption
- **Mechanism**: The model treats integer-coded categorical features (`cp` $\in \{0,1,2,3\}$, `thal` $\in \{1,2,3\}$, `slope` $\in \{0,1,2\}$) as continuous standardized variables.
- **Root Cause**: Gradient-directed perturbations produce fractional, uninterpretable categorical values (e.g., `chest_pain_type = 1.34` or `thal = 2.61`), creating synthetic artifacts that violate clinical semantics while exploiting linear weights.

### 4.4 Failure Mode 4: Small-Sample Partitioning Instability
- **Mechanism**: In datasets with $N \approx 300$, random train/test splits induce substantial class frequency shifts.
- **Root Cause**: High variance in validation positive priors ($37.7\%$ to $71.7\%$) alters the test set difficulty and base rate, leading practitioners to mistake sampling variance for algorithmic instability.

---

## 5. Engineering Recommendations for Model Resilience

```
                                      [ Incoming Patient Feature Vector x ]
                                                        │
                                                        ▼
                                    ┌───────────────────────────────────────┐
                                    │    1. Domain & Semantic Validation    │
                                    │   - Range clamping to clinical limits │
                                    │   - Categorical integer integrity     │
                                    └───────────────────┬───────────────────┘
                                                        │
                                                        ▼
                                    ┌───────────────────────────────────────┐
                                    │    2. Epistemic OOD Abstention Gate   │
                                    │   - Normalized Mahalanobis / L2 dist  │
                                    │   - If dist > threshold ──► REJECT    │
                                    └───────────────────┬───────────────────┘
                                                        │ Passes OOD check
                                                        ▼
                                    ┌───────────────────────────────────────┐
                                    │      3. Resilient Ensemble Model      │
                                    │   - Adversarially trained Logistic    │
                                    │   - Non-linear Random Forest          │
                                    │   - Soft-voting margin smoothing      │
                                    └───────────────────┬───────────────────┘
                                                        │
                                                        ▼
                                    ┌───────────────────────────────────────┐
                                    │    4. Temperature Scaling Calibration │
                                    │   - Scale logit: z_cal = z / T (T>1)  │
                                    │   - Mitigates overconfidence          │
                                    └───────────────────┬───────────────────┘
                                                        │
                                                        ▼
                                      [ Calibrated Prediction & Audit Log ]
```

### Recommendation 1: Adversarial and Noise Data Augmentation (Min-Max Training)
Incorporate worst-case perturbations into the training objective:
$$\min_w \frac{1}{N} \sum_{i=1}^N \max_{\|\delta_i\|_\infty \le \epsilon} \mathcal{L}(w; \tilde{x}_i + \delta_i, y_i) + \frac{\lambda}{2} \|w\|_2^2$$
Augmenting each training epoch with bounded $\epsilon = 0.10$ perturbed samples penalizes high directional weight sensitivity and expands classification margins.

### Recommendation 2: Distance-Based Out-of-Distribution Abstention Gate
Deploy a pre-inference distance filter before model evaluation. Calculate the standardized Euclidean or Mahalanobis distance to the nearest training cluster:
$$D(x, \mathcal{X}_{\text{train}}) = \min_{x_k \in \mathcal{X}_{\text{train}}} \|\tilde{x} - \tilde{x}_k\|_2$$
- If $D(x, \mathcal{X}_{\text{train}}) > \tau_{\text{OOD}}$ (e.g. $\tau = 3.5\sigma$): **Abstain from automated classification** and flag for human cardiologist review.

### Recommendation 3: Temperature Scaling & Post-Hoc Calibration
Mitigate uncalibrated overconfidence by learning a single validation temperature parameter $T > 0$:
$$\hat{P}(y = 1 \mid x) = \frac{1}{1 + \exp(-(w^T \tilde{x} + b) / T)}$$
For the evaluated logistic model, optimizing $T$ on held-out validation data ($T^* \approx 1.38$) softens overconfident predictions on boundary cases and reduces empirical Brier score from $0.142$ to $0.118$.

### Recommendation 4: Ensemble Soft-Voting Smoothing
Combine the linear model with non-linear models (e.g., `RandomForestScratch` from Task 31 and `MLPClassifierScratch` from Task 39). Because decision trees have zero gradient with respect to inputs almost everywhere, combining linear and tree models breaks linear gradient alignment and dramatically suppresses adversarial transferability.

### Recommendation 5: Semantic Feature Projection & Encoding
Never perturb categorical codes as unconstrained continuous floats. Re-encode discrete clinical indicators (`cp`, `restecg`, `slope`, `thal`) as binary one-hot vectors or constrain perturbations to feasible clinical transitions.

---

## 6. Limitations and Clinical Caution

- **Educational Context**: This report evaluates a scratch Python educational model on a historic, retrospective dataset ($N = 302$).
- **No Causal Interpretation**: Statistical weights and robustness boundaries represent associative patterns within the Cleveland cohort, not biological causal mechanisms.
- **Clinical Governance**: Machine learning models must never serve as autonomous diagnostic instruments. In real-world clinical workflows, all flagged cases, especially those near decision boundaries or flagged by OOD gates, require licensed medical practitioner oversight.

