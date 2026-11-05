# Heart Disease ML Project: Implementation and Findings

## 1. Scope

This repository is a staged educational exploration of heart-disease data processing and model development. Tasks 1–21 cover loading, cleaning, summaries, visualization, encoding, transformations, and introductory statistics. Tasks 22–49 continue through outlier handling and manually implemented classifiers, validation, feature selection, calibration, voting, stacking, and hyperparameter search. Tasks 50–82 add generalization analysis, a manual learning-rate scheduler, interpretability, fairness, robustness, input validation, user explanations, custom UI, CI, API deployment, latency profiling, and multiple imputation.

Most learning algorithms are implemented in `manual_ml.py` and demonstrated in task-specific folders. Later comparisons deduplicate exact repeated source rows and fit scaling on training partitions. Historical tasks do not all use one shared evaluation split, so results below must not be read as a single leaderboard.

## 2. Methodologies

- **Data preparation:** loading, duplicate/missing-value checks, summaries, plots, categorical encodings, normalization, outlier analysis, and feature engineering.
- **Classical ML:** scratch logistic regression, linear/kernel SVM, decision trees, random forest, boosting, and KD-tree k-NN.
- **Neural networks:** manual forward/backpropagation with dropout, batch normalization, validation loss, plateau-based learning-rate reduction, and update-norm tracking.
- **Model selection:** manual exhaustive search and Gaussian-process expected-improvement search; the later tasks preserve a held-out test split.
- **Interpretability:** coefficient and permutation rankings, exact interventional Shapley values, local LIME-style surrogates, and cross-method comparisons.
- **Fairness and robustness:** confusion/error analysis, disparate-impact and equalized-odds metrics, reweighting, sensitive-feature removal, split stability, adversarial stress tests, and synthetic OOD cases.
- **Missing data:** KNN, iterative chained conditional prediction, and one-pass predictive imputation are compared on an artificial mask because the source CSV has no missing feature cells.
- **Deployment/performance:** explicit versioned joblib state, a bearer-protected FastAPI API, AWS ECS/Fargate Terraform, CI validation, profiling, and bounded localhost load tests.

## 3. Selected experimental results

These are illustrative results from the local dataset and stated task seeds, not population-level evidence.

| Experiment | Recorded result | Interpretation |
|---|---|---|
| Task 50 tuning/generalization | Logistic CV F1 changed from about 0.844 to 0.846; held-out F1 remained 0.831. | The small CV gain did not improve this test split. |
| Task 51 scheduler | Constant and scheduled runs both had held-out F1 about 0.714 on the recorded seed. | Learning-rate reductions changed training dynamics but did not improve test F1 in that run. |
| Task 52 benchmark | In the local run, k-NN F1 was about 0.853 with near-zero fit time and about 6 ms inference; logistic F1 was about 0.817 with roughly 0.34 s fit. | k-NN fitting is cheap but inference grows with the reference set; timings are hardware-specific. |
| Task 56 importance | Consensus ranking emphasized `cp`, `ca`, `exang`, and `sex`; validation F1 was about 0.881. | Rankings varied by method, and lower-ranked permutation estimates were noisy or sometimes negative. |
| Task 58 calibration | Test FNR decreased from 3/33 to 2/33 while FPR increased from 8/28 to 10/28; test F1 changed about 0.845 to 0.838. | Thresholding traded missed positives for more false alarms. |
| Task 61 mitigation | Reweighting improved disparate impact from about 0.711 to 0.835 but lowered F1 from about 0.870 to 0.829. | Improving one fairness metric can worsen utility or another fairness metric. |
| Task 63 adversarial stress | Accuracy changed from about 0.918 clean to 0.754 under a bounded gradient-sign perturbation in one split. | The model is sensitive to coordinated input changes; those changes may not be realistic patient edits. |
| Task 64 OOD stress | A combined-tail synthetic point was about 4.49 standardized L2 from training data yet received about 0.98 confidence. | Confidence alone did not reveal distance from training support. |
| Task 79 local API benchmark | 100/100 localhost requests succeeded at 16 workers; about 278 requests/s, median 45.8 ms, p95 74.4 ms. | This is a bounded local smoke benchmark, not a cloud capacity test. |
| Task 80 API profiler | Direct request-stage medians were about 0.058 ms validation, 0.015 ms scaling, 0.047 ms inference, 0.034 ms explanation, and 0.027 ms serialization. On this run FP16 max probability error was 0.000315; INT8 max error was 0.008781. | In NumPy on this CPU, quantized batch inference did not outperform FP32 consistently despite smaller weight storage. Validate on target hardware. |
| Task 82 imputation | 479 of 3,926 cells were artificially masked; selected methods varied by feature (KNN for several categorical columns, MICE-like for cholesterol, predictive regression for several continuous columns). | The source data was complete; this synthetic-mask comparison does not establish a preferred method for naturally missing clinical data. |

## 4. Validation and reproducibility

Later task notebooks and apps were executed in the project virtual environment. Split seeds and deduplication are specified by task. Repeated holdouts reuse examples, so their spread is descriptive rather than an independent confidence interval. The source CSV had no missing values in recorded checks; imputation comparisons use artificial masking and do not establish the best method for naturally missing clinical data.

## 5. Interpretability and fairness caveats

Shapley results depend on the background and feature-replacement convention; exact coalition computation scales exponentially. LIME depends on perturbation sampling, kernel width, and local surrogate fit. Both can rely on implausible combinations when features are correlated. Integer-coded categories can be mistakenly treated as continuous. These methods explain model behavior, not causes.

Fairness demonstrations use only the CSV’s binary `sex` field. Group samples are small, intersectional attributes are absent, and the reported metrics are split-dependent. The analysis is not legal certification or clinical approval. No domain-expert interviews were conducted; Task 57 labels its review prompts as domain-informed considerations, not expert testimony.

## 6. Deployment status and challenges

- **Streamlit Cloud:** a GitHub Actions CI workflow and setup instructions are present. The repository has not been connected to a Streamlit Cloud account from this environment.
- **Model artifact:** Task 47 exports explicit versioned logistic-regression state and preprocessing metadata to a compressed joblib file. Only trusted joblib files should be loaded.
- **AWS API:** Task 78 contains FastAPI, tests, Dockerfile, and Terraform for HTTPS ALB ingress, private ECS/Fargate tasks, Secrets Manager token injection, logging, rollback, and CPU/memory autoscaling. Terraform was syntax-parsed, but AWS resources were not provisioned.
- **Required account resources:** AWS credentials, VPC/subnets, outbound access for private tasks, ACM certificate, DNS name, Secrets Manager token, and ECR image push.
- **Load testing:** Task 79 is fixed to localhost and capped. Larger tests should use an authorized staging environment with quotas, monitoring, and an agreed stop condition.

## 7. Recommendations

1. Verify dataset provenance, target semantics, feature units, and category codebooks; version these with each model artifact.
2. Use nested cross-validation and an external time- or site-separated validation cohort before declaring a best model.
3. Measure calibration, uncertainty, subgroup errors, and intervals with adequate subgroup samples.
4. Review false positives, false negatives, explanations, and OOD examples with qualified domain experts and affected stakeholders.
5. Treat feature attribution as diagnostic, not causal; use conditional or grouped perturbations for correlated and categorical features.
6. For deployment, require HTTPS and managed authentication, rotate credentials, add request limits, and avoid logging sensitive feature payloads.
7. Benchmark FP16/INT8, caching, and autoscaling on target hardware before enabling them in production.
8. Add automated checks for schema compatibility, artifact integrity, API tests, dependency/security updates, and rollback readiness.

## Conclusion

The repository demonstrates a broad manual ML and deployment workflow. Results emphasize that evaluation design, calibration, subgroup analysis, input constraints, and realistic operational testing matter as much as model scores. This project is educational and is not a clinical decision system; any real-world use would need independent validation, governance, and applicable regulatory review.