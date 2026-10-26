# Feature Importance Technical Report

## Executive summary

This report compares three global feature-importance approaches for a manually trained logistic model and adds exact Shapley values for individual predictions. On the recorded deduplicated dataset split, the consensus ranking places chest-pain code (`cp`), vessel count (`ca`), exercise-induced angina (`exang`), and sex (`sex`) at the top. These are model-specific associations, not causal or clinical recommendations.

## Experimental setup

- Source dataset: local heart CSV, exact duplicate rows removed; 302 unique records in the Task 53–55 workflow.
- Model: scratch logistic regression with standardized inputs, learning rate 0.06, 1,400 iterations, and L2 strength 0.02.
- Split for this report: seeded development, validation, and held-out test subsets. Model fit and scaling use training rows only.
- Validation permutation importance: five independently shuffled runs per feature; score is baseline validation F1 minus permuted validation F1.
- Shapley: exact interventional coalition enumeration for six held-out cases, with an eight-row empirical training background. Additivity was verified for local predictions.

Validation F1 was 0.881. On the held-out test set, the classifier had accuracy 0.803, precision 0.769, recall 0.909, and F1 0.833.

## Global rankings

| Consensus rank | Feature | Absolute standardized coefficient | Mean validation F1 drop (SD) | Mean absolute SHAP | SHAP SD across cases |
|---:|---|---:|---:|---:|---:|
| 1.67 | cp | 0.6448 | 0.0714 (0.0164) | 0.1111 | 0.0770 |
| 2.00 | ca | 0.6308 | 0.0527 (0.0478) | 0.1163 | 0.0637 |
| 2.67 | exang | 0.5400 | 0.0114 (0.0137) | 0.1263 | 0.0650 |
| 3.67 | sex | 0.5145 | 0.0192 (0.0165) | 0.0959 | 0.0852 |
| 5.33 | thal | 0.4955 | 0.0057 (0.0158) | 0.0775 | 0.0891 |
| 6.67 | oldpeak | 0.4096 | 0.0072 (0.0145) | 0.0239 | 0.0284 |
| 8.67 | trestbps | 0.3643 | -0.0176 (0.0145) | 0.0450 | 0.0486 |
| 8.67 | slope | 0.3362 | -0.0130 (0.0125) | 0.0447 | 0.0378 |
| 9.00 | thalach | 0.2508 | -0.0052 (0.0092) | 0.0221 | 0.0253 |
| 9.33 | chol | 0.1595 | -0.0061 (0.0083) | 0.0240 | 0.0268 |
| 10.00 | restecg | 0.1478 | 0.0000 (0.0000) | 0.0199 | 0.0179 |
| 11.33 | age | 0.1962 | -0.0152 (0.0000) | 0.0090 | 0.0079 |
| 12.00 | fbs | 0.0278 | -0.0061 (0.0083) | 0.0029 | 0.0004 |

Consensus rank averages each method's ranking after normalizing the magnitude scale. Negative permutation drops for some low-ranked features mean that shuffling slightly improved validation F1 in this finite sample; this is sampling variability, not evidence that a feature is beneficial to remove. Several permutation standard deviations are large relative to their means, so the ordering below the top features is uncertain.

## Local interpretation

The notebook visualizes one held-out prediction and verifies that the baseline plus all feature attributions reconstructs the model probability to numerical precision. Its example case had a model probability of 0.730 against a background expectation of 0.683. In the six-case sample, attribution variability was notable: `cp`, `exang`, and `sex` had larger case-to-case standard deviations, showing why a global ranking cannot replace individual review.

## Technical selection rationale

Use `cp`, `ca`, `exang`, and `sex` as the first candidates for follow-up investigation because they rank highly across coefficient, permutation, and SHAP summaries. Do not automatically discard features with small or negative measured permutation impact: correlated fields may substitute for one another, and the permutation sample is limited. Before feature selection, repeat permutation runs with more repeats, bootstrap the rank stability, inspect subgroup and error analyses, and validate the selected model on an external cohort.

## Method-specific limitations

- **Coefficient magnitude:** depends on scaling, regularization, model form, and correlated inputs; it represents a conditional linear association.
- **Permutation importance:** measures score dependence on the chosen validation set and metric. Correlated features can mask each other; finite shuffles create noisy or negative estimates.
- **Exact Shapley:** depends on the chosen background and interventional replacement rule. At 13 features it evaluates 8,192 coalitions per case; it does not estimate causality.
- **Categorical encoding:** `cp`, `slope`, `ca`, and `thal` are numeric codes. Treating them as continuous can imply artificial distances or fractional categories.

## Use limitation

This is an educational report on one small dataset and a scratch logistic model. It is not clinical evidence, diagnosis guidance, or a deployment feature-selection decision. Domain experts and data stewards should review codebooks, cohort representativeness, target definition, and consequences of errors before any real-world use.
