# Case Study: Interpreting Heart-Disease Classifiers

## Scope and evidence

This case study reviews the educational scratch models and explanation methods built in Tasks 53–56. It is a technical interpretability exercise, not a clinical validation study. The repository contains no clinician interviews, patient review, or institutional approval. Therefore, the qualitative perspectives below are domain-informed considerations and proposed expert-review prompts, not quotations or claims attributed to actual experts.

## Model and explanation context

The model is a manually trained logistic classifier over tabular fields such as age, sex, chest-pain code, resting blood pressure, cholesterol, maximum heart rate, exercise-induced angina, ST depression, slope, vessel count, and thalassemia code. Inputs are standardized. The label is the dataset’s binary target; the feature encodings and target definition are inherited from the supplied dataset and may not match a current care pathway.

Task 56 compares coefficient magnitude, validation permutation impact, and exact Shapley values. Tasks 53–55 show local Shapley and LIME-style explanations and measure their agreement. These answer different questions: coefficients describe a fitted linear relationship, permutation measures validation-score dependence, Shapley allocates a prediction difference relative to a selected background, and LIME approximates behavior near one case.

## Findings and interpretation

1. **Global importance is method-dependent.** A large standardized coefficient can have small permutation impact when correlated fields substitute for it. Shapley values distribute prediction differences according to the chosen background and missing-feature convention. A consensus rank is a prioritization aid, not a definitive causal ordering.
2. **Local reasons vary by case.** The same feature can raise one case’s score and lower another’s. A global mean can hide subgroups, atypical combinations, and model instability; local explanations should be checked against the individual’s observed values and data quality.
3. **LIME fidelity does not validate the classifier.** An independent perturbation R-squared evaluates how well a local surrogate mimics the model in its generated neighborhood. It does not establish that either the model or the neighborhood is clinically valid.
4. **Coded categories need special treatment.** Integer encodings such as chest-pain type, slope, vessel count, and thal code may be nominal or ordinal by dataset convention. Treating a code as a continuous number invents distances and intermediate values. One-hot/grouped interventions or conditional sampling should be evaluated with domain reviewers.
5. **Correlated and impossible feature combinations matter.** Independent permutation and interventional masking can break relationships among exercise response, ST depression, heart rate, and other variables. Explanations may rely on combinations absent from real patients.

## Domain-informed review prompts

These are qualitative considerations to bring to a clinician and a data steward; they are not interview findings:

- A clinician should verify that every explanation uses the correct units, codebook, and target definition, then decide whether the displayed factors are clinically plausible for that specific case.
- A clinical safety reviewer should inspect false negatives separately from false positives, because a missed case and an unnecessary follow-up have different consequences. The explanation must not obscure uncertainty or substitute for clinical assessment.
- A data steward should identify cohort selection, missingness, measurement era, duplicated records, subgroup coverage, and coding changes that could alter both predictions and explanations.
- A human-factors reviewer should check whether users read contribution signs, baseline, probability calibration, and limitations correctly, and whether the interface encourages overreliance.

## Recommended expert review protocol

1. Recruit clinicians representing the intended workflow, a statistician/model validator, and a data steward; record roles, conflicts, and review date.
2. Select representative true positives, true negatives, false positives, false negatives, boundary cases, and subgroup cases before showing explanations.
3. Ask reviewers to judge clinical plausibility, missing context, actionability, and harmful failure modes, separately from whether they agree with the model.
4. Compare explanation stability across backgrounds, seeds, model versions, and correlated-feature groups; document disagreement rather than collapsing it into a single score.
5. Repeat after any data, model, threshold, or user-interface change and maintain an audit log.

## Improvements for real-world explainability

- Validate calibration and subgroup performance on an external, temporally appropriate cohort before presenting explanations operationally.
- Use clinically defensible feature grouping and conditional/background sampling; display unsupported or out-of-distribution cases as uncertain.
- Provide both local and global views, uncertainty/stability indicators, model version, baseline population, and concise caveats.
- Audit false positives and false negatives with domain reviewers and monitor drift after deployment.
- Preserve data provenance, codebook, explanation settings, and test evidence for each released model version.

## Limitations

The small, duplicated source dataset was deduplicated in the continuation tasks but may still have sampling bias and limited representativeness. The scratch logistic model is intentionally simple. Exact Shapley computation has exponential cost; LIME depends on its perturbation distribution and kernel width. Neither feature attribution method estimates causal effects, proves clinical validity, or establishes that a feature should be used in care.