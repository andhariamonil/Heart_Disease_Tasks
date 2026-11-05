import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import load_data


CATEGORICAL = {'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'}


def initial_fill(values, observed, features):
    filled = values.copy()
    category_values = {}
    for column, feature in enumerate(features):
        present = values[observed[:, column], column]
        if feature in CATEGORICAL:
            allowed, counts = np.unique(present, return_counts=True)
            mode = allowed[np.argmax(counts)]
            category_values[column] = allowed
            filled[~observed[:, column], column] = mode
        else:
            filled[~observed[:, column], column] = np.median(present)
    return filled, category_values


def predict_conditional_column(filled, observed, target_column, feature, category_values, ridge=1.0):
    target_rows = observed[:, target_column]
    missing_rows = ~target_rows
    predictor_columns = [column for column in range(filled.shape[1]) if column != target_column]
    if not missing_rows.any() or not target_rows.any():
        return filled[:, target_column]
    train_x = filled[target_rows][:, predictor_columns]
    predict_x = filled[missing_rows][:, predictor_columns]
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales[scales == 0] = 1.0
    train_x = (train_x - means) / scales
    predict_x = (predict_x - means) / scales
    train_y = filled[target_rows, target_column]

    if feature in CATEGORICAL:
        classes = category_values[target_column]
        class_centers = np.array([train_x[train_y == category].mean(axis=0) if np.any(train_y == category) else train_x.mean(axis=0) for category in classes])
        distances = np.sum((predict_x[:, None, :] - class_centers[None, :, :]) ** 2, axis=2)
        predictions = classes[np.argmin(distances, axis=1)]
    else:
        design = np.column_stack([np.ones(len(train_x)), train_x])
        query = np.column_stack([np.ones(len(predict_x)), predict_x])
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
        predictions = query @ coefficients
    result = filled[:, target_column].copy()
    result[missing_rows] = predictions
    return result


def impute_predictive(values, observed, features, ridge=1.0):
    filled, categories = initial_fill(values, observed, features)
    for column, feature in enumerate(features):
        filled[:, column] = predict_conditional_column(filled, observed, column, feature, categories, ridge)
    return filled


def impute_mice(values, observed, features, iterations=4, ridge=1.0):
    filled, categories = initial_fill(values, observed, features)
    for _ in range(iterations):
        for column, feature in enumerate(features):
            filled[:, column] = predict_conditional_column(filled, observed, column, feature, categories, ridge)
    return filled


def impute_knn(values, observed, features, neighbors=7):
    filled, _ = initial_fill(values, observed, features)
    scales = np.nanstd(np.where(observed, values, np.nan), axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    rows, columns = np.where(~observed)
    for row, column in zip(rows, columns):
        candidates = np.flatnonzero(observed[:, column])
        candidates = candidates[candidates != row]
        common = observed[candidates] & observed[row][None, :]
        common[:, column] = False
        differences = (filled[candidates] - filled[row]) / scales
        counts = common.sum(axis=1)
        distance = np.sqrt((np.sum(np.where(common, differences ** 2, 0.0), axis=1) / np.maximum(counts, 1)))
        nearest = candidates[np.argsort(distance, kind='stable')[:min(neighbors, len(candidates))]]
        if not len(nearest):
            continue
        neighbor_values = values[nearest, column]
        if features[column] in CATEGORICAL:
            classes, class_counts = np.unique(neighbor_values, return_counts=True)
            filled[row, column] = classes[np.argmax(class_counts)]
        else:
            filled[row, column] = float(np.mean(neighbor_values))
    return filled


st.title('Task 82: Manual Multiple Imputation Comparison')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
values = frame[features].to_numpy(dtype=float)
missing_count = int(np.isnan(values).sum())
st.write(f'Actual missing feature values after duplicate removal: {missing_count}')
if missing_count:
    observed_values = values.copy()
    observed_mask = np.isfinite(values)
else:
    rng = np.random.default_rng(82)
    observed_values = values.copy()
    observed_mask = rng.random(values.shape) >= 0.12
    st.info('The supplied CSV is complete, so methods are compared on a reproducible 12% artificial holdout. The source data is not modified.')

masked = values.copy()
masked[~observed_mask] = np.nan
methods = {
    'KNN (k=7)': impute_knn(masked, observed_mask, features, neighbors=7),
    'MICE-like chained equations': impute_mice(masked, observed_mask, features, iterations=4),
    'One-pass predictive': impute_predictive(masked, observed_mask, features),
}
records = []
for column, feature in enumerate(features):
    held = ~observed_mask[:, column]
    if not held.any():
        continue
    for method, imputed in methods.items():
        if feature in CATEGORICAL:
            error = float(np.mean(imputed[held, column] != values[held, column]))
            score_name = 'Validation error rate'
        else:
            scale = float(np.quantile(values[:, column], 0.75) - np.quantile(values[:, column], 0.25)) or 1.0
            error = float(np.mean(np.abs(imputed[held, column] - values[held, column])) / scale)
            score_name = 'IQR-normalized MAE'
        records.append({'Feature': feature, 'Method': method, 'Metric': score_name, 'Error': error})

results = pd.DataFrame(records)
selected = results.loc[results.groupby('Feature')['Error'].idxmin()].sort_values('Feature')
st.subheader('Method comparison on masked cells')
st.dataframe(results.pivot(index='Feature', columns='Method', values='Error').style.format('{:.3f}'))
st.subheader('Per-column selected method')
st.dataframe(selected.set_index('Feature').style.format({'Error': '{:.3f}'}))
st.caption('Continuous columns use IQR-normalized MAE; integer-coded categorical columns use error rate. Selection is based on one artificial masking seed and should be repeated across masks before real use. MICE-like updating here uses ridge regression and nearest class-centroid categorical predictions, not a clinical imputation standard.')