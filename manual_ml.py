"""Small educational ML implementations used by follow-on heart-disease tasks."""

from __future__ import annotations

import heapq
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"


def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    frame = pd.read_csv(path)
    features = [column for column in FEATURES if column in frame.columns]
    return frame[features].to_numpy(dtype=float), frame[TARGET].to_numpy(dtype=int), features, frame


def stratified_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_size)))
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])
    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }


def manual_voting(probabilities: np.ndarray, weights: np.ndarray | None = None, mode: str = "soft", threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Combine rows of model probabilities with manually weighted hard or soft votes."""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have shape (models, samples)")
    if mode not in {"hard", "soft"}:
        raise ValueError("mode must be 'hard' or 'soft'")
    model_weights = np.ones(probabilities.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if model_weights.shape != (probabilities.shape[0],) or np.any(model_weights < 0) or model_weights.sum() <= 0:
        raise ValueError("weights must be nonnegative, match the model count, and have a positive sum")
    normalized_weights = model_weights / model_weights.sum()
    if mode == "soft":
        combined_scores = normalized_weights @ probabilities
    else:
        hard_votes = (probabilities >= threshold).astype(float)
        combined_scores = normalized_weights @ hard_votes
        tied = np.isclose(combined_scores, 0.5)
        if tied.any():
            combined_scores[tied] = normalized_weights @ probabilities[:, tied]
    return (combined_scores >= 0.5).astype(int), combined_scores


def class_balanced_indices(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return oversampled indices with equal representation for each observed class."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("at least two classes are required")
    target_count = int(counts.max())
    sampled = [rng.choice(np.flatnonzero(labels == label), size=target_count, replace=count < target_count) for label, count in zip(classes, counts)]
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def exact_shapley_values(predict_function, instance: np.ndarray, background: np.ndarray, batch_size: int = 256) -> tuple[float, np.ndarray]:
    """Compute exact interventional Shapley values by enumerating feature coalitions."""
    instance = np.asarray(instance, dtype=float).reshape(-1)
    background = np.asarray(background, dtype=float)
    if background.ndim != 2 or background.shape[1] != len(instance) or len(background) == 0:
        raise ValueError("background must be a nonempty matrix matching the instance feature count")
    feature_count = len(instance)
    if feature_count > 16:
        raise ValueError("exact coalition enumeration is limited to 16 features")
    coalition_count = 1 << feature_count
    feature_bits = 1 << np.arange(feature_count)
    mask_ids = np.arange(coalition_count, dtype=np.int64)
    mask_matrix = (mask_ids[:, None] & feature_bits[None, :]) != 0
    coalition_values = np.empty(coalition_count, dtype=float)
    for start in range(0, coalition_count, batch_size):
        stop = min(start + batch_size, coalition_count)
        coalition_masks = mask_matrix[start:stop]
        hybrid = np.where(coalition_masks[:, None, :], instance[None, None, :], background[None, :, :])
        predictions = np.asarray(predict_function(hybrid.reshape(-1, feature_count)), dtype=float).reshape(stop - start, len(background))
        coalition_values[start:stop] = predictions.mean(axis=1)
    factorial = math.factorial
    denominator = factorial(feature_count)
    attributions = np.zeros(feature_count, dtype=float)
    for feature in range(feature_count):
        bit = 1 << feature
        eligible = mask_ids[(mask_ids & bit) == 0]
        sizes = np.array([int(mask).bit_count() for mask in eligible])
        weights = np.array([factorial(int(size)) * factorial(feature_count - int(size) - 1) / denominator for size in sizes])
        attributions[feature] = np.sum(weights * (coalition_values[eligible | bit] - coalition_values[eligible]))
    return float(coalition_values[0]), attributions


def lime_perturbations(instance: np.ndarray, background: np.ndarray, sample_count: int = 1000, seed: int = 42) -> np.ndarray:
    """Create tabular LIME samples by independently drawing observed feature values."""
    instance = np.asarray(instance, dtype=float).reshape(-1)
    background = np.asarray(background, dtype=float)
    if background.ndim != 2 or background.shape[1] != len(instance) or len(background) == 0:
        raise ValueError("background must be a nonempty matrix matching the instance feature count")
    rng = np.random.default_rng(seed)
    row_indices = rng.integers(0, len(background), size=(sample_count, len(instance)))
    columns = np.arange(len(instance))[None, :]
    perturbed = background[row_indices, columns]
    perturbed[0] = instance
    return perturbed


def fit_lime_surrogate(perturbed: np.ndarray, predictions: np.ndarray, weights: np.ndarray, ridge: float = 1e-3) -> tuple[float, np.ndarray, float]:
    """Fit a local weighted linear surrogate and return intercept, coefficients, and weighted R-squared."""
    perturbed = np.asarray(perturbed, dtype=float)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if perturbed.ndim != 2 or len(perturbed) != len(predictions) or len(weights) != len(predictions):
        raise ValueError("perturbed rows, predictions, and weights must have matching lengths")
    design = np.column_stack([np.ones(len(perturbed)), perturbed])
    root_weights = np.sqrt(np.maximum(weights, 0.0))
    weighted_design = design * root_weights[:, None]
    weighted_targets = predictions * root_weights
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(weighted_design.T @ weighted_design + penalty, weighted_design.T @ weighted_targets)
    fitted = design @ coefficients
    total_weight = weights.sum()
    mean = float(np.sum(weights * predictions) / total_weight) if total_weight else float(predictions.mean())
    residual_sum = float(np.sum(weights * (predictions - fitted) ** 2))
    total_sum = float(np.sum(weights * (predictions - mean) ** 2))
    weighted_r2 = 1.0 - residual_sum / total_sum if total_sum > 1e-15 else 1.0
    return float(coefficients[0]), coefficients[1:], weighted_r2


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    return means, scales


def standardize(x: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (x - means) / scales


class LogisticRegressionScratch:
    def __init__(self, learning_rate: float = 0.08, iterations: int = 1800, l2: float = 0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticRegressionScratch":
        self.weights = np.zeros(x.shape[1], dtype=float)
        self.bias = 0.0
        weights = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(y),) or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, match y, and have a positive sum")
        weights = weights / weights.sum()
        for _ in range(self.iterations):
            probabilities = self._sigmoid(x @ self.weights + self.bias)
            error = probabilities - y
            weighted_error = weights * error
            self.weights -= self.learning_rate * (x.T @ weighted_error + self.l2 * self.weights)
            self.bias -= self.learning_rate * float(weighted_error.sum())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x @ self.weights + self.bias)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


def impurity(y: np.ndarray, criterion: str = "gini") -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    if criterion == "entropy":
        return float(-np.sum(probabilities * np.log2(probabilities + 1e-15)))
    return float(1.0 - np.sum(probabilities ** 2))


def candidate_thresholds(values: np.ndarray, max_thresholds: int = 20) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) < 2:
        return np.array([], dtype=float)
    midpoints = (unique[:-1] + unique[1:]) / 2
    if len(midpoints) > max_thresholds:
        positions = np.linspace(0, len(midpoints) - 1, max_thresholds, dtype=int)
        midpoints = midpoints[positions]
    return midpoints


def best_tree_split(x: np.ndarray, y: np.ndarray, features: np.ndarray | None = None, criterion: str = "gini", min_samples_split: int = 2, max_thresholds: int = 20) -> tuple[int | None, float | None, float]:
    if len(y) < min_samples_split or len(np.unique(y)) < 2:
        return None, None, 0.0
    feature_indices = np.arange(x.shape[1]) if features is None else features
    parent = impurity(y, criterion)
    best_gain, best_feature, best_threshold = 0.0, None, None
    for feature in feature_indices:
        for threshold in candidate_thresholds(x[:, feature], max_thresholds):
            left = x[:, feature] <= threshold
            if not left.any() or left.all():
                continue
            gain = parent - (left.mean() * impurity(y[left], criterion) + (~left).mean() * impurity(y[~left], criterion))
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, int(feature), float(threshold)
    return best_feature, best_threshold, best_gain


class DecisionTreeScratch:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 8, criterion: str = "gini", max_features: int | None = None, max_thresholds: int = 18):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.max_thresholds = max_thresholds

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "DecisionTreeScratch":
        self.feature_names = feature_names or [f"x{i}" for i in range(x.shape[1])]
        self.importances = np.zeros(x.shape[1], dtype=float)
        self.root = self._grow(x, y, 0)
        total = self.importances.sum()
        if total:
            self.importances /= total
        return self

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> dict[str, Any]:
        counts = np.bincount(y, minlength=2)
        prediction = int(np.argmax(counts))
        node: dict[str, Any] = {"prediction": prediction, "count": len(y), "positive_rate": float(y.mean()) if len(y) else 0.0}
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) < 2:
            return node
        features = np.arange(x.shape[1])
        if self.max_features is not None and self.max_features < len(features):
            features = np.random.choice(features, self.max_features, replace=False)
        feature, threshold, gain = best_tree_split(x, y, features, self.criterion, self.min_samples_split, self.max_thresholds)
        if feature is None or threshold is None:
            return node
        mask = x[:, feature] <= threshold
        if not mask.any() or mask.all():
            return node
        self.importances[feature] += gain * len(y)
        node.update({"feature": feature, "threshold": threshold, "left": self._grow(x[mask], y[mask], depth + 1), "right": self._grow(x[~mask], y[~mask], depth + 1)})
        return node

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(row, self.root)[1] for row in x])

    def _predict_one(self, row: np.ndarray, node: dict[str, Any]) -> tuple[int, float]:
        if "feature" not in node:
            return node["prediction"], node["positive_rate"]
        branch = "left" if row[node["feature"]] <= node["threshold"] else "right"
        return self._predict_one(row, node[branch])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)

    def decision_path(self, row: np.ndarray) -> list[str]:
        path: list[str] = []
        node = self.root
        while "feature" in node:
            feature = node["feature"]
            goes_left = row[feature] <= node["threshold"]
            path.append(f"{self.feature_names[feature]} {'<=' if goes_left else '>'} {node['threshold']:.3f}")
            node = node["left" if goes_left else "right"]
        path.append(f"predict {node['prediction']} (positive rate {node['positive_rate']:.2f})")
        return path


class RandomForestScratch:
    def __init__(self, n_trees: int = 15, max_depth: int = 4, min_samples_split: int = 8, seed: int = 42):
        self.n_trees, self.max_depth, self.min_samples_split, self.seed = n_trees, max_depth, min_samples_split, seed

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "RandomForestScratch":
        rng = np.random.default_rng(self.seed)
        self.trees: list[DecisionTreeScratch] = []
        self.importances = np.zeros(x.shape[1])
        for tree_index in range(self.n_trees):
            indices = rng.integers(0, len(y), len(y))
            tree_seed = self.seed + tree_index
            np.random.seed(tree_seed)
            tree = DecisionTreeScratch(self.max_depth, self.min_samples_split, max_features=max(1, int(math.sqrt(x.shape[1]))))
            tree.fit(x[indices], y[indices], feature_names)
            self.trees.append(tree)
            self.importances += tree.importances
        self.importances /= max(len(self.trees), 1)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.mean([tree.predict_proba(x) for tree in self.trees], axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


class GradientBoostingScratch:
    """Logistic gradient boosting with depth-one trees as educational weak learners."""

    def __init__(self, n_estimators: int = 25, learning_rate: float = 0.15, max_thresholds: int = 16):
        self.n_estimators, self.learning_rate, self.max_thresholds = n_estimators, learning_rate, max_thresholds

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostingScratch":
        prevalence = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        self.base_score = float(np.log(prevalence / (1 - prevalence)))
        self.scores = np.full(len(y), self.base_score)
        self.stumps: list[tuple[int, float, float, float]] = []
        self.importances = np.zeros(x.shape[1])
        for _ in range(self.n_estimators):
            probabilities = 1 / (1 + np.exp(-np.clip(self.scores, -30, 30)))
            residual = y - probabilities
            best_gain, best = -1.0, None
            for feature in range(x.shape[1]):
                for threshold in candidate_thresholds(x[:, feature], self.max_thresholds):
                    left = x[:, feature] <= threshold
                    if not left.any() or left.all():
                        continue
                    left_value = float(residual[left].mean())
                    right_value = float(residual[~left].mean())
                    error = float(((residual[left] - left_value) ** 2).sum() + ((residual[~left] - right_value) ** 2).sum())
                    gain = float((residual ** 2).sum() - error)
                    if gain > best_gain:
                        best_gain, best = gain, (feature, float(threshold), left_value, right_value)
            if best is None:
                break
            feature, threshold, left_value, right_value = best
            self.stumps.append(best)
            self.importances[feature] += max(best_gain, 0)
            self.scores += self.learning_rate * np.where(x[:, feature] <= threshold, left_value, right_value)
        total = self.importances.sum()
        if total:
            self.importances /= total
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = np.full(len(x), self.base_score)
        for feature, threshold, left_value, right_value in self.stumps:
            scores += self.learning_rate * np.where(x[:, feature] <= threshold, left_value, right_value)
        return 1 / (1 + np.exp(-np.clip(scores, -30, 30)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


class SVMLinearScratch:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1200, regularization: float = 0.02):
        self.learning_rate, self.epochs, self.regularization = learning_rate, epochs, regularization

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SVMLinearScratch":
        labels = np.where(y == 1, 1.0, -1.0)
        self.weights = np.zeros(x.shape[1])
        self.bias = 0.0
        for epoch in range(self.epochs):
            rate = self.learning_rate / (1 + 0.002 * epoch)
            margins = labels * (x @ self.weights + self.bias)
            active = margins < 1
            gradient_w = self.regularization * self.weights - (x[active].T @ labels[active]) / len(y)
            gradient_b = -float(labels[active].sum()) / len(y)
            self.weights -= rate * gradient_w
            self.bias -= rate * gradient_b
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.decision_function(x) >= 0).astype(int)


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[list[float], list[float], float]:
    thresholds = sorted(set([float("inf"), *scores.tolist(), float("-inf")]), reverse=True)
    positives, negatives = int(np.sum(y_true == 1)), int(np.sum(y_true == 0))
    tpr: list[float] = []
    fpr: list[float] = []
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y_true == 1)))
        fp = int(np.sum(predicted & (y_true == 0)))
        tpr.append(tp / positives if positives else 0.0)
        fpr.append(fp / negatives if negatives else 0.0)
    order = np.argsort(fpr)
    auc = float(np.trapezoid(np.array(tpr)[order], np.array(fpr)[order]))
    return fpr, tpr, auc


def confusion_frame(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])


def kernel_matrix(x: np.ndarray, z: np.ndarray, kind: str, degree: int = 3, gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    """Build polynomial or RBF Gram matrices from their defining equations."""
    dot_products = x @ z.T
    if kind == "polynomial":
        return (dot_products / x.shape[1] + coef0) ** degree
    if kind == "rbf":
        scale = gamma if gamma is not None else 1.0 / x.shape[1]
        squared_norms_x = np.sum(x * x, axis=1)[:, None]
        squared_norms_z = np.sum(z * z, axis=1)[None, :]
        squared_distances = np.maximum(squared_norms_x + squared_norms_z - 2 * dot_products, 0.0)
        return np.exp(-scale * squared_distances)
    raise ValueError("kind must be 'polynomial' or 'rbf'")


class KernelSVMScratch:
    """Kernel SVM trained with manually implemented sequential minimal optimization."""

    def __init__(self, kind: str, regularization: float = 1.0, epochs: int = 20, degree: int = 2, gamma: float | None = None):
        if kind not in {"polynomial", "rbf"}:
            raise ValueError("kind must be 'polynomial' or 'rbf'")
        self.kind = kind
        self.regularization = regularization
        self.epochs = epochs
        self.degree = degree
        self.gamma = gamma

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelSVMScratch":
        self.training_x = x.copy()
        self.signed_y = np.where(y == 1, 1.0, -1.0)
        gram = kernel_matrix(x, x, self.kind, self.degree, self.gamma)
        self.coefficients = np.zeros(len(y), dtype=float)
        self.bias = 0.0
        errors = -self.signed_y.copy()
        no_change_passes = 0
        for _ in range(self.epochs):
            changed = 0
            for first in range(len(y)):
                first_error = errors[first]
                first_label = self.signed_y[first]
                first_alpha = self.coefficients[first]
                violates_margin = first_label * first_error < -1e-3 and first_alpha < self.regularization - 1e-8
                can_shrink = first_label * first_error > 1e-3 and first_alpha > 1e-8
                if not (violates_margin or can_shrink):
                    continue

                differences = np.abs(first_error - errors)
                differences[first] = -1.0
                second = int(np.argmax(differences))
                second_error = errors[second]
                second_label = self.signed_y[second]
                first_old = self.coefficients[first]
                second_old = self.coefficients[second]
                if first_label != second_label:
                    lower = max(0.0, second_old - first_old)
                    upper = min(self.regularization, self.regularization + second_old - first_old)
                else:
                    lower = max(0.0, first_old + second_old - self.regularization)
                    upper = min(self.regularization, first_old + second_old)
                if upper - lower < 1e-10:
                    continue

                eta = 2 * gram[first, second] - gram[first, first] - gram[second, second]
                if eta >= -1e-12:
                    continue
                second_new = second_old - second_label * (first_error - second_error) / eta
                second_new = float(np.clip(second_new, lower, upper))
                if abs(second_new - second_old) < 1e-6:
                    continue
                first_new = first_old + first_label * second_label * (second_old - second_new)
                self.coefficients[first] = first_new
                self.coefficients[second] = second_new

                bias_first = self.bias - first_error - first_label * (first_new - first_old) * gram[first, first] - second_label * (second_new - second_old) * gram[first, second]
                bias_second = self.bias - second_error - first_label * (first_new - first_old) * gram[first, second] - second_label * (second_new - second_old) * gram[second, second]
                if 1e-8 < first_new < self.regularization - 1e-8:
                    new_bias = bias_first
                elif 1e-8 < second_new < self.regularization - 1e-8:
                    new_bias = bias_second
                else:
                    new_bias = (bias_first + bias_second) / 2
                errors += first_label * (first_new - first_old) * gram[:, first]
                errors += second_label * (second_new - second_old) * gram[:, second]
                errors += new_bias - self.bias
                self.bias = float(new_bias)
                changed += 1
            no_change_passes = no_change_passes + 1 if changed == 0 else 0
            if no_change_passes >= 3:
                break
        self.support_mask = self.coefficients > 1e-8
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return kernel_matrix(x, self.training_x, self.kind, self.degree, self.gamma) @ (self.coefficients * self.signed_y) + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.decision_function(x) >= 0).astype(int)


class KDTreeScratch:
    """Balanced KD-tree with bounding-box pruning for exact nearest neighbors."""

    def __init__(self, leaf_size: int = 16):
        self.leaf_size = leaf_size

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KDTreeScratch":
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.root = self._build(np.arange(len(self.x)))
        return self

    def _build(self, indices: np.ndarray) -> dict[str, Any]:
        points = self.x[indices]
        node: dict[str, Any] = {"low": points.min(axis=0), "high": points.max(axis=0)}
        if len(indices) <= self.leaf_size:
            node["indices"] = indices
            return node
        axis = int(np.argmax(node["high"] - node["low"]))
        order = indices[np.argsort(self.x[indices, axis], kind="mergesort")]
        midpoint = len(order) // 2
        node["left"] = self._build(order[:midpoint])
        node["right"] = self._build(order[midpoint:])
        return node

    @staticmethod
    def _box_distance_squared(point: np.ndarray, node: dict[str, Any]) -> float:
        delta = np.maximum(np.maximum(node["low"] - point, point - node["high"]), 0.0)
        return float(delta @ delta)

    def query(self, point: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k < 1 or k > len(self.x):
            raise ValueError("k must be between 1 and the training row count")
        nearest: list[tuple[float, int, int]] = []
        self.last_nodes_visited = 0

        def visit(node: dict[str, Any]) -> None:
            self.last_nodes_visited += 1
            if len(nearest) == k and self._box_distance_squared(point, node) > -nearest[0][0]:
                return
            if "indices" in node:
                for index in node["indices"]:
                    delta = point - self.x[index]
                    distance_squared = float(delta @ delta)
                    entry = (-distance_squared, -int(index), int(index))
                    if len(nearest) < k:
                        heapq.heappush(nearest, entry)
                    elif entry > nearest[0]:
                        heapq.heapreplace(nearest, entry)
                return
            children = [node["left"], node["right"]]
            children.sort(key=lambda child: self._box_distance_squared(point, child))
            visit(children[0])
            visit(children[1])

        visit(self.root)
        ordered = sorted(nearest, key=lambda item: (-item[0], item[2]))
        indices = np.array([item[2] for item in ordered], dtype=int)
        distances = np.sqrt(np.array([-item[0] for item in ordered], dtype=float))
        return indices, distances


class KNearestNeighborsScratch:
    """KD-tree-backed k-NN with deterministic, distance-aware vote tie breaks."""

    def __init__(self, k: int = 5, leaf_size: int = 16):
        self.k = k
        self.leaf_size = leaf_size

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNearestNeighborsScratch":
        self.tree = KDTreeScratch(self.leaf_size).fit(x, y)
        return self

    def predict_one(self, point: np.ndarray) -> int:
        indices, distances = self.tree.query(point, self.k)
        labels = self.tree.y[indices]
        values, counts = np.unique(labels, return_counts=True)
        tied = values[counts == counts.max()]
        if len(tied) == 1:
            return int(tied[0])
        distance_sums = {int(label): float(distances[labels == label].sum()) for label in tied}
        smallest_sum = min(distance_sums.values())
        closest = [label for label, total in distance_sums.items() if np.isclose(total, smallest_sum)]
        if len(closest) == 1:
            return closest[0]
        nearest_label = int(labels[0])
        return nearest_label if nearest_label in closest else min(closest)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(row) for row in x], dtype=int)


class MLPClassifierScratch:
    """Binary MLP with optional inverted dropout and manually differentiated batch normalization."""

    def __init__(self, hidden_layers: tuple[int, ...] = (16, 8), learning_rate: float = 0.01, epochs: int = 100, batch_size: int = 32, dropout_rate: float = 0.0, batch_normalization: bool = False, seed: int = 42):
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: tuple[np.ndarray, np.ndarray] | None = None, scheduler: dict[str, float | int] | None = None) -> "MLPClassifierScratch":
        rng = np.random.default_rng(self.seed)
        layer_sizes = (x.shape[1], *self.hidden_layers, 1)
        self.weights = [rng.normal(0, np.sqrt(2 / layer_sizes[i]), (layer_sizes[i], layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
        hidden_count = len(self.hidden_layers)
        self.gamma = [np.ones((1, size)) for size in self.hidden_layers]
        self.beta = [np.zeros((1, size)) for size in self.hidden_layers]
        self.running_mean = [np.zeros((1, size)) for size in self.hidden_layers]
        self.running_var = [np.ones((1, size)) for size in self.hidden_layers]
        self.loss_history: list[float] = []
        self.validation_loss_history: list[float] = []
        self.learning_rate_history: list[float] = []
        self.scheduler_events: list[dict[str, float | int]] = []
        self.update_norm_history: list[float] = []
        self.batch_variance_history: list[float] = []
        epsilon, momentum = 1e-5, 0.9
        labels = y.reshape(-1, 1).astype(float)
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        scheduler_factor = float(scheduler.get("factor", 0.5)) if scheduler else 0.5
        scheduler_patience = int(scheduler.get("patience", 5)) if scheduler else 5
        scheduler_min_delta = float(scheduler.get("min_delta", 1e-4)) if scheduler else 1e-4
        scheduler_min_rate = float(scheduler.get("min_learning_rate", 1e-4)) if scheduler else 1e-4
        if scheduler and (not 0 < scheduler_factor < 1 or scheduler_patience < 1):
            raise ValueError("scheduler factor must be in (0, 1) and patience must be positive")

        for _ in range(self.epochs):
            epoch_learning_rate = self.learning_rate
            epoch_update_squared = 0.0
            order = rng.permutation(len(x))
            epoch_loss = 0.0
            for start in range(0, len(x), self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                activations = [x[batch_idx]]
                caches: list[dict[str, np.ndarray | None]] = []
                for layer in range(hidden_count):
                    linear = activations[-1] @ self.weights[layer] + self.biases[layer]
                    if self.batch_normalization:
                        mean = linear.mean(axis=0, keepdims=True)
                        variance = linear.var(axis=0, keepdims=True)
                        inverse_std = 1.0 / np.sqrt(variance + epsilon)
                        normalized = (linear - mean) * inverse_std
                        scaled = self.gamma[layer] * normalized + self.beta[layer]
                        self.running_mean[layer] = momentum * self.running_mean[layer] + (1 - momentum) * mean
                        self.running_var[layer] = momentum * self.running_var[layer] + (1 - momentum) * variance
                        self.batch_variance_history.append(float(np.mean(variance)))
                    else:
                        normalized, inverse_std, scaled = linear, None, linear
                    activated = np.maximum(scaled, 0.0)
                    mask = None
                    if self.dropout_rate and len(batch_idx) > 1:
                        mask = (rng.random(activated.shape) >= self.dropout_rate).astype(float) / (1 - self.dropout_rate)
                        activated = activated * mask
                    caches.append({"normalized": normalized, "inverse_std": inverse_std, "scaled": scaled, "mask": mask})
                    activations.append(activated)

                logits = activations[-1] @ self.weights[-1] + self.biases[-1]
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
                batch_y = labels[batch_idx]
                epoch_loss += float(-np.sum(batch_y * np.log(probabilities + 1e-12) + (1 - batch_y) * np.log(1 - probabilities + 1e-12)))
                gradient = (probabilities - batch_y) / len(batch_idx)
                weight_gradients = [None] * len(self.weights)
                bias_gradients = [None] * len(self.biases)
                gamma_gradients = [None] * hidden_count
                beta_gradients = [None] * hidden_count
                weight_gradients[-1] = activations[-1].T @ gradient
                bias_gradients[-1] = gradient.sum(axis=0, keepdims=True)
                gradient = gradient @ self.weights[-1].T

                for layer in range(hidden_count - 1, -1, -1):
                    cache = caches[layer]
                    mask = cache["mask"]
                    if mask is not None:
                        gradient *= mask
                    gradient *= cache["scaled"] > 0
                    if self.batch_normalization:
                        normalized = cache["normalized"]
                        inverse_std = cache["inverse_std"]
                        gamma_gradients[layer] = np.sum(gradient * normalized, axis=0, keepdims=True)
                        beta_gradients[layer] = gradient.sum(axis=0, keepdims=True)
                        normalized_gradient = gradient * self.gamma[layer]
                        gradient = inverse_std / len(batch_idx) * (len(batch_idx) * normalized_gradient - normalized_gradient.sum(axis=0, keepdims=True) - normalized * np.sum(normalized_gradient * normalized, axis=0, keepdims=True))
                    weight_gradients[layer] = activations[layer].T @ gradient
                    bias_gradients[layer] = gradient.sum(axis=0, keepdims=True)
                    gradient = gradient @ self.weights[layer].T

                for layer in range(len(self.weights)):
                    epoch_update_squared += float(np.sum((epoch_learning_rate * weight_gradients[layer]) ** 2) + np.sum((epoch_learning_rate * bias_gradients[layer]) ** 2))
                    self.weights[layer] -= epoch_learning_rate * weight_gradients[layer]
                    self.biases[layer] -= epoch_learning_rate * bias_gradients[layer]
                if self.batch_normalization:
                    for layer in range(hidden_count):
                        epoch_update_squared += float(np.sum((epoch_learning_rate * gamma_gradients[layer]) ** 2) + np.sum((epoch_learning_rate * beta_gradients[layer]) ** 2))
                        self.gamma[layer] -= epoch_learning_rate * gamma_gradients[layer]
                        self.beta[layer] -= epoch_learning_rate * beta_gradients[layer]
            self.loss_history.append(epoch_loss / len(x))
            self.learning_rate_history.append(epoch_learning_rate)
            self.update_norm_history.append(float(np.sqrt(epoch_update_squared)))
            if validation_data is not None:
                validation_x, validation_y = validation_data
                validation_probabilities = np.clip(self.predict_proba(validation_x), 1e-12, 1 - 1e-12)
                validation_y = validation_y.astype(float)
                validation_loss = -np.mean(validation_y * np.log(validation_probabilities) + (1 - validation_y) * np.log(1 - validation_probabilities))
                self.validation_loss_history.append(float(validation_loss))
                if scheduler:
                    if validation_loss < best_validation_loss - scheduler_min_delta:
                        best_validation_loss = float(validation_loss)
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    if epochs_without_improvement >= scheduler_patience:
                        next_rate = max(scheduler_min_rate, self.learning_rate * scheduler_factor)
                        if next_rate < self.learning_rate:
                            self.learning_rate = next_rate
                            self.scheduler_events.append({"epoch": len(self.loss_history), "learning_rate": next_rate, "validation_loss": float(validation_loss)})
                        epochs_without_improvement = 0
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        activation = x
        for layer in range(len(self.hidden_layers)):
            linear = activation @ self.weights[layer] + self.biases[layer]
            if self.batch_normalization:
                normalized = (linear - self.running_mean[layer]) / np.sqrt(self.running_var[layer] + 1e-5)
                linear = self.gamma[layer] * normalized + self.beta[layer]
            activation = np.maximum(linear, 0.0)
        logits = activation @ self.weights[-1] + self.biases[-1]
        return 1.0 / (1.0 + np.exp(-np.clip(logits.ravel(), -35, 35)))

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


def architecture_features(architecture: tuple[int, ...]) -> np.ndarray:
    padded = list(architecture[:3]) + [0] * (3 - len(architecture[:3]))
    return np.array([len(architecture) / 3, *[width / 32 for width in padded]], dtype=float)


def bayesian_architecture_search(candidates: list[tuple[int, ...]], objective, initial: list[tuple[int, ...]] | None = None, evaluations: int = 6, seed: int = 42) -> list[dict[str, float | tuple[int, ...]]]:
    """Choose discrete architectures by Gaussian-process expected improvement."""
    rng = np.random.default_rng(seed)
    initial = initial or candidates[:2]
    evaluated: list[dict[str, float | tuple[int, ...]]] = []
    remaining = [candidate for candidate in candidates if candidate not in initial]
    for architecture in initial[:evaluations]:
        evaluated.append({"architecture": architecture, "score": float(objective(architecture))})
    while remaining and len(evaluated) < evaluations:
        observed_x = np.vstack([architecture_features(row["architecture"]) for row in evaluated])
        observed_y = np.array([row["score"] for row in evaluated])
        candidate_x = np.vstack([architecture_features(candidate) for candidate in remaining])
        distances = np.sum((observed_x[:, None, :] - observed_x[None, :, :]) ** 2, axis=2)
        kernel = np.exp(-0.5 * distances) + np.eye(len(observed_y)) * 1e-6
        cross_distance = np.sum((candidate_x[:, None, :] - observed_x[None, :, :]) ** 2, axis=2)
        cross_kernel = np.exp(-0.5 * cross_distance)
        inverse = np.linalg.inv(kernel)
        mean = cross_kernel @ inverse @ observed_y
        variance = np.maximum(1.0 - np.sum((cross_kernel @ inverse) * cross_kernel, axis=1), 1e-9)
        std = np.sqrt(variance)
        best = float(np.max(observed_y))
        z = (mean - best - 0.01) / std
        cdf = 0.5 * (1 + np.vectorize(math.erf)(z / np.sqrt(2)))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        expected_improvement = (mean - best - 0.01) * cdf + std * pdf
        expected_improvement += rng.random(len(remaining)) * 1e-12
        chosen_index = int(np.argmax(expected_improvement))
        architecture = remaining.pop(chosen_index)
        evaluated.append({"architecture": architecture, "score": float(objective(architecture))})
    return evaluated
