"""Small educational ML implementations used by follow-on heart-disease tasks."""

from __future__ import annotations

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

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        self.weights = np.zeros(x.shape[1], dtype=float)
        self.bias = 0.0
        for _ in range(self.iterations):
            probabilities = self._sigmoid(x @ self.weights + self.bias)
            error = probabilities - y
            self.weights -= self.learning_rate * (x.T @ error / len(y) + self.l2 * self.weights)
            self.bias -= self.learning_rate * float(error.mean())
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
