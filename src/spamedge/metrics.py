"""Spam-class metrics and their bootstrap confidence intervals.

Spam is the positive class everywhere. Precision answers "when the filter says
spam, how often is it right?" (low precision = real messages thrown away),
recall answers "how much of the spam does it catch?".

The test set has only 96 spam messages, so one more or one fewer caught spam
moves recall by about one point. Every number is therefore reported with a 95 %
bootstrap interval, and both models are scored on the *same* resamples so their
difference gets its own interval (a paired comparison).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

DEFAULT_THRESHOLD = 0.5
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95
METRIC_NAMES = ("precision", "recall", "f1", "pr_auc")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) with spam = 1 as the positive class."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """A ratio with an empty denominator is defined as 0 (e.g. no spam predicted -> precision 0)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def spam_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Metrics at a decision threshold (score >= threshold -> spam), plus the threshold-free PR-AUC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    tp, fp, fn, tn = confusion_counts(y_true, y_score >= threshold)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def stratified_resample_indices(y_true: np.ndarray, n_resamples: int = N_RESAMPLES,
                                seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Bootstrap indices that keep the number of spam and ham messages fixed.

    A plain bootstrap could draw a resample with very few spam messages, and
    recall would then swing for reasons that have nothing to do with the model.
    Returns an array of shape (n_resamples, len(y_true)).
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("the bootstrap needs both classes in y_true")
    return np.hstack([
        rng.choice(pos, size=(n_resamples, len(pos)), replace=True),
        rng.choice(neg, size=(n_resamples, len(neg)), replace=True),
    ])


def bootstrap_metrics(y_true: np.ndarray, y_score: np.ndarray, indices: np.ndarray,
                      threshold: float = DEFAULT_THRESHOLD) -> dict[str, np.ndarray]:
    """Each metric computed on every resample: {name: array of shape (n_resamples,)}."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    out = {name: np.empty(len(indices)) for name in METRIC_NAMES}
    for i, idx in enumerate(indices):
        m = spam_metrics(y_true[idx], y_score[idx], threshold)
        for name in METRIC_NAMES:
            out[name][i] = m[name]
    return out


def percentile_interval(samples: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)
