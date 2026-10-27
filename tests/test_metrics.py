"""Tests for the metrics. Every expected value is worked out by hand in the comments."""

import numpy as np
import pytest

from spamedge.metrics import (
    bootstrap_metrics,
    confusion_counts,
    percentile_interval,
    precision_recall_f1,
    spam_metrics,
    stratified_resample_indices,
)

# 5 spam then 5 ham. The prediction catches 3 spam, misses 2, and flags 1 ham.
Y_TRUE = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
Y_PRED = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])


def close(got, expected, tol=1e-9):
    assert abs(got - expected) <= tol, f"ECHEC : obtenu {got}, attendu {expected}"


def test_confusion_counts():
    expected = (3, 1, 2, 4)  # tp, fp, fn, tn
    got = confusion_counts(Y_TRUE, Y_PRED)
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


def test_precision_recall_f1_by_hand():
    # precision = 3 / (3 + 1) = 0.75 ; recall = 3 / (3 + 2) = 0.6
    # F1 = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 0.9 / 1.35 = 0.6667
    p, r, f1 = precision_recall_f1(3, 1, 2)
    close(p, 0.75)
    close(r, 0.6)
    close(f1, 2 / 3)


def test_no_spam_predicted_gives_zero_instead_of_a_division_error():
    expected = (0.0, 0.0, 0.0)
    got = precision_recall_f1(tp=0, fp=0, fn=5)
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


def test_score_equal_to_the_threshold_counts_as_spam():
    m = spam_metrics(np.array([1, 0]), np.array([0.5, 0.49]), threshold=0.5)
    got = (m["tp"], m["fp"], m["fn"], m["tn"])
    expected = (1, 0, 0, 1)
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


def test_pr_auc_by_hand():
    # Ranked by score: spam (0.9), ham (0.8), spam (0.7), ham (0.1).
    # Average precision = sum over each spam of (recall step) * (precision there):
    #   first spam:  recall 0 -> 0.5, precision 1/1 -> 0.5 * 1   = 0.5
    #   second spam: recall 0.5 -> 1, precision 2/3 -> 0.5 * 2/3 = 0.3333
    # total = 0.8333
    m = spam_metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]))
    close(m["pr_auc"], 0.5 + 0.5 * 2 / 3)


def test_resamples_keep_the_class_counts():
    idx = stratified_resample_indices(Y_TRUE, n_resamples=50, seed=1)
    expected_shape = (50, 10)
    assert idx.shape == expected_shape, f"ECHEC : forme obtenue {idx.shape}, attendue {expected_shape}"
    spam_per_resample = set(Y_TRUE[idx].sum(axis=1).tolist())
    assert spam_per_resample == {5}, f"ECHEC : spam par rééchantillon {spam_per_resample}, attendu {{5}}"


def test_perfect_classifier_has_a_zero_width_interval():
    scores = Y_TRUE.astype(float)  # spam scored 1, ham scored 0
    idx = stratified_resample_indices(Y_TRUE, n_resamples=100, seed=0)
    samples = bootstrap_metrics(Y_TRUE, scores, idx)
    for name, values in samples.items():
        got = percentile_interval(values)
        assert got == (1.0, 1.0), f"ECHEC : {name} intervalle obtenu {got}, attendu (1.0, 1.0)"


def test_percentile_interval_by_hand():
    # 101 values 0, 1, ..., 100: the 2.5th and 97.5th percentiles are 2.5 and 97.5.
    low, high = percentile_interval(np.arange(101, dtype=float))
    close(low, 2.5)
    close(high, 97.5)


def test_bootstrap_needs_both_classes():
    with pytest.raises(ValueError):
        stratified_resample_indices(np.array([1, 1, 1]))
