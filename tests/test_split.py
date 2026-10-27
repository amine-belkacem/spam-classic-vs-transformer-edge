"""Tests for de-duplication and the stratified split."""

import pandas as pd
import pytest

from spamedge.split import dedup_key, deduplicate, split_train_val_test


def frame(texts, labels):
    return pd.DataFrame({"text": texts, "label": pd.Series(labels, dtype="int64")})


def test_dedup_key_ignores_case_and_whitespace():
    expected = "hello world"
    got = dedup_key("  Hello\tWORLD \n")
    assert got == expected, f"ECHEC : obtenu {got!r}, attendu {expected!r}"


def test_deduplicate_keeps_the_first_copy_of_each_text():
    # 'Hi'/'hi ' and 'Win cash'/'win  CASH' are the same message -> 5 rows become 3.
    df = frame(["Hi", "hi ", "Win cash", "win  CASH", "Bye"], [0, 0, 1, 1, 0])
    out = deduplicate(df)
    expected_texts, expected_labels = ["Hi", "Win cash", "Bye"], [0, 1, 0]
    got_texts, got_labels = list(out["text"]), list(out["label"])
    assert got_texts == expected_texts, f"ECHEC : obtenu {got_texts}, attendu {expected_texts}"
    assert got_labels == expected_labels, f"ECHEC : obtenu {got_labels}, attendu {expected_labels}"


def test_deduplicate_refuses_a_text_with_two_labels():
    with pytest.raises(ValueError) as info:
        deduplicate(frame(["Call me", "CALL ME"], [0, 1]))
    assert "1 texts appear with both labels" in str(info.value), f"ECHEC : message obtenu '{info.value}'"


def synthetic(n=1000, n_spam=200):
    return frame([f"message {i}" for i in range(n)], [1] * n_spam + [0] * (n - n_spam))


def test_split_sizes_and_spam_counts_follow_the_fractions():
    # 1000 rows, 200 spam (20 %). Test = 15 % of 1000 = 150 rows, 20 % of them spam = 30.
    # Validation = 15 % of the whole = 150 rows (0.15 / 0.85 of the 850 left), 30 spam.
    # Train = 1000 - 150 - 150 = 700 rows, 200 - 30 - 30 = 140 spam.
    train, val, test = split_train_val_test(synthetic())
    expected = {"train": (700, 140), "val": (150, 30), "test": (150, 30)}
    got = {name: (len(part), int(part["label"].sum())) for name, part in
           (("train", train), ("val", val), ("test", test))}
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


def test_split_parts_are_disjoint_and_cover_everything():
    train, val, test = split_train_val_test(synthetic())
    sets = [set(p["text"]) for p in (train, val, test)]
    overlaps = len(sets[0] & sets[1]) + len(sets[0] & sets[2]) + len(sets[1] & sets[2])
    assert overlaps == 0, f"ECHEC : {overlaps} messages partagés entre les parties, attendu 0"
    covered = len(sets[0] | sets[1] | sets[2])
    assert covered == 1000, f"ECHEC : {covered} messages couverts, attendu 1000"


def test_split_is_reproducible_with_the_same_seed():
    a = split_train_val_test(synthetic(), seed=7)
    b = split_train_val_test(synthetic(), seed=7)
    for pa, pb in zip(a, b):
        assert list(pa["text"]) == list(pb["text"]), "ECHEC : deux appels avec la même graine donnent des splits différents"


def test_split_rejects_fractions_that_leave_no_training_data():
    with pytest.raises(ValueError):
        split_train_val_test(synthetic(), val_fraction=0.5, test_fraction=0.5)
