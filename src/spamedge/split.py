"""De-duplication and the train / validation / test split.

The SMS file repeats some messages word for word (5,574 lines, 5,171 distinct
texts), and a dozen more only differ by case or spacing. If one copy lands in
train and another in test, the test score measures memory, not generalisation.
So duplicates are removed on a normalised key *before* splitting.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def dedup_key(text: str) -> str:
    """Lower-case and collapse every run of whitespace to one space."""
    return " ".join(text.split()).lower()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each normalised text.

    Raises if two copies of the same text carry different labels: that would be
    a labelling problem to look at, not something to resolve by picking one.
    """
    keys = df["text"].map(dedup_key)
    labels_per_key = df.groupby(keys)["label"].nunique()
    conflicts = labels_per_key[labels_per_key > 1]
    if len(conflicts):
        raise ValueError(f"{len(conflicts)} texts appear with both labels, e.g. {conflicts.index[0][:60]!r}")
    return df.loc[~keys.duplicated()].reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split, so the spam rate is the same in the three parts."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError(f"invalid fractions: val={val_fraction}, test={test_fraction}")
    train_val, test = train_test_split(
        df, test_size=test_fraction, stratify=df["label"], random_state=seed
    )
    # val_fraction is a share of the whole dataset, so rescale it to what is left after the test cut.
    train, val = train_test_split(
        train_val,
        test_size=val_fraction / (1 - test_fraction),
        stratify=train_val["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
