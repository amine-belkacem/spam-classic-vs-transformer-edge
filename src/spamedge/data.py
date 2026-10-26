"""Loading and sanity checks for the two UCI datasets.

Both loaders return the same frame: one row per message, a `text` column and an
integer `label` column (1 = spam, 0 = ham). Everything downstream (splits,
models, benchmarks) only ever sees that shape, so the SMS and YouTube data can be
swapped for the domain-shift experiment without special cases.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Counts published on the UCI pages, checked again on the files I downloaded on 2026-10-26.
SMS_EXPECTED_ROWS = 5574
SMS_EXPECTED_SPAM = 747
YOUTUBE_EXPECTED_ROWS = 1956
YOUTUBE_EXPECTED_SPAM = 1005

SMS_LABELS = {"ham": 0, "spam": 1}

# The YouTube CSVs also carry the comment author's name, a comment id and a date.
# Only the text and the label are needed, so the other columns are never loaded:
# no personal data reaches the frames, the results or the figures.
YOUTUBE_COLUMNS = ["CONTENT", "CLASS"]

_HASH_CHUNK_BYTES = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def parse_sms_lines(lines: Iterable[str]) -> pd.DataFrame:
    """Parse lines of the form '<ham|spam>\\t<message>'.

    Blank lines are skipped. Anything else that does not match the format raises,
    with the line number, instead of being silently dropped: a parsing bug here
    would shift every number in the project.
    """
    texts, labels = [], []
    for lineno, line in enumerate(lines, start=1):
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        tag, sep, text = line.partition("\t")
        if not sep:
            raise ValueError(f"line {lineno}: no tab between label and message: {line[:60]!r}")
        if tag not in SMS_LABELS:
            raise ValueError(f"line {lineno}: unknown label {tag!r}, expected one of {sorted(SMS_LABELS)}")
        texts.append(text)
        labels.append(SMS_LABELS[tag])
    return pd.DataFrame({"text": texts, "label": pd.Series(labels, dtype="int64")})


def check_counts(df: pd.DataFrame, expected_rows: int, expected_spam: int, name: str) -> None:
    rows, spam = len(df), int(df["label"].sum())
    if (rows, spam) != (expected_rows, expected_spam):
        raise ValueError(
            f"{name}: got {rows} rows / {spam} spam, expected {expected_rows} / {expected_spam}"
        )


def load_sms(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        df = parse_sms_lines(f)
    check_counts(df, SMS_EXPECTED_ROWS, SMS_EXPECTED_SPAM, "SMS Spam Collection")
    return df


def video_name(csv_path: Path) -> str:
    """'Youtube03-LMFAO.csv' -> 'LMFAO'."""
    stem = csv_path.stem
    _, sep, name = stem.partition("-")
    if not sep or not name:
        raise ValueError(f"unexpected YouTube file name: {csv_path.name}")
    return name


def read_youtube_csv(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, usecols=YOUTUBE_COLUMNS, encoding="utf-8")
    bad = ~raw["CLASS"].isin([0, 1])
    if bad.any():
        raise ValueError(f"{csv_path.name}: {int(bad.sum())} rows with CLASS outside 0/1")
    return pd.DataFrame({
        "text": raw["CONTENT"].astype(str),
        "label": raw["CLASS"].astype("int64"),
        "video": video_name(csv_path),
    })


def load_youtube(csv_dir: Path) -> pd.DataFrame:
    files = sorted(csv_dir.glob("Youtube*.csv"))
    if not files:
        raise FileNotFoundError(f"no Youtube*.csv file in {csv_dir}")
    df = pd.concat([read_youtube_csv(f) for f in files], ignore_index=True)
    check_counts(df, YOUTUBE_EXPECTED_ROWS, YOUTUBE_EXPECTED_SPAM, "YouTube Spam Collection")
    return df
