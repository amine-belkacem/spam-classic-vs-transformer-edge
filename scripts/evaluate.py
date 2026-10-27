"""Score the frozen models on the test set, with paired bootstrap intervals.

Usage:
    python scripts/evaluate.py --seed 0

This is the only script that reads the test split. Both models are scored on
the same 1,000 stratified resamples, so the DistilBERT - classic difference gets
its own interval: if that interval contains 0, the test set cannot tell the two
models apart on that metric.

Writes results/test_metrics.json and results/test_scores.csv (label + the two
scores per test message, no text), which the figure script reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spamedge import classic, transformer  # noqa: E402
from spamedge.metrics import (  # noqa: E402
    BOOTSTRAP_SEED,
    CI_LEVEL,
    DEFAULT_THRESHOLD,
    METRIC_NAMES,
    N_RESAMPLES,
    bootstrap_metrics,
    percentile_interval,
    spam_metrics,
    stratified_resample_indices,
)


def rounded(d: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="which DistilBERT run to evaluate")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    test = pd.read_csv(args.data / "sms_test.csv", keep_default_na=False)
    y = test["label"].to_numpy()

    lr = joblib.load(args.models / "classic.joblib")
    tok, bert = transformer.load_finetuned(args.models / f"distilbert-seed{args.seed}")
    scores = {
        "classic": classic.spam_probability(lr, test["text"]),
        "distilbert": transformer.spam_probability(bert, tok, test["text"]),
    }

    idx = stratified_resample_indices(y, N_RESAMPLES, BOOTSTRAP_SEED)
    samples = {name: bootstrap_metrics(y, s, idx, DEFAULT_THRESHOLD) for name, s in scores.items()}

    report = {
        "n_test": int(len(y)), "n_spam": int(y.sum()), "threshold": DEFAULT_THRESHOLD,
        "bootstrap": {"resamples": N_RESAMPLES, "seed": BOOTSTRAP_SEED, "stratified": True,
                      "level": CI_LEVEL},
        "distilbert_seed": args.seed,
        "models": {},
        "difference_distilbert_minus_classic": {},
    }
    for name, s in scores.items():
        point = spam_metrics(y, s, DEFAULT_THRESHOLD)
        report["models"][name] = {
            **rounded(point),
            "ci": {m: [round(v, 4) for v in percentile_interval(samples[name][m])] for m in METRIC_NAMES},
        }
    for m in METRIC_NAMES:
        diff = samples["distilbert"][m] - samples["classic"][m]
        point = report["models"]["distilbert"][m] - report["models"]["classic"][m]
        report["difference_distilbert_minus_classic"][m] = {
            "point": round(point, 4), "ci": [round(v, 4) for v in percentile_interval(diff)]}

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "test_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"label": y, **{k: np.round(v, 6) for k, v in scores.items()}}).to_csv(
        args.results / "test_scores.csv", index=False)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
