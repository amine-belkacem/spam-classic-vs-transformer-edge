"""Train the TF-IDF + logistic regression baseline, picking its settings on validation.

Usage:
    python scripts/train_classic.py

Every (C, class_weight) pair of the grid is trained on train and scored on
validation (spam F1 at threshold 0.5). The best pair is kept and saved to
models/classic.joblib. The test set is not touched here: it is only read by
scripts/evaluate.py, once both models are frozen.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spamedge.classic import C_GRID, CLASS_WEIGHT_GRID, build_classic, spam_probability  # noqa: E402
from spamedge.metrics import spam_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", type=Path, default=Path("models/classic.joblib"))
    parser.add_argument("--log", type=Path, default=Path("results/classic_validation.json"))
    args = parser.parse_args()

    train = pd.read_csv(args.data / "sms_train.csv", keep_default_na=False)
    val = pd.read_csv(args.data / "sms_val.csv", keep_default_na=False)

    runs = []
    for c, cw in itertools.product(C_GRID, CLASS_WEIGHT_GRID):
        model = build_classic(c, cw)
        t0 = time.perf_counter()
        model.fit(train["text"], train["label"])
        fit_s = time.perf_counter() - t0
        m = spam_metrics(val["label"].to_numpy(), spam_probability(model, val["text"]))
        runs.append({"C": c, "class_weight": cw, "fit_seconds": round(fit_s, 2),
                     **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        print(f"C={c:<6} class_weight={str(cw):9s} val F1={m['f1']:.4f}  "
              f"P={m['precision']:.4f} R={m['recall']:.4f}  fit {fit_s:.1f}s")

    # Ties on F1 go to the higher PR-AUC, then to the smaller C (stronger regularisation).
    best = max(runs, key=lambda r: (r["f1"], r["pr_auc"], -r["C"]))
    print(f"best: C={best['C']} class_weight={best['class_weight']} val F1={best['f1']}")

    model = build_classic(best["C"], best["class_weight"])
    model.fit(train["text"], train["label"])
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps({"best": best, "grid": runs}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
