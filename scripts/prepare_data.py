"""De-duplicate the SMS data and write the train / validation / test split.

Usage:
    python scripts/prepare_data.py

Writes data/processed/sms_{train,val,test}.csv (local, not committed) and
results/split_summary.json, which records the counts so a reader can check them
against the README without downloading anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spamedge.data import load_sms  # noqa: E402
from spamedge.split import SPLIT_SEED, deduplicate, split_train_val_test  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--summary", type=Path, default=Path("results/split_summary.json"))
    args = parser.parse_args()

    sms = load_sms(args.raw / "sms" / "SMSSpamCollection")
    unique = deduplicate(sms)
    train, val, test = split_train_val_test(unique)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_rows": len(sms),
        "source_spam": int(sms["label"].sum()),
        "after_dedup_rows": len(unique),
        "after_dedup_spam": int(unique["label"].sum()),
        "split_seed": SPLIT_SEED,
        "parts": {},
    }
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(args.out / f"sms_{name}.csv", index=False)
        summary["parts"][name] = {
            "rows": len(part),
            "spam": int(part["label"].sum()),
            "spam_rate": round(float(part["label"].mean()), 4),
        }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
