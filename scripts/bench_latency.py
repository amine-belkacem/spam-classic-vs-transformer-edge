"""Per-message latency and on-disk size of both models on this machine.

Usage:
    python scripts/bench_latency.py --seed 0 --threads 4

Protocol (same for every model):
  - one message per call (batch size 1), raw text in, spam probability out,
    so the TF-IDF transform and the tokeniser are inside the timing;
  - the messages are the test split in a fixed shuffled order, cycled;
  - 3 runs; in each run every model gets 50 warm-up calls then 1,000 timed calls
    (time.perf_counter_ns), and the order of the models alternates between runs
    so neither always goes first;
  - before each run the system CPU load is measured over 3 s; above 15 % the
    script stops, unless --allow-busy is given (the result is then flagged);
  - PyTorch is pinned to --threads threads, and the value is saved with the
    results. scikit-learn's logistic regression works on a sparse matrix here,
    which runs on one thread.

Writes results/latency_<machine tag>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spamedge import classic, transformer  # noqa: E402
from spamedge.bench import (  # noqa: E402
    MAX_IDLE_CPU_PERCENT,
    REPEATS,
    TIMED_CALLS,
    WARMUP_CALLS,
    latency_stats,
    machine_info,
    size_on_disk,
    summarize_runs,
    system_cpu_percent,
    time_single_calls,
)

ORDER_SEED = 0


def rounded(d: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="which DistilBERT run to time")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--allow-busy", action="store_true", help="measure even if the CPU is busy")
    parser.add_argument("--tag", default="pc", help="machine tag used in the output file name")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    test = pd.read_csv(args.data / "sms_test.csv", keep_default_na=False)
    rng = np.random.default_rng(ORDER_SEED)
    texts = test["text"].to_numpy()[rng.permutation(len(test))].tolist()

    lr_path = args.models / "classic.joblib"
    bert_path = args.models / f"distilbert-seed{args.seed}"
    lr = joblib.load(lr_path)
    tok, bert = transformer.load_finetuned(bert_path)
    candidates = {
        "classic": (lambda t: classic.spam_probability(lr, [t])[0], lr_path),
        "distilbert_pytorch_fp32": (lambda t: transformer.spam_probability(bert, tok, [t])[0], bert_path),
    }

    runs = {name: [] for name in candidates}
    pooled = {name: [] for name in candidates}
    idle_checks, busy = [], False
    for r in range(args.repeats):
        load = system_cpu_percent()
        idle_checks.append(round(load, 1))
        if load > MAX_IDLE_CPU_PERCENT:
            msg = f"run {r + 1}: system CPU load {load:.0f} % > {MAX_IDLE_CPU_PERCENT:.0f} %"
            if not args.allow_busy:
                print(f"ECHEC {msg}: close the other programs and start again", file=sys.stderr)
                return 1
            print(f"warning: {msg}, measuring anyway (--allow-busy)", file=sys.stderr)
            busy = True
        order = list(candidates) if r % 2 == 0 else list(reversed(candidates))
        for name in order:
            ns = time_single_calls(candidates[name][0], texts)
            pooled[name].append(ns)
            stats = latency_stats(ns)
            runs[name].append(rounded(stats))
            print(f"run {r + 1}  {name:26s} p50 {stats['p50_ms']:8.3f} ms  p95 {stats['p95_ms']:8.3f} ms  "
                  f"p99 {stats['p99_ms']:8.3f} ms  (CPU load before run {load:.0f} %)", flush=True)

    report = {
        "machine": machine_info(torch_threads=torch.get_num_threads()),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {"batch_size": 1, "warmup_calls": WARMUP_CALLS, "timed_calls_per_run": TIMED_CALLS,
                     "repeats": args.repeats, "texts": "test split, shuffled with seed 0, cycled",
                     "max_idle_cpu_percent": MAX_IDLE_CPU_PERCENT},
        "cpu_load_before_each_run_percent": idle_checks,
        "busy_machine": busy,
        "versions": {"torch": torch.__version__},
        "models": {},
    }
    for name, (_, path) in candidates.items():
        report["models"][name] = {
            "size_bytes": size_on_disk(path),
            "pooled": rounded(latency_stats(np.concatenate(pooled[name]))),
            "across_runs": {k: rounded(v) for k, v in summarize_runs(runs[name]).items()},
            "runs": runs[name],
        }

    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"latency_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
