"""Draw the figures from the JSON / CSV files in results/.

Usage:
    python scripts/make_figures.py

Nothing is recomputed or smoothed here: the PR curves are the exact step
functions from scikit-learn, and the bars are the numbers in the JSON files.
Colours follow one fixed palette (classic = teal, DistilBERT = violet); every
bar and curve also carries a text label, because teal is pale on white.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

COLORS = {"classic": "#1baf7a", "distilbert": "#6250d6"}
LABELS = {"classic": "TF-IDF + logistic regression", "distilbert": "DistilBERT (fine-tuned)"}
LATENCY_KEYS = {"classic": "classic", "distilbert": "distilbert_pytorch_fp32"}
DPI = 150


def pr_curves(scores: pd.DataFrame, metrics: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    y = scores["label"].to_numpy()
    for name in ("classic", "distilbert"):
        precision, recall, _ = precision_recall_curve(y, scores[name].to_numpy())
        m = metrics["models"][name]
        ax.step(recall, precision, where="post", color=COLORS[name], lw=2,
                label=f"{LABELS[name]}  (PR-AUC {m['pr_auc']:.3f})")
        ax.plot(m["recall"], m["precision"], "o", color=COLORS[name], ms=8, mec="black", mew=0.8)
        ax.annotate(f"threshold 0.5\nP={m['precision']:.3f}  R={m['recall']:.3f}",
                    (m["recall"], m["precision"]), textcoords="offset points",
                    xytext=(-150, -38 if name == "classic" else -80), fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": COLORS[name]})
    ax.set_xlabel("Recall (share of spam caught)")
    ax.set_ylabel("Precision (share of flags that are spam)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.5, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"SMS test set: {metrics['n_test']} messages, {metrics['n_spam']} spam")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def quality_vs_cost(metrics: dict, latency: dict, out: Path) -> None:
    """Three dot plots. No bars: the F1 axis does not start at 0 and the other two
    are logarithmic, and in both cases a bar's length would mean nothing."""
    names = ["classic", "distilbert"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    for i, name in enumerate(names):
        m = metrics["models"][name]
        lo, hi = m["ci"]["f1"]
        ax.errorbar(i, m["f1"], yerr=[[m["f1"] - lo], [hi - m["f1"]]], fmt="o", ms=9,
                    color=COLORS[name], ecolor=COLORS[name], elinewidth=2, capsize=8)
        # Left model labelled on its left, right model on its right, so no label crosses a CI bar.
        side = -1 if i == 0 else 1
        ax.text(i + side * 0.12, m["f1"], f"{m['f1']:.3f}\n95 % CI\n[{lo:.3f}, {hi:.3f}]",
                va="center", ha="right" if side < 0 else "left", fontsize=9)
    ax.set_ylim(0.88, 1.0)
    ax.set_xlim(-1.3, 2.3)
    ax.set_ylabel("Spam F1 on test")
    ax.set_title("Quality (dot = F1, bar = 95 % bootstrap CI)", fontsize=10)

    ax = axes[1]
    for i, name in enumerate(names):
        s = latency["models"][LATENCY_KEYS[name]]["pooled"]  # all timed calls of all runs
        ax.plot([i, i], [s["p50_ms"], s["p95_ms"]], color=COLORS[name], lw=2)
        ax.plot(i, s["p50_ms"], "o", ms=9, color=COLORS[name])
        ax.plot(i, s["p95_ms"], "_", ms=16, mew=2, color=COLORS[name])
        ax.text(i + 0.08, s["p50_ms"], f"p50 {s['p50_ms']:.2f} ms\np95 {s['p95_ms']:.2f} ms",
                va="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 200)
    ax.set_ylabel("Latency per message, batch 1 (ms, log scale)")
    ax.set_title("Latency (dot = p50, tick = p95)", fontsize=10)

    ax = axes[2]
    for i, name in enumerate(names):
        mb = latency["models"][LATENCY_KEYS[name]]["size_bytes"] / 1e6
        ax.plot(i, mb, "o", ms=9, color=COLORS[name])
        ax.text(i + 0.08, mb, f"{mb:.1f} MB", va="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 1000)
    ax.set_ylabel("Size on disk (MB, log scale)")
    ax.set_title("Model + vectorizer / tokenizer files", fontsize=10)

    for ax in axes[1:]:  # the quality panel sets its own range: it has labels on both sides
        ax.set_xlim(-0.4, 2.3)
    for ax in axes:
        ax.set_xticks(range(len(names)), ["TF-IDF + LR", "DistilBERT"])
        ax.grid(axis="y", alpha=0.3)
    cpu = latency["machine"]["cpu"]
    fig.suptitle(f"Measured on {cpu}, PyTorch {latency['machine'].get('torch_threads')} threads", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--tag", default="pc")
    args = parser.parse_args()

    metrics = json.loads((args.results / "test_metrics.json").read_text(encoding="utf-8"))
    latency = json.loads((args.results / f"latency_{args.tag}.json").read_text(encoding="utf-8"))
    if latency.get("busy_machine"):
        raise ValueError("these latencies were measured on a busy machine; measure again before plotting")
    scores = pd.read_csv(args.results / "test_scores.csv")
    if not np.isin(scores["label"], [0, 1]).all():
        raise ValueError("test_scores.csv: labels outside 0/1")

    args.figures.mkdir(parents=True, exist_ok=True)
    pr_curves(scores, metrics, args.figures / "pr_curves_test.png")
    quality_vs_cost(metrics, latency, args.figures / f"quality_vs_cost_{args.tag}.png")
    print(f"figures written to {args.figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
