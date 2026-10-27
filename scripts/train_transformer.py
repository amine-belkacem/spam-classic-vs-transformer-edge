"""Fine-tune DistilBERT on the SMS training split, on CPU.

Usage:
    python scripts/train_transformer.py --seed 0

The model is scored on validation after every epoch and the best epoch (spam F1,
threshold 0.5) is saved to models/distilbert-seed<seed>/. The test set is not
read here. Losses, per-epoch validation scores and wall-clock times go to
results/distilbert_training_seed<seed>.json.

Hyper-parameters are the usual ones for BERT-style fine-tuning (learning rate
5e-5, 10 % linear warm-up then linear decay, weight decay 0.01). I did not tune
them: the question is how far an off-the-shelf fine-tune goes, and every tuning
run on CPU costs tens of minutes.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spamedge.metrics import spam_metrics  # noqa: E402
from spamedge.transformer import (  # noqa: E402
    MAX_LENGTH,
    MODEL_NAME,
    MODEL_REVISION,
    count_truncated,
    encode,
    load_pretrained,
    spam_probability,
)

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
LOG_EVERY_STEPS = 50


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    seed_everything(args.seed)
    train = pd.read_csv(args.data / "sms_train.csv", keep_default_na=False)
    val = pd.read_csv(args.data / "sms_val.csv", keep_default_na=False)
    tokenizer, model = load_pretrained()
    print(f"truncated at {MAX_LENGTH} tokens: train {count_truncated(tokenizer, train['text'])}, "
          f"val {count_truncated(tokenizer, val['text'])}")

    # Biases and normalisation weights are left out of weight decay. The usual recipe
    # matches the name "LayerNorm.weight", but DistilBERT calls most of them
    # "sa_layer_norm" / "output_layer_norm", so the name test would miss them.
    # They are exactly the 1-D parameters, which is what is tested here.
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.ndim > 1], "weight_decay": WEIGHT_DECAY},
        {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE)
    steps_per_epoch = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRACTION * total_steps), total_steps)

    texts = train["text"].tolist()
    labels = torch.tensor(train["label"].to_numpy(), dtype=torch.long)
    rng = np.random.default_rng(args.seed)
    out_dir = args.models / f"distilbert-seed{args.seed}"
    history, best_f1 = [], -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(texts))
        t0, losses = time.perf_counter(), []
        for step, start in enumerate(range(0, len(order), BATCH_SIZE), start=1):
            idx = order[start:start + BATCH_SIZE]
            batch = encode(tokenizer, [texts[i] for i in idx])
            loss = model(**batch, labels=labels[idx]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if step % LOG_EVERY_STEPS == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} "
                      f"loss {np.mean(losses[-LOG_EVERY_STEPS:]):.4f} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
        train_s = time.perf_counter() - t0

        m = spam_metrics(val["label"].to_numpy(), spam_probability(model, tokenizer, val["text"]))
        history.append({"epoch": epoch, "train_seconds": round(train_s, 1),
                        "mean_train_loss": round(float(np.mean(losses)), 5),
                        **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        print(f"epoch {epoch} done in {train_s:.0f}s  val F1={m['f1']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f}", flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    best = max(history, key=lambda h: h["f1"])  # same rule as above: first epoch reaching the max
    log = {
        "seed": args.seed, "model": MODEL_NAME, "revision": MODEL_REVISION,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "warmup_fraction": WARMUP_FRACTION, "max_length": MAX_LENGTH,
        "best_epoch": best["epoch"], "history": history,
        "machine": {"cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
                    "torch": torch.__version__, "python": platform.python_version()},
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"distilbert_training_seed{args.seed}.json").write_text(
        json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch {best['epoch']} (val F1 {best['f1']}), saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
