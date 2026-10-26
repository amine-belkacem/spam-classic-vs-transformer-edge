# Specification

I wrote this before any model code, so the targets below are targets, not results.
Where a target turns out to be wrong, the README will say so instead of quietly
moving the goalposts.

## 1. Question

When is TF-IDF + logistic regression good enough to filter spam, and when is
DistilBERT worth what it costs? I compare quality (precision, recall, F1),
cost (latency, throughput, size, memory) and robustness, on a PC and on a
Raspberry Pi 5. Then I try a cascade that only calls DistilBERT on the messages
the cheap model is unsure about.

## 2. Measurable requirements

| ID | Requirement | How it is measured |
|---|---|---|
| R1 | Spam-class precision, recall, F1, plus PR-AUC, on a test set never used for tuning | 95 % bootstrap confidence interval, 1,000 resamples |
| R2 | DistilBERT run-to-run variance | fine-tuned with 5 seeds, mean ± std |
| R3 | Per-message latency, batch size 1 (the real use case), PC and Pi 5 | 50 warm-up calls, then 1,000 timed calls: p50 / p95 / p99 |
| R4 | Throughput at batch size 32 | 10 repetitions, mean ± std |
| R5 | On-disk size of everything needed to predict (model + vectorizer or tokenizer) | bytes, measured by a script |
| R6 | Peak memory during inference | max RSS, measured in a separate process |
| R7 | int8 ONNX vs FP32 | prediction agreement (%) and F1 difference |
| R8 | Cascade | share of messages sent to DistilBERT vs F1, with mean latency |
| R9 | Robustness to obfuscation | recall vs obfuscation strength (0 to 50 %), 5 draws per level |
| R10 | Domain shift | train on SMS, test on YouTube comments, same metrics as R1 |
| R11 | Reproducibility | one command rebuilds every number and figure; seeds and versions pinned |
| R12 | Pi measurement conditions | temperature, throttling flags and thread count logged with every result |

## 3. Architecture

```text
                 ┌─────────────────────────────┐
  data/raw  ───► │ prepare_data.py              │  de-duplicate → stratified split
 (UCI SMS,       │ (fixed seed)                 │  70 / 15 / 15 (train/val/test)
  YouTube)       └──────────────┬──────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
   ┌─────────────────────┐            ┌──────────────────────────┐
   │ Classic              │            │ Transformer               │
   │ TF-IDF words 1-2 +   │            │ fine-tuned DistilBERT     │
   │ chars 3-5            │            │ → ONNX FP32               │
   │ → logistic regr.     │            │ → dynamic int8            │
   └──────────┬──────────┘            └────────────┬─────────────┘
              └───────────────┬─────────────────────┘
                              ▼
            ┌───────────────────────────────────┐
            │ Cascade: p = LR(x)                 │
            │ p < t_low  → ham                   │
            │ p > t_high → spam                  │
            │ otherwise  → DistilBERT(x)         │
            └─────────────────┬─────────────────┘
                              ▼
   benchmarks: quality · latency · throughput · size · memory · robustness
   run on the PC (x86) and on the Raspberry Pi 5 (ARM)  →  results/*.json → figures/
```

## 4. Design choices

| Choice | Why | Rejected |
|---|---|---|
| Logistic regression | Gives probabilities the cascade can threshold; fast; the weights can be read | Naive Bayes (poorly calibrated), linear SVM (no native probabilities) |
| TF-IDF on words + character n-grams | Character n-grams survive `fr33` or `W.I.N`; I want a fair baseline, not a straw man | Words only, which would make the baseline weaker on purpose |
| `distilbert-base-uncased` | 66 M parameters, about 40 % fewer than BERT-base; the usual reference for this kind of comparison | BERT-base (too heavy for the Pi); MiniLM / TinyBERT kept as a possible extension |
| ONNX Runtime on the Pi | Runs on ARM64, supports dynamic int8, loads faster than PyTorch | PyTorch on the Pi is still measured as a reference; TFLite would need a TensorFlow detour |
| Dynamic int8 quantization | No calibration set, suits the linear layers of a Transformer | Static quantization: more work for an uncertain gain here |
| Thresholds tuned on validation only | Tuning on test inflates every number | — |
| De-duplication before splitting | The SMS file has 403 duplicated messages (5,574 lines, 5,171 unique texts). Without it, the same message can sit in train and test | — |
| Python 3.12, pyproject, pytest, pyflakes, GitHub Actions | Standard, reproducible; the CI stays on CPU and small inputs | Notebooks only (not testable) |

## 5. Data

| Dataset | Size | License | Source |
|---|---|---|---|
| SMS Spam Collection | 5,574 SMS, 747 spam | CC BY 4.0 | Almeida, T. & Hidalgo, J. (2011), UCI, https://doi.org/10.24432/C5CC84 |
| YouTube Spam Collection | 1,956 comments, 1,005 spam | CC BY 4.0 | Alberto, T. & Lochter, J. (2015), UCI, https://doi.org/10.24432/C58885 |

The data is not stored in the repository. `scripts/download_data.py` fetches it
and checks the SHA-256 of every file and the counts above. The YouTube CSVs
contain comment authors' names; the loader never reads that column.

## 6. Test plan

- Unit tests with expected values worked out by hand in the comments
  (metrics on a small confusion matrix, de-duplication, split disjointness and
  stratification, cascade routing, obfuscation determinism).
- Regression tests: ONNX FP32 logits equal to PyTorch within 1e-4; int8
  predictions agree with FP32 above a fixed threshold.
- An end-to-end run on 200 messages in the CI.
- Every assertion prints `ECHEC : obtenu X, attendu Y`, and I check that a
  deliberately wrong expected value makes the test fail.

## 7. Success criteria

| Criterion | Target |
|---|---|
| Every number rebuilt by one command, CI green | required |
| TF-IDF baseline spam precision on test | ≥ 0.95 (plausible from the literature on this dataset, to be checked) |
| PC **and** Pi 5 measurements for all four model variants | required |
| int8 vs FP32 F1 loss | ≤ 1 point, otherwise documented |
| Cascade | hypothesis: recover ≥ 90 % of the LR → DistilBERT F1 gap while sending ≤ 20 % of messages to DistilBERT. Published either way. |
| Robustness curves and domain-shift matrix | generated and discussed |
| Limits listed in the README | required |
