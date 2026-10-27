"""DistilBERT for spam classification: loading, batching, inference.

The training loop lives in scripts/train_transformer.py. It is a plain PyTorch
loop rather than the Hugging Face Trainer: about forty lines I can read and
explain end to end, and no extra dependency.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
# Pinned revision of the Hub repository (Apache-2.0), so a later upload cannot change the results.
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
# On the training split, the 99th percentile of the token count is 76 and the
# maximum 238 (measured with this tokenizer). 128 keeps all but a handful of
# messages whole, and longer ones still keep their first 128 tokens.
MAX_LENGTH = 128
INFERENCE_BATCH_SIZE = 64


def load_pretrained(name: str = MODEL_NAME, revision: str | None = MODEL_REVISION):
    """Tokenizer + DistilBERT with a fresh 2-class head, ready for fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision, num_labels=2)
    return tokenizer, model


def load_finetuned(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def encode(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    """Tokenise one batch, padded to its own longest message (not to MAX_LENGTH).

    SMS are short (median 19 tokens), so padding every batch to 128 would make
    most of the computation run on padding.
    """
    return tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_LENGTH,
                     return_tensors="pt")


@torch.inference_mode()
def spam_probability(model, tokenizer, texts, batch_size: int = INFERENCE_BATCH_SIZE) -> np.ndarray:
    """Probability of the spam class for each text, softmax over the two logits."""
    model.eval()
    texts = list(texts)
    out = np.empty(len(texts))
    for start in range(0, len(texts), batch_size):
        batch = encode(tokenizer, texts[start:start + batch_size])
        logits = model(**batch).logits
        out[start:start + batch_size] = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return out


def count_truncated(tokenizer, texts) -> int:
    """How many texts are longer than MAX_LENGTH tokens (special tokens included)."""
    lengths = [len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]]
    return sum(n > MAX_LENGTH for n in lengths)
