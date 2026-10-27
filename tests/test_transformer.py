"""Tests for the DistilBERT helpers, on a tiny randomly initialised model built
locally (one layer, 16-dim), so nothing is downloaded and the test takes a second.
What is checked is the plumbing, not the accuracy: padding, masking, batching."""

import numpy as np
import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

from spamedge.transformer import MAX_LENGTH, count_truncated, spam_probability

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
         "win", "cash", "now", "hello", "see", "you", "call", "."]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    vocab_file = tmp_path_factory.mktemp("tok") / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab_file))
    torch.manual_seed(0)
    config = DistilBertConfig(vocab_size=len(VOCAB), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
                              max_position_embeddings=MAX_LENGTH, num_labels=2)
    model = DistilBertForSequenceClassification(config).eval()
    return tokenizer, model


TEXTS = ["win cash now", "hello", "see you . call you now . see you", "call"]


def test_probabilities_have_one_value_per_text_in_0_1(tiny):
    tokenizer, model = tiny
    p = spam_probability(model, tokenizer, TEXTS)
    assert p.shape == (len(TEXTS),), f"ECHEC : forme obtenue {p.shape}, attendue {(len(TEXTS),)}"
    assert np.all((p > 0) & (p < 1)), f"ECHEC : probabilités hors de ]0, 1[ : {p}"


def test_padding_does_not_change_the_prediction(tiny):
    # Batched together, the short texts are padded up to the longest one. If the
    # attention mask were ignored, the padding would leak into their scores.
    tokenizer, model = tiny
    one_by_one = spam_probability(model, tokenizer, TEXTS, batch_size=1)
    all_at_once = spam_probability(model, tokenizer, TEXTS, batch_size=len(TEXTS))
    gap = float(np.max(np.abs(one_by_one - all_at_once)))
    assert gap < 1e-5, f"ECHEC : écart maximal {gap:.2e} entre batch 1 et batch 4, attendu < 1e-5"


def test_count_truncated(tiny):
    tokenizer, _ = tiny
    # "win " * 200 -> 200 tokens + [CLS] + [SEP] = 202 > 128; "hello" -> 3 tokens.
    expected = 1
    got = count_truncated(tokenizer, ["win " * 200, "hello"])
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"
