"""The classic baseline: TF-IDF features + logistic regression.

Two TF-IDF blocks are concatenated:
  - words and word pairs, which capture "free entry", "call now", "txt STOP";
  - character 3- to 5-grams inside word boundaries, which still match when a
    spammer writes "fr33" or splits a word, and cope with SMS spelling ("2nite").
I want a fair baseline, not a weak one, so it gets both.

sublinear_tf replaces a raw count c by 1 + log(c): a word repeated five times
should not weigh five times more than a word seen once.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# Grid searched on the validation set only. The first grid stopped at C = 100 and the best
# value landed on that edge, so 1000 was added. I also tried 1e4 and 1e5 by hand: validation
# F1 0.9738 for both vs 0.9735 at 1000, i.e. one spam caught and one extra false alarm.
# The curve is flat past 1000, so the grid stops there.
C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
CLASS_WEIGHT_GRID = (None, "balanced")
MAX_ITER = 5000


def build_classic(c: float = 10.0, class_weight: str | None = None) -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                  lowercase=True, min_df=2)),
    ])
    clf = LogisticRegression(C=c, class_weight=class_weight, max_iter=MAX_ITER)
    return Pipeline([("tfidf", features), ("logreg", clf)])


def spam_probability(model: Pipeline, texts) -> np.ndarray:
    """Probability of the spam class (label 1) for each text."""
    classes = list(model.classes_)
    return model.predict_proba(list(texts))[:, classes.index(1)]
