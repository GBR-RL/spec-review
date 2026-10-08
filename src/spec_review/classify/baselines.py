"""Classical classifiers: TF-IDF with logistic regression or a linear SVM."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

Predict = Callable[[Sequence[str]], Sequence[str]]


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")


def tfidf_lr(texts: Sequence[str], labels: Sequence[str]) -> Predict:
    model = make_pipeline(
        _vectorizer(), LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")
    )
    model.fit(list(texts), list(labels))
    return lambda xs: [str(p) for p in model.predict(list(xs))]


def tfidf_svm(texts: Sequence[str], labels: Sequence[str]) -> Predict:
    model = make_pipeline(_vectorizer(), LinearSVC(C=1.0, class_weight="balanced"))
    model.fit(list(texts), list(labels))
    return lambda xs: [str(p) for p in model.predict(list(xs))]


METHODS = {"tfidf_lr": tfidf_lr, "tfidf_svm": tfidf_svm}
