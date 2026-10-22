"""Предобработка и TF-IDF признаки (§4.4).

Две ветки: char n-gram по строке без пробельных символов и word unigram по
``str.split()``. Все функции, которые попадают в векторизаторы, определены на
уровне модуля, чтобы пайплайн сериализовался через joblib.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import BranchConfig, FeatureConfig

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


def preprocess(texts: Iterable) -> list[str]:
    """Хук предобработки: null → "", затем ``lower()``. На чистом входе идемпотентен."""
    out = []
    for t in texts:
        if t is None or (isinstance(t, float) and math.isnan(t)):
            out.append("")
        else:
            out.append(str(t).lower())
    return out


def strip_whitespace(text: str) -> str:
    """Удаляет все пробельные символы (Unicode ``\\s``) — вход char-ветки."""
    return _WS_RE.sub("", text)


def whitespace_tokenize(text: str) -> list[str]:
    """Токенизация word-ветки: только по пробельным символам."""
    return text.split()


def _identity(text: str) -> str:
    return text


def make_char_vectorizer(cfg: BranchConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        preprocessor=strip_whitespace,
        lowercase=False,
        ngram_range=tuple(cfg.ngram_range),
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf,
        dtype=np.float32,
    )


def make_word_vectorizer(cfg: BranchConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        preprocessor=_identity,
        tokenizer=whitespace_tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=tuple(cfg.ngram_range),
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf,
        dtype=np.float32,
    )


def _to_csr32(X: sp.spmatrix) -> sp.csr_matrix:
    X = sp.csr_matrix(X, dtype=np.float32)
    X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr[-1] < np.iinfo(np.int32).max:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X


def vstack_csr(blocks: Sequence[sp.csr_matrix], n_cols: int) -> sp.csr_matrix:
    """Склейка CSR-блоков без промежуточного COO (пик памяти ≈ 2× результата)."""
    if not blocks:
        return sp.csr_matrix((0, n_cols), dtype=np.float32)
    nnz = sum(b.nnz for b in blocks)
    idx_dtype = np.int32 if nnz < np.iinfo(np.int32).max else np.int64
    indptr = [np.zeros(1, dtype=idx_dtype)]
    offset = 0
    for b in blocks:
        indptr.append((b.indptr[1:].astype(idx_dtype) + offset))
        offset += b.nnz
    X = sp.csr_matrix(
        (
            np.concatenate([b.data for b in blocks]),
            np.concatenate([b.indices for b in blocks]),
            np.concatenate(indptr),
        ),
        shape=(sum(b.shape[0] for b in blocks), n_cols),
    )
    return X


class FeatureBuilder:
    """Предобработка + две TF-IDF ветки → разреженная матрица float32.

    Хранит ссылку на функцию предобработки: при загрузке артефакта инференс
    использует ровно её.
    """

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.preprocess = preprocess
        self.char = make_char_vectorizer(cfg.char)
        self.word = make_word_vectorizer(cfg.word)
        self.fitted = False

    def fit(self, texts: Sequence[str], seed: int = 0) -> "FeatureBuilder":
        texts = list(texts)
        n = self.cfg.vectorizer_fit_rows
        if n is not None and len(texts) > n:
            rng = np.random.default_rng(seed)
            sel = np.sort(rng.choice(len(texts), size=n, replace=False))
            texts = [texts[i] for i in sel]
            log.info("Векторизаторы обучаются на подвыборке train: %d строк", n)
        pre = self.preprocess(texts)
        self.char.fit(pre)
        self.word.fit(pre)
        self.fitted = True
        log.info(
            "Словарь: char=%d, word=%d, всего=%d",
            self.n_char_features, self.n_word_features, self.n_features,
        )
        return self

    @property
    def n_char_features(self) -> int:
        return len(self.char.vocabulary_)

    @property
    def n_word_features(self) -> int:
        return len(self.word.vocabulary_)

    @property
    def n_features(self) -> int:
        return self.n_char_features + self.n_word_features

    def _transform_block(self, texts: Sequence) -> sp.csr_matrix:
        pre = self.preprocess(texts)
        Xc = self.char.transform(pre)
        Xw = self.word.transform(pre)
        return _to_csr32(sp.hstack([Xc, Xw], format="csr", dtype=np.float32))

    def transform(self, texts: Sequence, chunk_rows: int | None = None) -> sp.csr_matrix:
        """Векторизация чанками; результат — CSR float32 с int32-индексами."""
        if not self.fitted:
            raise RuntimeError("FeatureBuilder не обучен")
        chunk = chunk_rows or self.cfg.transform_chunk_rows
        texts = list(texts) if not isinstance(texts, list) else texts
        blocks = [self._transform_block(texts[i : i + chunk]) for i in range(0, len(texts), chunk)]
        return vstack_csr(blocks, self.n_features)


def empty_rows(X: sp.csr_matrix) -> np.ndarray:
    """Маска строк с нулевым вектором признаков."""
    return np.diff(X.indptr) == 0


def matrix_stats(X: sp.csr_matrix) -> dict:
    nnz = np.diff(X.indptr)
    return {
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "nnz": int(X.nnz),
        "nnz_per_row_mean": float(nnz.mean()) if len(nnz) else 0.0,
        "nnz_per_row_max": int(nnz.max()) if len(nnz) else 0,
        "bytes": int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes),
        "empty_rows": int((nnz == 0).sum()),
    }
