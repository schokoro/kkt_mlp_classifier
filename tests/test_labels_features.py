import joblib
import numpy as np
import pandas as pd

from kkt_clf.config import BranchConfig, FeatureConfig
from kkt_clf.features import (
    FeatureBuilder,
    empty_rows,
    make_char_vectorizer,
    make_word_vectorizer,
    preprocess,
    strip_whitespace,
    whitespace_tokenize,
)
from kkt_clf.labels import LabelMaps


def test_label_maps_sorted_and_parents(fx):
    df = pd.read_parquet(fx["ods_small"])
    m = LabelMaps.from_df(df)
    assert m.l1 == sorted(m.l1) and m.l2 == sorted(m.l2) and m.l3 == sorted(m.l3)
    for r in df.drop_duplicates("L3").itertuples():
        i3 = m.l3.index(r.L3)
        assert m.l2[m.parent_l2_of_l3[i3]] == r.L2
        assert m.l1[m.parent_l1_of_l2[m.parent_l2_of_l3[i3]]] == r.L1
        assert m.l1[m.l1_of_l3(np.array([i3]))[0]] == r.L1
    assert (m.encode("L3", ["нет такого", m.l3[0], None]) == [-1, 0, -1]).all()
    assert list(m.decode("L3", np.array([0, -1]))) == [m.l3[0], None]
    m2 = LabelMaps.from_dict(m.to_dict())
    assert m2.l3 == m.l3 and (m2.parent_l2_of_l3 == m.parent_l2_of_l3).all()
    m3 = LabelMaps.from_df(df.sample(frac=1, random_state=5))
    assert m3.to_dict() == m.to_dict()


def test_preprocess_hook():
    assert preprocess([None, float("nan"), "АбВ", "уже"]) == ["", "", "абв", "уже"]
    x = ["молоко 3.2% 1л"]
    assert preprocess(preprocess(x)) == preprocess(x)


def test_char_branch_strips_all_whitespace():
    s = "мо ло\tко 1 л\nx"
    assert strip_whitespace(s) == "молоко1лx"
    v = make_char_vectorizer(BranchConfig((2, 3), 1, None, True)).fit(["ab cd"])
    vocab = set(v.vocabulary_)
    assert "bc" in vocab and "bcd" in vocab
    assert not any(ch.isspace() for g in vocab for ch in g)
    assert v.lowercase is False


def test_word_branch_whitespace_tokens():
    assert whitespace_tokenize("кефир 2.5% , 1 л x") == ["кефир", "2.5%", ",", "1", "л", "x"]
    v = make_word_vectorizer(BranchConfig((1, 1), 1, None, True)).fit(["кефир 2.5% , 1 л", "Big"])
    assert {"2.5%", ",", "1", "л", "Big"} <= set(v.vocabulary_)
    assert v.token_pattern is None and v.lowercase is False


def test_feature_builder(fx, tmp_path):
    df = pd.read_parquet(fx["ods_small"])
    cfg = FeatureConfig(BranchConfig((2, 4), 2, 3000, True), BranchConfig((1, 1), 1, 1000, True), 500, 7)
    fb = FeatureBuilder(cfg).fit(df["name"].tolist(), seed=0)
    assert fb.n_char_features <= 3000 and fb.n_word_features <= 1000
    names = df["name"].tolist()[:50] + ["", None, "ꙮꙮ"]
    X = fb.transform(names)
    assert X.dtype == np.float32 and X.indices.dtype == np.int32
    assert X.shape == (53, fb.n_features)
    assert empty_rows(X)[-3:].all() and not empty_rows(X)[:50].any()
    X_one = fb.transform(names, chunk_rows=1000)
    assert (X != X_one).nnz == 0
    p = tmp_path / "fb.joblib"
    joblib.dump(fb, p)
    fb2 = joblib.load(p)
    assert (fb2.transform(names) != X).nnz == 0
