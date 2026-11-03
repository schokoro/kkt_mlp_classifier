import copy

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from kkt_clf import Predictor, run_training, verify_roundtrip
from kkt_clf.config import build_feature_config
from kkt_clf.features import FeatureBuilder
from kkt_clf import selection as sel
from kkt_clf.vectorizer import (VectorizerArtifactError, fit_vectorizer,
    load_vectorizer_artifact, refit_selected_vectorizer)
from tests.test_s01_vectorizer import make_cfg, assert_features_equal


def test_global_known_chi2_ties_and_branch_budgets():
    X = sp.csr_matrix([[2., 0, 1, 2, 0], [0, 2, 1, 0, 2]])
    result = sel.select_features(X, np.array([0, 1]), sel.SelectionConfig(k=4, char_share=.5), 3)
    np.testing.assert_array_equal(result.char.indices, [0, 1])
    np.testing.assert_allclose(result.char.scores, [2, 2])
    np.testing.assert_array_equal(result.word.indices, [3, 4])
    assert sel.branch_budgets(10, .8, 2, 20) == (2, 2)


def test_ovr_positive_sign_round_robin_imbalance_and_fallback():
    X = sp.csr_matrix([[3., 0, 1, 0], [3, 0, 1, 0], [3, 0, 1, 0], [0, 3, 1, 0]])
    y = np.array([0, 0, 0, 1])
    sums = np.asarray(X.sum(axis=0)).ravel()
    _, observed, expected = sel._class_chi2(X[y == 1], sums, sums.sum())
    assert observed[1] > expected[1] and observed[0] < expected[0]
    result, report = sel._select_branch_class_aware(X, y, 4)
    assert dict(zip(result.indices, result.origin))[0] == 'class:0'
    assert dict(zip(result.indices, result.origin))[1] == 'class:1'
    assert len(set(result.indices)) == 4
    assert report['n_fallback'] >= 1
    again, report2 = sel._select_branch_class_aware(X, y, 4)
    np.testing.assert_array_equal(result.indices, again.indices)
    assert result.origin == again.origin and report == report2


def test_overlapping_candidates_round_robin_and_no_positive(monkeypatch):
    # Controlled scores force overlapping candidates and a class without positive terms.
    values = iter([(np.array([9., 8, 1, 0]), np.array([2., 2, 2, 0]), np.ones(4)),
                   (np.array([9., 7, 6, 0]), np.array([2., 0, 2, 0]), np.ones(4)),
                   (np.ones(4), np.zeros(4), np.ones(4))])
    monkeypatch.setattr(sel, '_class_chi2', lambda *args: next(values))
    result, report = sel._select_branch_class_aware(sp.csr_matrix(np.ones((3, 4))), np.arange(3), 4)
    assert dict(zip(result.indices, result.origin)) == {0: 'class:0', 1: 'class:0', 2: 'class:1', 3: 'global_fallback'}
    assert report['n_positive_per_class'] == [3, 2, 0]
    assert report['n_fallback'] == 1


def test_resource_stress_no_dense_class_by_feature_or_full_class_rankings(tmp_path):
    # Run in a fresh process so the peak is measured relative to the imported
    # libraries and fixture, not to unrelated allocations in the pytest process.
    import json
    import subprocess
    import sys

    script = r"""
import json
import os
import threading
import numpy as np
import scipy.sparse as sp
from kkt_clf import selection as sel

classes, features, k = 128, 100_000, 90_000
X = sp.csr_matrix((np.ones(classes), (np.arange(classes), np.arange(classes))),
                  shape=(classes, features))
y = np.arange(classes)
# Worst case: every class has a positive candidate for every feature.
# The sparse input and one-feature-vector-at-a-time scores avoid themselves
# allocating a dense C x F fixture while exercising full ranking storage.
def scores(*args):
    return (np.arange(features, dtype=np.float64),
            np.full(features, 2.0), np.ones(features))
sel._class_chi2 = scores

def rss():
    with open('/proc/self/statm', encoding='ascii') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')

baseline = rss()
peak = [baseline]
stop = threading.Event()
def sample():
    while not stop.wait(0.001):
        peak[0] = max(peak[0], rss())
thread = threading.Thread(target=sample)
thread.start()
try:
    result, report = sel._select_branch_class_aware(X, y, k, ratings_dir=RATINGS_DIR)
    peak[0] = max(peak[0], rss())
finally:
    stop.set()
    thread.join()
print(json.dumps({'delta': peak[0] - baseline, 'selected': len(result.indices),
                  'positive': report['n_positive_per_class']}))
""".replace('RATINGS_DIR', repr(str(tmp_path)))
    proc = subprocess.run([sys.executable, '-c', script], check=True,
                          capture_output=True, text=True)
    measurement = json.loads(proc.stdout.strip().splitlines()[-1])
    assert measurement['selected'] == 90_000
    assert measurement['positive'] == [100_000] * 128
    # Even a float32 C x F matrix exceeds this budget; C x K arrays of
    # indices and scores are larger still. Includes output and working memory.
    limit = 128 * 100_000 * np.dtype(np.float32).itemsize
    assert measurement['delta'] < limit, measurement


@pytest.fixture
def fitted(tmp_path):
    path = tmp_path / 'corpus.parquet'
    names = ['alpha beta gamma', 'alpha beta', 'beta delta', 'gamma delta',
             'alpha delta', 'gamma beta', 'alpha gamma', 'delta beta', 'alpha beta']
    pd.DataFrame({'name': names, 'L3': ['unused'] * len(names)}).to_parquet(path, index=False)
    cfg = build_feature_config('ods_light', {'vectorizer_fit_rows': 6, 'char': {'min_df': 1}, 'word': {'min_df': 1}})
    builder, meta = fit_vectorizer(cfg, str(path), max_rows=9, seed=7)
    char = sorted(builder.char.vocabulary_.values())[:3]
    word = sorted(builder.word.vocabulary_.values())[:2]
    selection = {'method': 'global', 'k_requested': 5, 'char_share': .6, 'seed': 7,
                 'char': {'indices': char}, 'word': {'indices': [builder.n_char_features + i for i in word]}}
    return path, names, builder, meta, selection


def test_refit_subsample_idf_normalization_and_save_load(fitted, tmp_path, monkeypatch):
    path, names, full, meta, selection = fitted
    old_vocab = copy.deepcopy(full.char.vocabulary_)
    seen = []
    from sklearn.feature_extraction.text import TfidfVectorizer
    original_fit = TfidfVectorizer.fit
    def record(self, docs, *args, **kwargs):
        docs = list(docs)
        seen.append(docs)
        return original_fit(self, docs, *args, **kwargs)
    monkeypatch.setattr(TfidfVectorizer, 'fit', record)
    out = tmp_path / 'selected'
    result = refit_selected_vectorizer(str(path), full, meta, selection, out_dir=str(out))
    chosen = np.sort(np.random.default_rng(7).choice(len(names), size=6, replace=False))
    assert seen == [full.preprocess([names[i] for i in chosen])] * 2
    assert full.char.vocabulary_ == old_vocab
    for branch, offset in [('char', 0), ('word', full.n_char_features)]:
        old = getattr(full, branch)
        new = getattr(result.builder, branch)
        idx = np.array(selection[branch]['indices']) - offset
        np.testing.assert_allclose(new.idf_, old.idf_[idx])
        x = new.transform(result.builder.preprocess(names))
        norm = np.sqrt(np.asarray(x.power(2).sum(axis=1)).ravel())
        np.testing.assert_allclose(norm[norm > 0], 1, atol=1e-6)
    loaded = load_vectorizer_artifact(str(out)).builder
    assert_features_equal(result.builder, loaded, names)
    # On a document containing selected and discarded terms, slicing loses L2 norm.
    old = full.transform(names)
    selected_idx = selection['char']['indices'] + selection['word']['indices']
    sliced = old[:, selected_idx]
    new = result.builder.transform(names)
    assert np.max(np.abs((new - sliced).data)) > 1e-5


def test_refit_rejects_changed_corpus(fitted):
    path, names, full, meta, selection = fitted
    pd.DataFrame({'name': names + ['changed']}).to_parquet(path, index=False)
    with pytest.raises(VectorizerArtifactError, match='корпус|Корпус'):
        refit_selected_vectorizer(str(path), full, meta, selection)


def test_selected_pretrained_train_save_load_inference(fitted, tmp_path, monkeypatch):
    path, names, full, meta, selection = fitted
    out = tmp_path / 'selected'
    selected = refit_selected_vectorizer(str(path), full, meta, selection, out_dir=str(out)).builder
    def forbidden(*args, **kwargs):
        pytest.fail('pretrained mode must not fit')
    monkeypatch.setattr(FeatureBuilder, 'fit', forbidden)
    cfg = make_cfg(tmp_path, 'pretrained', str(out))
    for i in range(2):
        cfg.output_dir = str(tmp_path / f'run_selected_{i}')
        run = run_training(cfg)
        predictor = Predictor.load(run.artifact_dir, 'cpu')
        assert_features_equal(selected, predictor.features, names + ['', None])
        assert verify_roundtrip(run)


def test_selection_rejects_unlabeled_train():
    X = sp.csr_matrix([[1., 1.], [1., 1.]])
    with pytest.raises(sel.SelectionError, match='train'):
        sel.select_features(X, np.array([0, -1]), sel.SelectionConfig(k=2, char_share=.5), 1)


def test_selection_artifact_roundtrip(fitted, tmp_path):
    _, names, full, _, _ = fitted
    X = full.transform(names)
    y = np.arange(len(names)) % 2
    result = sel.select_features(X, y, sel.SelectionConfig(method='class_aware', k=8, char_share=.5), full.n_char_features)
    path = tmp_path / 'selection'
    provenance = {'vectorizer': 'full-v1', 'labeled': 'train-v1'}
    sel.save_selection_artifact(str(path), result, provenance)
    loaded, metadata = sel.load_selection_artifact(str(path))
    assert metadata['provenance'] == provenance
    for branch in ('char', 'word'):
        before, after = getattr(result, branch), getattr(loaded, branch)
        np.testing.assert_array_equal(before.indices, after.indices)
        np.testing.assert_allclose(before.scores, after.scores)
        assert before.origin == after.origin
