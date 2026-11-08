"""Tiny nine-run S08 mechanics smoke, using independent generated sources only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kkt_clf import s08
from kkt_clf.artifact import load_artifact
from kkt_clf.inference import predict_frame
from kkt_clf.train import predict
from kkt_clf.vectorizer import load_vectorizer_artifact
from tests.test_s08_integrity_independent import synthetic, guard_closed_reads
from tests.test_s08_fixtures import sources, file_sha, write_json, reseal


@pytest.fixture
def executed(synthetic):
    cfg, root, src = synthetic
    plan = s08.write_plan(cfg, root)
    summary = s08.execute(root)
    assert len(summary['success']) == 9, summary
    assert summary['failed'] == summary['skipped'] == summary['pending'] == []
    return cfg, root, src, plan


def test_nine_run_smoke_exact_dev_only_rank_refit_train_save_load_inference(synthetic, monkeypatch):
    cfg, root, src = synthetic
    guard_closed_reads(monkeypatch, src['parts']['sealed_test']['file'])
    before_full = {str(p): file_sha(p) for p in src['full'].rglob('*') if p.is_file()}
    plan = s08.write_plan(cfg, root)
    assert plan['training']['train']['epochs'] == 14
    assert plan['training']['restore_best'] is False
    assert plan['training']['training_rows'] == 'full_development_union'
    original_train = s08.train_model
    original_selection = s08.select_features
    original_refit = s08.refit_selected_vectorizer
    full = load_vectorizer_artifact(cfg.full_tfidf)
    expected_X = full.builder.transform(src['pool'].sort_values('row_id').name.tolist())
    expected_y3 = np.array([0] * 10 + [1] * 10)
    calls, rankings, refits = [], [], []
    def training(model, X, y, Xm, ym, loss, train_cfg, weights, device, seed, **kwargs):
        assert X.shape[0] == Xm.shape[0] == len(src['pool']) == 20
        np.testing.assert_array_equal(y[2], expected_y3)
        np.testing.assert_array_equal(ym[2], expected_y3)
        assert train_cfg.epochs == 14 and kwargs['restore_best'] is False
        assert (X != Xm).nnz == 0
        calls.append((seed, X.shape[1]))
        return original_train(model, X, y, Xm, ym, loss, train_cfg, weights, device, seed, **kwargs)
    def selection(X, y, *args, **kwargs):
        assert X.shape == expected_X.shape and (X != expected_X).nnz == 0
        np.testing.assert_array_equal(y, expected_y3)
        rankings.append(True)
        return original_selection(X, y, *args, **kwargs)
    def refit(corpus, builder, meta, selection, **kwargs):
        assert Path(corpus).resolve() == Path(cfg.corpus).resolve()
        assert meta['source'] == full.meta['source'] and meta['fit'] == full.meta['fit']
        for branch in ('char', 'word'):
            assert getattr(builder, branch).get_feature_names_out().tolist() == getattr(full.builder, branch).get_feature_names_out().tolist()
        refits.append(True)
        return original_refit(corpus, builder, meta, selection, **kwargs)
    monkeypatch.setattr(s08, 'refit_selected_vectorizer', refit)
    monkeypatch.setattr(s08, 'train_model', training)
    monkeypatch.setattr(s08, 'select_features', selection)
    summary = s08.execute(root)
    assert len(summary['success']) == 9, summary
    assert summary['failed'] == summary['skipped'] == summary['pending'] == []
    assert sorted(calls) == sorted([(s, 32) for s in (42, 43, 44)] +
                                  [(s, 8) for s in (42, 43, 44)] * 2)
    assert len(rankings) == len(refits) == 2
    assert before_full == {str(p): file_sha(p) for p in src['full'].rglob('*') if p.is_file()}
    frozen = s08.verify_training_freeze(root)
    assert frozen['gate'] == 'awaiting_user_approval'
    corpus_names = pd.read_parquet(cfg.corpus).name.tolist()
    for method in ('class_aware', 'global'):
        selected = s08.verify_selected(root, plan, method, full)
        assert selected.meta['source'] == full.meta['source']
        assert selected.meta['fit'] == full.meta['fit']
        X = selected.builder.transform(corpus_names)
        assert X.shape == (21, 8)
        for branch, offset, count in [('char', 0, 6), ('word', 6, 2)]:
            new = getattr(selected.builder, branch)
            old = getattr(full.builder, branch)
            indices = [old.vocabulary_[name] for name in new.get_feature_names_out()]
            assert len(indices) == count
            np.testing.assert_allclose(new.idf_, old.idf_[indices], atol=1e-6)
            norms = np.asarray(X[:, offset:offset+count].power(2).sum(axis=1)).ravel()
            np.testing.assert_allclose(norms[norms > 0], 1, atol=1e-6)
            sliced_norms = np.asarray(old.transform(corpus_names)[:, indices].power(2).sum(axis=1)).ravel()
            assert np.any(np.abs(sliced_norms - norms) > 1e-5), 'Refit must renormalize, not slice'
    # Public inference and raw predictor round-trip are checked only on development names.
    from sklearn.feature_extraction.text import TfidfVectorizer
    def forbidden_fit(*args, **kwargs):
        pytest.fail('Inference attempted to refit frozen TF-IDF')
    monkeypatch.setattr(TfidfVectorizer, 'fit', forbidden_fit)
    monkeypatch.setattr(TfidfVectorizer, 'fit_transform', forbidden_fit)
    names = src['pool'].name.tolist()
    for run in plan['runs']:
        checkpoint = root / 'runs' / run['run_id'] / 'checkpoint'
        art = load_artifact(str(checkpoint), torch.device('cpu'))
        reloaded = load_artifact(str(checkpoint), torch.device('cpu'))
        X = art.features.transform(names)
        raw, _ = predict(art.model, X, torch.device('cpu'), 8, 1)
        again, _ = predict(reloaded.model, reloaded.features.transform(names), torch.device('cpu'), 8, 1)
        np.testing.assert_array_equal(raw.l3_top1, again.l3_top1)
        result = predict_frame(art.model, X, art.maps, art.rule, torch.device('cpu'), batch_size=8)
        assert len(result) == 20 and result.l3_top1_raw.notna().all()
        assert art.maps.to_dict() == src['maps']
        saved = json.loads((root / 'runs' / run['run_id'] / 'result.json').read_text())
        assert saved['validation_macro_f1'] is saved['test_macro_f1'] is None
        assert not list((root / 'runs' / run['run_id']).glob('*test*'))


@pytest.mark.parametrize('damage', ['weights', 'label_maps', 'vocabulary', 'training', 'result_failed',
                                     'result_stale', 'selected_vocabulary', 'training_freeze'])
def test_frozen_training_rejects_every_model_feature_or_success_tamper(executed, damage):
    cfg, root, src, plan = executed
    base = root / 'runs' / plan['runs'][0]['run_id']
    files = {'weights': base / 'checkpoint' / 'model.pt',
             'label_maps': base / 'checkpoint' / 'label_maps.json',
             'vocabulary': base / 'checkpoint' / 'features.joblib',
             'training': base / 'training.json',
             'selected_vocabulary': root / 'shared' / 'class_aware' / 'tfidf' / 'features.joblib',
             'training_freeze': root / 'training_freeze.json'}
    if damage in ('result_failed', 'result_stale'):
        path = base / 'result.json'
        value = json.loads(path.read_text())
        if damage == 'result_failed':
            value['status'] = 'failed'
        else:
            value['provenance']['plan_sha256'] = '0' * 64
        write_json(path, reseal(value))
    else:
        with files[damage].open('ab') as handle:
            handle.write(b'tampered')
    with pytest.raises((RuntimeError, ValueError)):
        s08.verify_training_freeze(root)


def test_raw_top1_ties_and_empty_feature_bias_path_follow_frozen_runtime(executed):
    import scipy.sparse as sp
    _, root, _, plan = executed
    run = plan['runs'][0]
    art = load_artifact(str(root / 'runs' / run['run_id'] / 'checkpoint'), torch.device('cpu'))
    # Mutate only this in-memory test model; saved checkpoint remains untouched.
    with torch.no_grad():
        for parameter in art.model.parameters():
            parameter.zero_()
    X = sp.csr_matrix((2, art.features.n_features), dtype=np.float32)
    expected = int(torch.topk(torch.full((1, 2), .5), k=1, dim=1).indices[0, 0])
    raw, _ = predict(art.model, X, torch.device('cpu'), 8, 1)
    assert raw.l3_top1.tolist() == [expected, expected]
    assert np.isfinite(raw.l3_prob).all()
    assert json.loads((root / 'protocol.json').read_text())['tie_rule'].startswith('torch.topk(k=1)')
