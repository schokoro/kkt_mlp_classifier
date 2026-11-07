"""Small independent contract checks for S06 gates 2–4."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from kkt_clf import s06_comparison as comparison
from kkt_clf import s06_report as report
from kkt_clf.experiment import _l3_val_metrics
from kkt_clf.selection import branch_budgets


def test_plan_pairs_all_configurations_and_resume_rejects_changes(tmp_path):
    cfg = comparison.S06ComparisonConfig(output_root=str(tmp_path))
    Path(cfg.protocol_path).parent.mkdir(parents=True)
    Path(cfg.protocol_path).write_text('{}')
    plan = comparison.write_plan(cfg)
    assert plan['protocol_sha256'] == comparison._sha256_file(cfg.protocol_path)
    assert len(plan['runs']) == 30
    assert len({r['run_id'] for r in plan['runs']}) == 30
    for config in cfg.planned_configs():
        assert {r['seed'] for r in plan['runs'] if r['config_slug'] == config['slug']} == {42, 43, 44}
    before = open(cfg.planned_path, 'rb').read()
    comparison.write_plan(cfg)
    assert open(cfg.planned_path, 'rb').read() == before
    with pytest.raises(comparison.S06ComparisonError):
        comparison.write_plan(comparison.S06ComparisonConfig(output_root=str(tmp_path), seeds=(42, 43)))
    assert open(cfg.planned_path, 'rb').read() == before


def test_branch_budget_never_silently_shrinks(tmp_path, monkeypatch):
    cfg = comparison.S06ComparisonConfig(
        output_root=str(tmp_path), methods=('global',), budgets=(8,), char_share=.75,
    )
    manifest = {'label_maps': {}, 'parts': {'outer_train': {'sha256': 'train'}}}
    monkeypatch.setattr(comparison, 'verify_preconditions', lambda _: {
        'manifest': manifest, 'verification': {}, 'protocol': {},
    })
    builder = SimpleNamespace(n_char_features=3, n_features=13)
    monkeypatch.setattr(comparison, 'load_vectorizer_artifact', lambda _: SimpleNamespace(builder=builder))
    maps = SimpleNamespace(l3=['a'], to_dict=lambda: {})
    monkeypatch.setattr(comparison.LabelMaps, 'from_dict', lambda _: maps)
    from scipy.sparse import csr_matrix
    monkeypatch.setattr(comparison, '_transform_outer_train', lambda *args: (
        csr_matrix((1, 13)), np.zeros((3, 1), dtype=int), None, None,
    ))
    with pytest.raises(comparison.S06ComparisonError, match='(?i)бюджет|budget'):
        comparison.run_rankings(cfg, torch.device('cpu'))


def test_ranking_transform_reads_outer_train_only(tmp_path, monkeypatch):
    cfg = comparison.S06ComparisonConfig(output_root=str(tmp_path))
    calls = []
    data = pd.DataFrame({'name': ['train'], 'L3': ['a']})
    monkeypatch.setattr(comparison, 'load_part', lambda manifest, part: calls.append(part) or data)
    monkeypatch.setattr(comparison.LabelMaps, 'from_dict', lambda _: object())
    monkeypatch.setattr(comparison, '_encode_labels', lambda maps, frame, part: np.array([[0], [0], [0]]))
    from scipy.sparse import csr_matrix
    monkeypatch.setattr(comparison, '_transform', lambda builder, names, chunks: csr_matrix([[1.0]]))
    comparison._transform_outer_train(cfg, object(), {'label_maps': {}}, torch.device('cpu'))
    assert calls == ['outer_train']


def test_checkpoint_missing_fails_closed(tmp_path):
    cfg = comparison.S06ComparisonConfig(output_root=str(tmp_path))
    with pytest.raises(comparison.S06ComparisonError, match='checkpoint'):
        comparison._verify_reference_checkpoint(cfg, {}, {}, None, {}, torch.device('cpu'))


def test_fixed_vocab_refit_refuses_incomplete_provenance():
    from kkt_clf.vectorizer import refit_selected_vectorizer, VectorizerError
    with pytest.raises(VectorizerError, match='[Мм]етадан'):
        refit_selected_vectorizer('unused', SimpleNamespace(fitted=True), {}, None)


def test_macro_f1_fixed_universe_and_invalid_prediction():
    macro, acc, rows = _l3_val_metrics(np.array([0, 0, 1]), np.array([0, -1, 0]), ['a', 'b', 'absent'])
    assert macro == pytest.approx(0.25)
    assert acc == pytest.approx(1 / 3)
    assert [r['support'] for r in rows] == [2, 1, 0]
    assert rows[2]['f1'] is None


def test_paired_deltas_and_missing_baseline():
    cfg = comparison.S06ComparisonConfig(methods=('global',), budgets=(8,))
    plan = {'runs': [{'run_id': cfg.run_id(c['slug'], s), 'config_slug': c['slug'], 'method': c['method'], 'budget': c['budget'], 'seed': s} for c in cfg.planned_configs() for s in (42, 43)]}
    def result(slug, seed, f1):
        return {'status': 'success', 'config_slug': slug, 'seed': seed, 'comparison_val_eval': {'l3_macro_f1': f1, 'l3_accuracy': f1}}
    results = {'none_seed42': result('none', 42, .123456), 'global_k8_seed42': result('global_k8', 42, .223457), 'global_k8_seed43': result('global_k8', 43, .9)}
    rows = report._runs_frame(plan, results).set_index('run_id')
    assert rows.loc['global_k8_seed42', 'delta_macro_f1_vs_none_same_seed'] == pytest.approx(.100001)
    assert pd.isna(rows.loc['global_k8_seed43', 'delta_macro_f1_vs_none_same_seed'])


def test_support_bins_unweighted_and_confusion_includes_null():
    pc = pd.DataFrame([{'run_id': 'none_seed42', 'method': 'none', 'budget': None, 'seed': 42, 'L3': label, 'support': support, 'f1': f1} for label, support, f1 in [('a', 2, 1.), ('b', 4, 0.)]])
    bins = report._support_bins_frame(pc, pd.DataFrame()).set_index('support_bin')
    assert bins.loc['2-4', 'macro_f1_bin'] == pytest.approx(.5)
    assert bins.loc['5-9', 'n_L3'] == 0
    assert pd.isna(bins.loc['5-9', 'macro_f1_bin'])
    pred = pd.DataFrame([{'run_id': 'none_seed42', 'true_L3': 'a', 'predicted_L3': None, 'correct': False}, {'run_id': 'none_seed42', 'true_L3': 'a', 'predicted_L3': 'b', 'correct': False}])
    confusion = report._confusion_frame(pred, pc)
    assert len(confusion) == 2
    assert confusion['error_share'].tolist() == [0.5, 0.5]


def test_prediction_loader_only_reads_planned_runs(tmp_path):
    path = tmp_path / 'runs' / 'predictions'
    path.mkdir(parents=True)
    for run in ('allowed', 'sealed_test'):
        pd.DataFrame({'row_id': [1], 'run_id': [run], 'true_L3': ['a'], 'predicted_L3': ['a'], 'correct': [True]}).to_parquet(path / f'{run}.parquet')
    assert report._load_predictions(str(tmp_path), ['allowed'])['run_id'].tolist() == ['allowed']


def test_report_rebuild_from_saved_tables_without_training(tmp_path, monkeypatch):
    cfg = comparison.S06ComparisonConfig(output_root=str(tmp_path))
    (tmp_path / 'shared').mkdir()
    (tmp_path / 'shared' / 'training_protocol.json').write_text('{}')
    comparison.write_plan(cfg)
    report.build_s06_report(str(tmp_path))
    # A table-only re-render must not depend on run results, splits or model operations.
    (tmp_path / 'runs' / 'planned.json').unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('training or inference called')
    monkeypatch.setattr(comparison, 'run_train', forbidden)
    monkeypatch.setattr(comparison, 'run_refit_selected', forbidden)
    report.build_s06_report(str(tmp_path), str(tmp_path / 'rerender'))
    assert (tmp_path / 'rerender' / 'report.html').exists()
