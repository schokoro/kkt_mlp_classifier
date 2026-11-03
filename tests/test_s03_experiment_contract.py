"""Contract checks for the S03 experiment and its saved-result report."""
import json
from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from kkt_clf import experiment as exp


def test_default_plan_has_seven_distinct_configurations():
    plans = exp.planned_runs(exp.ExperimentConfig())
    assert [(p.method, p.requested_k) for p in plans] == [
        ('none', None), ('global', 50000), ('global', 100000), ('global', 200000),
        ('class_aware', 50000), ('class_aware', 100000), ('class_aware', 200000)]
    assert len({p.run_id for p in plans}) == 7


def test_validation_metrics_fixed_classes_and_zero_support():
    macro, accuracy, rows = exp._l3_val_metrics(np.array([0, 0, 1]), np.array([0, 1, 0]), ['a', 'b', 'absent'])
    assert accuracy == pytest.approx(1 / 3)
    assert macro == pytest.approx(1 / 4)  # F1(a)=1/2, F1(b)=0
    assert [r['L3'] for r in rows] == ['a', 'b', 'absent']
    assert rows[0]['precision'] == pytest.approx(1 / 2)
    assert rows[0]['recall'] == pytest.approx(1 / 2)
    assert rows[1]['f1'] == 0
    assert rows[2]['support'] == 0
    assert all(rows[2][k] is None for k in ('precision', 'recall', 'f1'))


def test_selector_receives_only_training_labels(monkeypatch, tmp_path):
    class StopSelection(Exception):
        pass
    train_labels = np.array([0, 1, 0])
    def check(X, y, cfg, n_char, l3_labels):
        np.testing.assert_array_equal(y, train_labels)
        assert X.shape[0] == len(train_labels)
        raise StopSelection
    monkeypatch.setattr(exp, 'select_features', check)
    ctx = SimpleNamespace(cfg=SimpleNamespace(selection=SimpleNamespace(char_share=.5), seed=42),
                          device=torch.device('cpu'), ensure_full_train=lambda: (sp.eye(3, format='csr'), False),
                          y_train=(None, None, train_labels), n_char_full=2, l3_labels=['a', 'b'])
    with pytest.raises(StopSelection):
        exp._run_selected(exp.PlannedRun('global_2', 'global', 2), ctx, {'stages': []}, str(tmp_path))


def _saved_result(run_id, score, val='same-val', labels='same-labels'):
    return {'run_id': run_id, 'status': 'success', 'error': None,
            'val_sha256': val, 'label_map_sha256': labels, 'seed': 42,
            'n_train': 10, 'n_val': 3, 'n_classes_eval': 3,
            'n_features': 10, 'n_char': 7, 'n_word': 3,
            'metrics': {'l3_macro_f1': score, 'l3_accuracy': score},
            'per_class': [{'L3': 'a', 'support': 2, 'precision': score, 'recall': score, 'f1': score},
                          {'L3': 'b', 'support': 1, 'precision': 0., 'recall': 0., 'f1': 0.},
                          {'L3': 'absent', 'support': 0, 'precision': None, 'recall': None, 'f1': None}],
            'train_seconds': 2., 'val_inference_seconds': .1, 'model_size_bytes': 100,
            'artifacts': {'model': 'model', 'vectorizer': 'vectorizer', 'selection': None},
            'stages': [{'run_id': run_id, 'stage': stage, 'status': 'success', 'seconds': 1.,
                        'peak_rss_bytes': 1234, 'peak_vram_bytes': None, 'measurement_note': 'CPU RSS sample'}
                       for stage in ('selection', 'selected_tfidf_refit', 'mlp_train', 'validation_inference')]}


@pytest.fixture
def saved_experiment(tmp_path):
    plan = [{'run_id': p.run_id, 'method': p.method, 'requested_k': p.requested_k}
            for p in exp.planned_runs(exp.ExperimentConfig())]
    metadata = {'experiment_id': 'fixture', 'created_at': '2026-01-01', 'git_commit': 'abc', 'git_dirty': False,
                'planned_runs': plan, 'config': {'seed': 42, 'selection': {}, 'data': {}, 'model': {}, 'train': {}},
                'environment': {'device': 'cpu', 'versions': {'python': '3.12'}},
                'inputs': {'corpus': {'path': 'corpus', 'sha256': 'a'}, 'labeled': {'path': 'data', 'sha256': 'b'},
                           'full_tfidf': {'path': 'full', 'artifact_version': '1'}},
                'splits': {x: {'sha256': x} for x in ('train', 'val', 'test')},
                'label_maps': {'sha256': 'same-labels', 'l3_train_counts': {'a': 5, 'b': 5, 'absent': 1}},
                'metric_protocol': {'metric': 'raw top-1', 'zero_division': 0}, 'limitations': ['fixture']}
    (tmp_path / 'experiment.json').write_text(json.dumps(metadata), encoding='utf8')
    for result in (_saved_result('none', .5), _saved_result('global_50000', .75),
                   {'run_id': 'class_aware_50000', 'status': 'failed', 'error': 'out of memory', 'stages': [], 'artifacts': {}}):
        d = tmp_path / 'runs' / result['run_id']
        d.mkdir(parents=True)
        (d / 'run_result.json').write_text(json.dumps(result), encoding='utf8')
    return tmp_path


def test_report_schemas_statuses_metrics_and_missing_measurements(saved_experiment):
    paths = exp.build_report(str(saved_experiment))
    assert {p.name for p in (saved_experiment / 'report').iterdir()} >= {'report.html', 'runs.csv', 'per_class.parquet', 'stages.csv', 'manifest.json'}
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert set(runs.columns) >= set('status error method requested_k n_features n_char n_word n_train n_val n_classes_eval seed l3_macro_f1 l3_accuracy delta_macro_f1 delta_accuracy train_seconds val_inference_seconds model_size_bytes vectorizer_artifact model_artifact selection_artifact'.split())
    assert len(runs) == 7
    assert runs.loc['global_50000', 'delta_macro_f1'] == pytest.approx(.25)
    assert runs.loc['global_50000', 'delta_accuracy'] == pytest.approx(.25)
    assert runs.loc['class_aware_50000', 'status'] == 'failed'
    assert 'memory' in runs.loc['class_aware_50000', 'error']
    assert runs.loc['global_100000', 'status'] == 'skipped'
    assert pd.isna(runs.loc['global_100000', 'train_seconds'])
    pc = pd.read_parquet(paths['per_class_parquet'])
    assert set(pc.columns) >= set('run_id L3 support precision recall f1 baseline_f1 delta_f1 baseline_recall delta_recall'.split())
    assert set(pc.run_id) == {'none', 'global_50000'}
    row = pc.query("run_id == 'global_50000' and L3 == 'a'").iloc[0]
    assert row.delta_f1 == pytest.approx(.25) and row.delta_recall == pytest.approx(.25)
    absent = pc.query("L3 == 'absent'")
    assert absent[['f1', 'recall', 'delta_f1', 'delta_recall']].isna().all().all()
    stages = pd.read_csv(paths['stages_csv'])
    assert set(stages.columns) >= set('run_id stage status seconds peak_rss_bytes peak_vram_bytes measurement_note'.split())
    assert set(stages.stage) >= {'selection', 'selected_tfidf_refit', 'mlp_train', 'validation_inference'}
    assert stages.peak_vram_bytes.isna().all()
    manifest = json.loads(open(paths['manifest_json'], encoding='utf8').read())
    assert set(manifest) >= set('schema_version experiment_id date git_commit git_dirty versions device resource_limitations inputs splits label_maps configs metric_protocol planned_runs'.split())
    assert len(manifest['planned_runs']) == 7


@pytest.mark.parametrize('field', ['val_sha256', 'label_map_sha256'])
def test_no_deltas_for_incomparable_validation_or_classes(saved_experiment, field):
    path = saved_experiment / 'runs/global_50000/run_result.json'
    r = json.loads(path.read_text())
    r[field] = 'different'
    path.write_text(json.dumps(r))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert pd.isna(runs.loc['global_50000', 'delta_macro_f1'])
    pc = pd.read_parquet(paths['per_class_parquet']).query("run_id == 'global_50000'")
    assert pc[['baseline_f1', 'baseline_recall', 'delta_f1', 'delta_recall']].isna().all().all()


class _ExternalAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []
    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('src', 'href') and value and not value.startswith(('#', 'data:')):
                self.external.append((tag, value))


def test_report_regenerates_without_training_and_is_self_contained(saved_experiment, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('report generation must not train or select features')
    monkeypatch.setattr(exp, '_train_mlp', forbidden)
    monkeypatch.setattr(exp, 'select_features', forbidden)
    paths = exp.build_report(str(saved_experiment))
    exp.build_report(str(saved_experiment))
    html = open(paths['report_html'], encoding='utf8').read()
    parser = _ExternalAssets()
    parser.feed(html)
    assert not parser.external
    assert '<script' not in html.lower()
    assert 'ODS/PoC' in html and 'support' in html
    assert 'out of memory' in html and 'global_100000' in html


def test_missing_comparison_identity_does_not_prove_same_validation(saved_experiment):
    for run_id in ('none', 'global_50000'):
        path = saved_experiment / 'runs' / run_id / 'run_result.json'
        result = json.loads(path.read_text())
        result.pop('val_sha256')
        result.pop('label_map_sha256')
        path.write_text(json.dumps(result))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert pd.isna(runs.loc['global_50000', 'delta_macro_f1'])


def test_failed_baseline_has_no_comparisons(saved_experiment):
    path = saved_experiment / 'runs/none/run_result.json'
    result = json.loads(path.read_text())
    result.update(status='failed', error='training failed')
    path.write_text(json.dumps(result))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert pd.isna(runs.loc['global_50000', 'delta_macro_f1'])
    pc = pd.read_parquet(paths['per_class_parquet'])
    assert set(pc.run_id) == {'global_50000'}
    assert pc[['baseline_f1', 'delta_f1', 'baseline_recall', 'delta_recall']].isna().all().all()


def test_all_runs_share_prepared_splits_and_maps(monkeypatch, tmp_path):
    cfg = exp.ExperimentConfig(output_dir=str(tmp_path / 'output'))
    monkeypatch.setattr(cfg, 'validate', lambda: None)
    monkeypatch.setattr(exp, 'resolve_device', lambda *a: torch.device('cpu'))
    context = object()
    preparations, seen = [], []
    def prepare(*args):
        preparations.append(1)
        return context
    monkeypatch.setattr(exp, '_prepare_context', prepare)
    monkeypatch.setattr(exp, '_write_experiment_meta', lambda *a: 'experiment.json')
    monkeypatch.setattr(exp, '_fresh_run_result', lambda p, c: {'run_id': p.run_id, 'stages': [], 'error': None})
    def run(plan, ctx, result, path):
        seen.append((plan.run_id, ctx))
    monkeypatch.setattr(exp, '_run_none', run)
    monkeypatch.setattr(exp, '_run_selected', run)
    result = exp.run_experiment(cfg)
    assert len(preparations) == 1
    assert len(seen) == 7 and all(ctx is context for _, ctx in seen)
    assert set(result['statuses'].values()) == {'success'}
