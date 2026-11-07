"""Isolated regression checks for the issues recorded in S06 review 2."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from kkt_clf import s06_baseline as baseline, s06_comparison as comp, s06_report as report
from tests.test_s06_baseline_independent import setup, freeze


def test_missing_baseline_gate_blocks_before_ranking(tmp_path, monkeypatch):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path))
    Path(cfg.baseline_dir).mkdir(parents=True)
    monkeypatch.setattr(comp, 'verify_gate0_split', lambda _: {'policy': {'min_count': 12}})
    monkeypatch.setattr(comp, 'verify_full_tfidf', lambda *a: {'decision': 'ok', 'full_tfidf_id': 'full'})
    monkeypatch.setattr(comp, 'load_training_protocol', lambda _: {'fixed_epochs': 1})
    monkeypatch.setattr(comp, 'validate_protocol_against_inputs', lambda *a: None)
    with pytest.raises((comp.S06ComparisonError, ValueError, FileNotFoundError)):
        comp.verify_preconditions(cfg)


def test_checkpoint_hash_mismatch_blocks_attribution(tmp_path, monkeypatch):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path))
    ckpt = Path(cfg.baseline_dir) / 'checkpoints' / 'seed_42'
    ckpt.mkdir(parents=True)
    (ckpt / 'model.pt').write_bytes(b'changed')
    (Path(cfg.baseline_dir) / 'selected_checkpoint.json').write_text(json.dumps({
        'roundtrip': {'status': 'success'}, 'checkpoint_files_sha256': {'model.pt': 'old-hash'}}))
    maps = SimpleNamespace(to_dict=lambda: {'labels': 'same'})
    builder = SimpleNamespace(n_features=2)
    art = SimpleNamespace(meta={'seed': 42, 'train': {'epochs_completed': 1, 'restored_best': False},
                                'full_tfidf': {'full_tfidf_id': 'full'},
                                'train_fingerprints': {'outer_train_sha256': 'outer', 'train_monitor_sha256': 'mon'}},
                          model=SimpleNamespace(n_features=2), maps=maps, features=builder)
    monkeypatch.setattr(comp, 'load_artifact', lambda *a: art)
    monkeypatch.setattr(comp.LabelMaps, 'from_dict', lambda _: maps)
    monkeypatch.setattr(comp, '_resolved_train_sections', lambda _: (None, None, None, 1))
    monkeypatch.setattr(comp, 'feature_contract_fingerprint', lambda _: 'same')
    manifest = {'label_maps': {}, 'parts': {'outer_train': {'sha256': 'outer'}, 'train_monitor': {'sha256': 'mon'}}}
    with pytest.raises((comp.S06ComparisonError, ValueError)):
        comp._verify_reference_checkpoint(cfg, manifest, {'full_tfidf_id': 'full'}, SimpleNamespace(builder=builder), {}, torch.device('cpu'))


@pytest.mark.parametrize('change', ['protocol', 'corpus', 'split', 'full'])
def test_plan_resume_rejects_changed_inputs(tmp_path, change):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path))
    Path(cfg.protocol_path).parent.mkdir(parents=True)
    Path(cfg.protocol_path).write_text('{"version": "first"}')
    comp.write_plan(cfg)
    before = Path(cfg.planned_path).read_bytes()
    if change == 'protocol': Path(cfg.protocol_path).write_text('{"version": "second"}')
    elif change == 'corpus': cfg.corpus_path = 'another-corpus'
    elif change == 'split': cfg.splits_manifest = 'another-split'
    else: cfg.full_tfidf_path = 'another-full'
    with pytest.raises((comp.S06ComparisonError, ValueError)):
        comp.write_plan(cfg)
    assert Path(cfg.planned_path).read_bytes() == before


def test_embedded_selection_report_must_match_file(setup, monkeypatch):
    cfg, manifest, verification, loaded, _ = setup
    protocol = copy.deepcopy(freeze(setup))
    protocol['selection_report']['candidates'][0]['best_epoch'] += 1
    monkeypatch.setattr(baseline, 'load_training_protocol', lambda _: protocol)
    monkeypatch.setattr(baseline, 'run_refit', lambda *a, **kw: pytest.fail('refit started'))
    with pytest.raises((baseline.S06Gate1Error, ValueError)):
        baseline.run_gate1_refit(cfg, manifest, verification, loaded, torch.device('cpu'))


@pytest.mark.parametrize('predicted,correct', [('a', True), (None, False)])
def test_confusion_handles_perfect_and_null_predictions(predicted, correct):
    pc = pd.DataFrame([{'run_id': 'r', 'method': 'none', 'budget': None, 'seed': 42,
                        'L3': 'a', 'support': 1, 'f1': 1.0 if correct else 0.0}])
    pred = pd.DataFrame([{'run_id': 'r', 'true_L3': 'a', 'predicted_L3': predicted, 'correct': correct}])
    result = report._confusion_frame(pred, pc)
    assert len(result) == (0 if correct else 1)
    if not correct:
        assert pd.isna(result.iloc[0]['predicted_L3'])
        assert result.iloc[0]['count'] == 1


def test_report_excludes_failed_and_pending_predictions(tmp_path):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path), methods=('global',), budgets=(8,), seeds=(42,))
    Path(cfg.protocol_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.protocol_path).write_text('{}')
    plan = comp.write_plan(cfg)
    pred_dir = tmp_path / 'runs' / 'predictions'
    pred_dir.mkdir(parents=True)
    for run in plan['runs']:
        pd.DataFrame([{'row_id': 1, 'run_id': run['run_id'], 'true_L3': 'a', 'predicted_L3': 'b', 'correct': False}]).to_parquet(pred_dir / (run['run_id'] + '.parquet'))
    failed = Path(cfg.run_result_path('none', 42))
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text(json.dumps({'status': 'failed', 'reason': 'evaluation failed'}))
    report.build_s06_report(str(tmp_path))
    assert pd.read_parquet(tmp_path / 'report' / 'predictions_val.parquet').empty


def test_smoke_preserves_existing_directory(tmp_path, monkeypatch):
    from scripts import s06_comparison_smoke as smoke
    root = tmp_path
    path = root / 'outputs' / 'S06_recovery_smoke'
    path.mkdir(parents=True)
    marker = path / 'user-data'
    marker.write_bytes(b'\x00keep\xff')
    nested = path / 'nested' / 'user-data'
    nested.parent.mkdir()
    nested.write_bytes(b'\xffother\x00')
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    monkeypatch.setattr(smoke, 'ROOT', str(root))
    monkeypatch.setattr(smoke, 'SMOKE_BASE', str(path))
    selected = []
    def stop(sandbox):
        selected.append(Path(sandbox))
        raise RuntimeError('stop before execution')
    monkeypatch.setattr(smoke, 'make_tiny_data', stop)
    with pytest.raises(RuntimeError, match='stop before execution'):
        smoke.main()
    assert len(selected) == 1 and selected[0] != path and selected[0].is_dir()
    assert {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()} == before


def test_baseline_cli_reports_original_error(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_s06_baseline.py'
    p = subprocess.run([sys.executable, str(script), 'all', '--output', str(tmp_path / 'out'),
                        '--splits-manifest', str(tmp_path / 'missing.json')], capture_output=True, text=True)
    assert p.returncode != 0
    assert 'NameError' not in p.stderr
    assert 'missing.json' in p.stderr


def test_cached_refit_rejects_different_provenance(tmp_path, monkeypatch):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path), methods=('global',), budgets=(4,), seeds=(42,))
    inputs = {'manifest': {'parts': {'outer_train': {'sha256': 'current'}}},
              'verification': {'full_tfidf_id': 'current'}, 'protocol': {}}
    monkeypatch.setattr(comp, 'verify_preconditions', lambda _: inputs)
    builder = SimpleNamespace(n_char_features=3, n_word_features=1, n_features=4)
    monkeypatch.setattr(comp, 'load_vectorizer_artifact', lambda _: SimpleNamespace(builder=builder, meta={}))
    result = Path(cfg.config_dir('global_k4')) / 'refit_result.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({'status': 'success', 'vocab': {'n_char': 3, 'n_word': 1, 'n_total': 4},
                                  'provenance': {'outer_train_sha256': 'old', 'full_tfidf_id': 'old'}}))
    with pytest.raises((comp.S06ComparisonError, ValueError)):
        comp.run_refit_selected(cfg)


def test_cached_ranking_rejects_other_split(tmp_path, monkeypatch):
    import numpy as np
    import scipy.sparse as sp
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path), methods=('global',), budgets=(4,), seeds=(42,))
    inputs = {'manifest': {'parts': {'outer_train': {'sha256': 'current'}}, 'label_maps': {}},
              'verification': {'full_tfidf_id': 'current'}, 'protocol': {}}
    monkeypatch.setattr(comp, 'verify_preconditions', lambda _: inputs)
    builder = SimpleNamespace(n_char_features=3, n_features=4)
    monkeypatch.setattr(comp, 'load_vectorizer_artifact', lambda _: SimpleNamespace(builder=builder))
    monkeypatch.setattr(comp.LabelMaps, 'from_dict', lambda _: SimpleNamespace(l3=['a'], to_dict=lambda: {}))
    monkeypatch.setattr(comp, '_transform_outer_train', lambda *a: (sp.csr_matrix([[1, 0, 0, 0]]), np.array([[0], [0], [0]]), None, None))
    monkeypatch.setattr(comp, '_load_rankings_meta', lambda *a: ('old', {'outer_train_sha256': 'old', 'full_tfidf_id': 'old'}))
    monkeypatch.setattr(comp, '_append_stage_rows', lambda *a: pytest.fail('stale ranking accepted for reuse'))
    with pytest.raises((comp.S06ComparisonError, ValueError)):
        comp.run_rankings(cfg, torch.device('cpu'))


def test_cached_selection_rejects_other_split(tmp_path, monkeypatch):
    import numpy as np
    import scipy.sparse as sp
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path), methods=('global',), budgets=(4,), seeds=(42,))
    inputs = {'manifest': {'parts': {'outer_train': {'sha256': 'current'}}, 'label_maps': {}},
              'verification': {'full_tfidf_id': 'current'}, 'protocol': {}}
    monkeypatch.setattr(comp, 'verify_preconditions', lambda _: inputs)
    monkeypatch.setattr(comp, 'load_vectorizer_artifact', lambda _: SimpleNamespace(builder=SimpleNamespace(n_char_features=3, n_features=4)))
    monkeypatch.setattr(comp.LabelMaps, 'from_dict', lambda _: SimpleNamespace(l3=['a'], to_dict=lambda: {}))
    monkeypatch.setattr(comp, '_transform_outer_train', lambda *a: (sp.csr_matrix([[1, 0, 0, 0]]), np.array([[0], [0], [0]]), None, None))
    monkeypatch.setattr(comp, '_load_rankings_meta', lambda *a: ('valid', {'outer_train_sha256': 'current', 'full_tfidf_id': 'current'}))
    monkeypatch.setattr(comp, '_append_stage_rows', lambda *a: None)
    monkeypatch.setattr(comp, '_sha256_file', lambda _: 'hash')
    Path(cfg.selection_dir('global_k4')).mkdir(parents=True)
    monkeypatch.setattr(comp, 'load_selection_artifact', lambda _: (SimpleNamespace(k_requested=4), {'provenance': {'labeled': {'outer_train_sha256': 'old'}}}))
    with pytest.raises((comp.S06ComparisonError, ValueError)):
        comp.run_rankings(cfg, torch.device('cpu'))


def test_failed_run_is_not_counted_as_done_on_resume(tmp_path, monkeypatch):
    import numpy as np
    import scipy.sparse as sp
    from kkt_clf.config import LossConfig, TrainConfig
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path), methods=(), budgets=(4,), seeds=(42,))
    Path(cfg.protocol_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.protocol_path).write_text('{}')
    comp.write_plan(cfg)
    result = Path(cfg.run_result_path('none', 42))
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({'status': 'failed', 'reason': 'previous failure'}))
    monkeypatch.setattr(comp, 'verify_preconditions', lambda _: {'manifest': {'label_maps': {}}, 'verification': {}, 'protocol': {}})
    monkeypatch.setattr(comp, '_resolved_train_sections', lambda _: (None, LossConfig(), TrainConfig(), 1))
    monkeypatch.setattr(comp, 'load_vectorizer_artifact', lambda _: SimpleNamespace(builder=object(), meta={}))
    monkeypatch.setattr(comp.LabelMaps, 'from_dict', lambda _: SimpleNamespace(l3=['a'], n_classes=lambda _: 1))
    frame = pd.DataFrame({'row_id': [1], 'name': ['x'], 'L1': ['a'], 'L2': ['a'], 'L3': ['a']})
    monkeypatch.setattr(comp, 'load_part', lambda *a: frame)
    monkeypatch.setattr(comp, '_encode_labels', lambda *a: np.array([[0], [0], [0]]))
    monkeypatch.setattr(comp, 'compute_class_weights', lambda *a: None)
    monkeypatch.setattr(comp, '_read_gate1_baseline_runs', lambda _: pd.DataFrame())
    monkeypatch.setattr(comp, '_transform', lambda *a: sp.csr_matrix([[1]]))
    monkeypatch.setattr(comp, '_stage_recorded', lambda *a: True)
    summary = comp.run_train(cfg, torch.device('cpu'))
    assert summary['n_done'] == 0
    assert 'none_seed42' in summary['failed'] or 'none_seed42' in summary['skipped']
    assert json.loads(result.read_text())['status'] == 'failed'
