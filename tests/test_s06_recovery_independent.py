"""Independent fail-closed and report recovery checks for S06."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from kkt_clf import s06_comparison as comp, s06_report as report


@pytest.mark.parametrize('field', ['protocol', 'corpus', 'split', 'full', 'seeds'])
def test_existing_plan_rejects_changed_inputs_without_overwrite(tmp_path, field):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path))
    Path(cfg.protocol_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.protocol_path).write_text('{"epochs": 2}')
    comp.write_plan(cfg)
    before = Path(cfg.planned_path).read_bytes()
    if field == 'protocol':
        Path(cfg.protocol_path).write_text('{"epochs": 3}')
    elif field == 'corpus':
        cfg.corpus_path = str(tmp_path / 'different-corpus')
    elif field == 'split':
        cfg.splits_manifest = str(tmp_path / 'different-split')
    elif field == 'full':
        cfg.full_tfidf_path = str(tmp_path / 'different-features')
    else:
        cfg.seeds = (42, 43)
    with pytest.raises((comp.S06ComparisonError, ValueError)):
        comp.write_plan(cfg)
    assert Path(cfg.planned_path).read_bytes() == before


@pytest.mark.parametrize('field', ['split', 'label_maps', 'feature', 'checkpoint'])
@pytest.mark.parametrize('mutation', ['missing', 'changed'])
def test_reference_checkpoint_rejects_unproven_contract(tmp_path, monkeypatch, field, mutation):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path))
    ckpt = Path(cfg.baseline_dir) / 'checkpoints' / 'seed_42'
    ckpt.mkdir(parents=True)
    weights = ckpt / 'model.pt'
    weights.write_bytes(b'valid weights')
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    selected = {'roundtrip': {'status': 'success'}, 'checkpoint_files_sha256': {'model.pt': digest}}
    selected_path = Path(cfg.baseline_dir) / 'selected_checkpoint.json'
    selected_path.write_text(json.dumps(selected))
    manifest = {'label_maps': {'version': 'one'}, 'parts': {
        'outer_train': {'sha256': 'train'}, 'train_monitor': {'sha256': 'monitor'}}}
    maps = SimpleNamespace(to_dict=lambda: {'version': 'one'})
    features = SimpleNamespace(fingerprint='features')
    art = SimpleNamespace(meta={'seed': 42, 'train': {'epochs_completed': 2, 'restored_best': False},
        'full_tfidf': {'full_tfidf_id': 'full'},
        'train_fingerprints': {'outer_train_sha256': 'train', 'train_monitor_sha256': 'monitor'}},
        model=SimpleNamespace(n_features=4), maps=maps, features=features)
    full = SimpleNamespace(builder=SimpleNamespace(n_features=4, fingerprint='features'))
    if field == 'split':
        art.meta['train_fingerprints']['outer_train_sha256'] = None if mutation == 'missing' else 'other'
    elif field == 'label_maps':
        art.maps = SimpleNamespace(to_dict=lambda: {} if mutation == 'missing' else {'version': 'other'})
    elif field == 'feature':
        art.features.fingerprint = None if mutation == 'missing' else 'other'
    else:
        selected['checkpoint_files_sha256'] = {} if mutation == 'missing' else {'model.pt': 'wrong'}
        selected_path.write_text(json.dumps(selected))
    monkeypatch.setattr(comp, 'load_artifact', lambda *a: art)
    monkeypatch.setattr(comp.LabelMaps, 'from_dict', lambda _: maps)
    monkeypatch.setattr(comp, '_resolved_train_sections', lambda _: (None, None, None, 2))
    monkeypatch.setattr(comp, 'feature_contract_fingerprint', lambda obj: obj.fingerprint)
    with pytest.raises((comp.S06ComparisonError, ValueError, FileNotFoundError)):
        comp._verify_reference_checkpoint(cfg, manifest, {'full_tfidf_id': 'full'}, full, {}, torch.device('cpu'))


def test_corrupt_saved_ranking_is_rejected(tmp_path):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path))
    directory = Path(cfg.rankings_dir('global'))
    directory.mkdir(parents=True)
    data = directory / 'ranking.npy'
    data.write_bytes(b'original')
    (directory / 'meta.json').write_text(json.dumps({'format': comp.RATINGS_FORMAT_VERSION,
        'method': 'global', 'files': {'ranking.npy': {'sha256': hashlib.sha256(data.read_bytes()).hexdigest(), 'bytes': 8}},
        'branches': {}}))
    (directory / 'rankings_meta.json').write_text(json.dumps({'ratings_dir': str(directory),
        'files': {'ranking.npy': {'sha256': hashlib.sha256(data.read_bytes()).hexdigest()}}}))
    data.write_bytes(b'corrupted')
    with pytest.raises((comp.S06ComparisonError, comp.SelectionArtifactError, ValueError)):
        comp._load_rankings_meta(cfg, 'global')


def test_non_success_statuses_do_not_contribute_to_summary():
    plan = {'runs': [{'run_id': f'global_k8_seed{s}', 'config_slug': 'global_k8',
        'method': 'global', 'budget': 8, 'seed': s} for s in (42, 43, 44, 45)]}
    results = {f'global_k8_seed{s}': {'status': status, 'config_slug': 'global_k8', 'seed': s,
        'comparison_val_eval': {'l3_macro_f1': score, 'l3_accuracy': score}}
        for s, status, score in [(42, 'success', .5), (43, 'failed', .99), (44, 'skipped', .98)]}
    frame = report._runs_frame(plan, results)
    assert dict(zip(frame.seed, frame.status)) == {42: 'success', 43: 'failed', 44: 'skipped', 45: 'pending'}
    summary = report._method_summary(frame)[0]
    assert summary['n_seed_success'] == 1
    assert summary['l3_macro_f1']['mean'] == pytest.approx(.5)


def test_baseline_is_in_summary():
    plan = {'runs': [{'run_id': 'none_seed42', 'config_slug': 'none', 'method': 'none', 'budget': None, 'seed': 42}]}
    results = {'none_seed42': {'status': 'success', 'config_slug': 'none', 'seed': 42,
        'comparison_val_eval': {'l3_macro_f1': .5, 'l3_accuracy': .5}}}
    summary = report._method_summary(report._runs_frame(plan, results))
    assert any(row['method'] == 'none' and row['n_seed_success'] == 1 for row in summary)


def test_null_prediction_is_counted_as_error():
    per_class = pd.DataFrame([{'run_id': 'none_seed42', 'method': 'none', 'budget': None,
        'seed': 42, 'L3': 'a', 'support': 2}])
    predictions = pd.DataFrame([{'run_id': 'none_seed42', 'true_L3': 'a', 'predicted_L3': None, 'correct': False},
        {'run_id': 'none_seed42', 'true_L3': 'a', 'predicted_L3': 'a', 'correct': True}])
    confusion = report._confusion_frame(predictions, per_class)
    assert len(confusion) == 1 and pd.isna(confusion.iloc[0]['predicted_L3'])
    assert confusion.iloc[0]['count'] == 1 and confusion.iloc[0]['error_share'] == pytest.approx(.5)


def test_smoke_chooses_new_directory_without_touching_existing(tmp_path, monkeypatch):
    from scripts import s06_comparison_smoke as smoke
    occupied = tmp_path / 'S06_recovery_smoke'
    occupied.mkdir()
    marker = occupied / 'user.txt'
    marker.write_bytes(b'\x00keep\xff')
    monkeypatch.setattr(smoke, 'SMOKE_BASE', str(occupied))
    selected = Path(smoke._fresh_sandbox())
    assert selected != occupied and not selected.exists()
    assert marker.read_bytes() == b'\x00keep\xff'


def test_html_rebuild_uses_report_tables_with_sources_unavailable(tmp_path, monkeypatch):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path), methods=(), budgets=(8,), seeds=(42,))
    Path(cfg.protocol_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.protocol_path).write_text('{}')
    comp.write_plan(cfg)
    report.build_s06_report(str(tmp_path))
    (tmp_path / 'runs').rename(tmp_path / 'unavailable_runs')
    if (tmp_path / 'shared').exists():
        (tmp_path / 'shared').rename(tmp_path / 'unavailable_shared')
    def forbidden(*args, **kwargs):
        pytest.fail('model or training access during report reconstruction')
    monkeypatch.setattr(comp, 'run_train', forbidden)
    monkeypatch.setattr(comp, 'load_artifact', forbidden)
    destination = tmp_path / 'rebuilt'
    report.build_s06_report(str(tmp_path), str(destination))
    assert (destination / 'report.html').is_file()
    assert pd.read_csv(destination / 'runs.csv').status.tolist() == ['pending']


def test_report_does_not_mix_predictions_from_non_successful_runs(tmp_path):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path), methods=('global',), budgets=(8,), seeds=(42, 43))
    Path(cfg.protocol_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.protocol_path).write_text('{}')
    plan = comp.write_plan(cfg)
    prediction_dir = tmp_path / 'runs' / 'predictions'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    statuses = ('failed', 'skipped', 'pending', 'failed')
    for run, status in zip(plan['runs'], statuses):
        pd.DataFrame([{'row_id': 1, 'run_id': run['run_id'], 'true_L3': 'a',
                       'predicted_L3': 'b', 'correct': False}]).to_parquet(prediction_dir / (run['run_id'] + '.parquet'))
        if status != 'pending':
            result = Path(cfg.run_result_path(run['config_slug'], run['seed']))
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps({'status': status, 'reason': 'not completed'}))
    report.build_s06_report(str(tmp_path))
    runs = pd.read_csv(tmp_path / 'report' / 'runs.csv')
    assert runs.status.tolist() == list(statuses)
    assert pd.read_parquet(tmp_path / 'report' / 'predictions_val.parquet').empty
    assert pd.read_csv(tmp_path / 'report' / 'confusion_pairs.csv').empty


def test_plan_with_missing_protocol_fingerprint_is_not_accepted(tmp_path):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path))
    Path(cfg.protocol_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.protocol_path).write_text('{}')
    comp.write_plan(cfg)
    path = Path(cfg.planned_path)
    plan = json.loads(path.read_text())
    plan.pop('protocol_sha256')
    path.write_text(json.dumps(plan))
    before = path.read_bytes()
    with pytest.raises((comp.S06ComparisonError, ValueError, KeyError)):
        comp.write_plan(cfg)
    assert path.read_bytes() == before
