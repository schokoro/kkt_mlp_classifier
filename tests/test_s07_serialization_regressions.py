"""Independent strict-serialization regressions; synthetic sources only."""
import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kkt_clf import s07, s07_report as report
from kkt_clf.config import TrainConfig
from kkt_clf.vectorizer import load_vectorizer_artifact
from tests.test_s07_integrity_independent import synthetic, _hash, _json


def _strict_json(path):
    def reject_constant(value):
        pytest.fail(f'Nonstandard JSON constant in {path}: {value}')
    return json.loads(path.read_text(), parse_constant=reject_constant)


@pytest.fixture
def periodic_training(synthetic):
    cfg, root = synthetic
    # Change and pin only this generated fixture's protocol before freezing a plan.
    path = Path(cfg.protocol)
    protocol = json.loads(path.read_text())
    protocol['resolved_sections']['train'].update(batch_size=1, scheduler_eval_every_steps=10)
    _json(path, protocol)
    cfg = dataclasses.replace(cfg, fixture_pins={**cfg.fixture_pins, 'protocol': _hash(path)})
    s07.write_plan(cfg, root)
    plan = s07.load_plan(root)
    run = plan['runs'][0]
    parts = s07.load_split(root, plan, run['split_id'])
    full = load_vectorizer_artifact(cfg.full_tfidf)
    return root, plan, run, parts, full.builder, plan['sources']['features']['sha256']


def test_constant_lr_periodic_trace_real_train_reload_inference_report(periodic_training, monkeypatch):
    root, plan, run, parts, builder, fid = periodic_training
    cfg = plan['training']['train']
    assert cfg['scheduler'] == 'none' and cfg['scheduler_eval_every_steps'] == 10
    expected_steps = len(parts['outer_train']) * cfg['epochs']
    assert cfg['batch_size'] == 1 and expected_steps >= 10
    captured = []
    original_train = s07.train_model

    def capture_real_training(*args, **kwargs):
        assert kwargs['restore_best'] is False
        result = original_train(*args, **kwargs)
        captured.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(s07, 'train_model', capture_real_training)
    device = torch.device('cpu')
    s07._train_one(root, plan, run, builder, fid, parts, device)
    history, best, raw_trace = captured[0]
    assert [t['step'] for t in raw_trace] == list(range(10, expected_steps + 1, 10))
    assert all(np.isnan(t['val_loss_subset']) for t in raw_trace)
    base = root / 'runs' / run['run_id']
    saved = s07.read_sealed(base / 'training.json')
    assert len(saved['history']) == cfg['epochs'] == 1  # Pinned fixture's frozen epoch rule.
    assert [row['epoch'] for row in history] == [1]
    assert saved['history'] == history and saved['best_monitor_diagnostic_only'] == best
    assert not best['restored_best']
    assert all(np.isfinite(row[key]) for row in history for key in ('train_loss', 'val_loss'))
    for raw, stored in zip(raw_trace, saved['lr_trace'], strict=True):
        assert stored == {**raw, 'val_loss_subset': None}
        assert stored['lr'] == cfg['lr']
    reloaded = s07._verify_checkpoint(root, plan, run, builder, fid, device)
    assert all(torch.isfinite(v).all() for v in reloaded.model.state_dict().values())
    receipt = s07.read_sealed(base / 'checkpoint_receipt.json')
    assert receipt['provenance'] == saved['provenance'] == s07.run_provenance(plan, run, fid)
    assert receipt['files'] == s07.tree(base / 'checkpoint')
    s07._evaluate_one(root, plan, run, builder, fid, parts, device)
    result = s07.read_sealed(base / 'result.json')
    assert result['status'] == 'success' and result['training_sha256'] == s07.sha(base / 'training.json')
    assert all(np.isfinite(v) for v in result['metrics'].values())
    predictions = pd.read_parquet(base / 'predictions_val.parquet')
    assert predictions.row_id.tolist() == parts['comparison_val'].row_id.tolist()
    summary = report.build_report(root)
    assert summary['statuses'] == {'success': 1, 'failed': 0, 'skipped': 0, 'pending': 35}
    manifest, tables = report.load_report_tables(root / 'report')
    assert not manifest['complete'] and len(tables['runs.csv']) == 36
    report.render_report(root / 'report', root / 'rebuilt.html')
    assert (root / 'rebuilt.html').is_file()
    for path in root.rglob('*.json'):
        _strict_json(path)


def _finite_record_inputs():
    return ({'run_id': 'synthetic'}, [{'epoch': 1, 'train_loss': .5, 'val_loss': .6,
                                     'val_l3_macro_f1': .25}],
            {'epoch': 1, 'score': .25, 'restored_best': False},
            [{'step': 10, 'epoch': 1, 'val_loss_subset': float('nan'), 'lr': .001}])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('field', ['train_loss', 'val_loss', 'val_l3_macro_f1', 'best_score',
                                  'trace_lr', 'trace_extra', 'provenance'])
def test_nonfinite_scientific_values_are_not_recursively_scrubbed(field, value):
    provenance, history, best, trace = _finite_record_inputs()
    if field in history[0]:
        history[0][field] = value
    elif field == 'best_score':
        best['score'] = value
    elif field == 'provenance':
        provenance['invalid'] = value
    else:
        trace[0]['lr' if field == 'trace_lr' else 'computed_metric'] = value
    with pytest.raises((ValueError, s07.S07Error)):
        s07._training_record(provenance, history, best, trace, TrainConfig(scheduler='none'))


@pytest.mark.parametrize('scheduler', ['none', 'cosine', 'onecycle', 'plateau'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), .375])
def test_only_explicitly_uncomputed_subset_loss_can_be_null(scheduler, value):
    args = _finite_record_inputs()
    args[3][0]['val_loss_subset'] = value
    original = copy.deepcopy(args)
    if np.isfinite(value) or (np.isnan(value) and scheduler != 'plateau'):
        result = s07._training_record(*args, TrainConfig(scheduler=scheduler))
        expected = None if np.isnan(value) else value
        assert result['lr_trace'][0]['val_loss_subset'] == expected
        assert result['history'] == original[1] and result['best_monitor_diagnostic_only'] == original[2]
        if np.isnan(value):
            assert np.isnan(args[3][0]['val_loss_subset'])
        else:
            assert args[3][0]['val_loss_subset'] == value
    else:
        with pytest.raises((ValueError, s07.S07Error)):
            s07._training_record(*args, TrainConfig(scheduler=scheduler))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_model_fails_before_checkpoint_receipt(periodic_training, monkeypatch, value):
    root, plan, run, parts, builder, fid = periodic_training

    def contaminated_training(model, *args, **kwargs):
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = value
        _, history, best, trace = _finite_record_inputs()
        return history, best, trace

    monkeypatch.setattr(s07, 'train_model', contaminated_training)
    with pytest.raises(s07.S07Error, match='[Nn]onfinite'):
        s07._train_one(root, plan, run, builder, fid, parts, torch.device('cpu'))
    base = root / 'runs' / run['run_id']
    assert not (base / 'checkpoint_receipt.json').exists()
    assert not (base / 'checkpoint').exists()
    assert not (base / 'result.json').exists()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('operation', [s07.digest, s07.sealed])
def test_global_hashing_and_sealing_remain_strict(operation, value):
    with pytest.raises(ValueError):
        operation({'nested': [{'scientific_metric': value}]})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('field', ['train_loss', 'val_loss'])
def test_nonfinite_history_fails_before_successful_checkpoint(periodic_training, monkeypatch, field, value):
    root, plan, run, parts, builder, fid = periodic_training

    def invalid_loss_training(*args, **kwargs):
        _, history, best, trace = _finite_record_inputs()
        history[0][field] = value
        return history, best, trace

    monkeypatch.setattr(s07, 'train_model', invalid_loss_training)
    with pytest.raises((ValueError, s07.S07Error)):
        s07._train_one(root, plan, run, builder, fid, parts, torch.device('cpu'))
    base = root / 'runs' / run['run_id']
    assert not (base / 'checkpoint_receipt.json').exists()
    assert not (base / 'checkpoint').exists()
    assert not (base / 'result.json').exists()
