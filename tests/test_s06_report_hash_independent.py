import hashlib
import json
import subprocess

import pandas as pd
import pytest

from kkt_clf import s06_report as report


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved(tmp_path, monkeypatch):
    # Report metadata normally probes Git; this fixture must not run Git.
    def no_subprocess(*args, **kwargs):
        raise OSError('subprocess disabled in isolated test')
    monkeypatch.setattr(subprocess, 'run', no_subprocess)
    root = tmp_path / 'experiment'
    runs = [dict(run_id=f'none_seed{seed}', config_slug='none', method='none',
                 budget=None, seed=seed, status='pending', reason=None)
            for seed in (42, 43, 44)]
    val = root / 'splits' / 'comparison_val.parquet'
    val.parent.mkdir(parents=True)
    pd.DataFrame({'row_id': [10, 20], 'L3': ['a', 'b']}).to_parquet(val, index=False)
    part = {'file': str(val), 'sha256': _sha(val), 'bytes': val.stat().st_size, 'n_rows': 2}
    fingerprint = {key: part[key] for key in ('sha256', 'bytes', 'n_rows')}
    maps = {'L1': ['top'], 'L2': ['middle'], 'L3': ['a', 'b']}
    split_manifest = root / 'splits' / 'manifest.json'
    _write_json(split_manifest, {'parts': {'comparison_val': part}, 'label_maps': maps})
    protocol = root / 'shared' / 'training_protocol.json'
    _write_json(protocol, {'splits': {'comparison_val': fingerprint}})
    plan = {'runs': runs, 'n_planned': len(runs), 'protocol_sha256': _sha(protocol), 'config': {
        'splits_manifest': str(split_manifest), 'seeds': [42, 43, 44],
        'reference_seed': 42, 'methods': [], 'budgets': [], 'char_share': 0.5}}
    _write_json(root / 'runs' / 'planned.json', plan)
    maps_sha = hashlib.sha256(json.dumps(
        maps, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
    _write_json(root / 'runs' / 'baseline_full' / 'manifest.json', {
        'label_maps': {'l3_universe_sha256': maps_sha,
                       'n_classes': {'L1': 1, 'L2': 1, 'L3': 2}},
        'inputs': {'splits': {'comparison_val': fingerprint}}})
    pred = root / 'runs' / 'predictions' / 'none_seed42.parquet'
    pred.parent.mkdir(parents=True)
    pd.DataFrame({'row_id': [10, 20], 'run_id': ['none_seed42'] * 2,
                  'true_L3': ['a', 'b'], 'predicted_L3': ['a', 'a'],
                  'correct': [True, False]}).to_parquet(pred, index=False)
    result_path = root / 'runs' / 'none' / 'run_result_seed_42.json'
    result = {**runs[0], 'schema_version': report.RUN_RESULT_SCHEMA, 'status': 'success',
              'hashes': {'model_pt_sha256': 'model', 'tfidf_features_sha256': 'features',
                         'predictions_sha256': _sha(pred)},
              'artifacts': {'checkpoint_dir': 'checkpoint', 'tfidf_artifact': 'tfidf'},
              'comparison_val_eval': {'predictions_path': str(pred), 'l3_macro_f1': 1 / 3,
                                      'l3_accuracy': 0.5}}
    _write_json(result_path, result)
    _write_json(root / 'runs' / 'none' / 'run_result_seed_43.json',
                {'status': 'failed', 'reason': 'fixture failure'})
    return root, val, pred, result_path


def test_verified_success_and_plan_status_alignment(saved):
    root, _, _, _ = saved
    info = report.build_s06_report(str(root))
    assert info['n_runs'] == 3
    assert info['statuses'] == {'success': 1, 'failed': 1, 'pending': 1}
    statuses = pd.read_csv(root / 'report' / 'runs.csv').set_index('run_id')['status'].to_dict()
    assert statuses == {'none_seed42': 'success', 'none_seed43': 'failed', 'none_seed44': 'pending'}
    manifest = json.loads((root / 'report' / 'manifest.json').read_text())
    assert [manifest['statuses'][k] for k in ('n_planned', 'n_success', 'n_failed', 'n_pending')] == [3, 1, 1, 1]
    predictions = pd.read_parquet(root / 'report' / 'predictions_val.parquet')
    assert predictions['run_id'].tolist() == ['none_seed42', 'none_seed42']
    assert predictions['row_id'].tolist() == [10, 20]


@pytest.mark.parametrize(('change', 'message'), [
    ('missing_hash', 'fingerprint'), ('wrong_hash', 'sha256 предсказаний'),
    ('row_id', 'row_id'), ('true_L3', 'true_L3'), ('run_id', 'чужого run_id'),
    ('split', 'fingerprint из manifest'), ('result_run_id', 'полю.*run_id'),
])
def test_report_rejects_unverified_success(saved, change, message):
    root, val, pred, result_path = saved
    result = json.loads(result_path.read_text())
    if change == 'missing_hash':
        del result['hashes']['predictions_sha256']
    elif change == 'wrong_hash':
        result['hashes']['predictions_sha256'] = '0' * 64
    elif change == 'result_run_id':
        result['run_id'] = 'foreign'
    elif change == 'split':
        pd.DataFrame({'row_id': [10, 21], 'L3': ['a', 'b']}).to_parquet(val, index=False)
    else:
        frame = pd.read_parquet(pred)
        frame.loc[0, change] = {'row_id': 999, 'true_L3': 'b', 'run_id': 'foreign'}[change]
        frame.to_parquet(pred, index=False)
        result['hashes']['predictions_sha256'] = _sha(pred)
    _write_json(result_path, result)
    with pytest.raises(report.S06ReportError, match=message):
        report.build_s06_report(str(root))
    assert not (root / 'report').exists()
