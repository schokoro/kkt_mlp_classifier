"""Regression checks for serialized model sizes in the S06 report."""
import json

import pandas as pd
import pytest

from kkt_clf import s06_report as report
from tests.test_s06_report_hash_independent import saved, _sha, _write_json  # noqa: F401


@pytest.fixture
def model_size_inputs(saved):
    root, _, baseline_pred, baseline_result_path = saved
    plan_path = root / 'runs' / 'planned.json'
    plan = json.loads(plan_path.read_text())
    selected = dict(run_id='global_k8_seed42', config_slug='global_k8', method='global',
                    budget=8, seed=42, status='pending', reason=None)
    plan['runs'].append(selected)
    plan['n_planned'] = len(plan['runs'])
    _write_json(plan_path, plan)

    baseline = json.loads(baseline_result_path.read_text())
    for result, result_path, payload in [
        (baseline, baseline_result_path, b'baseline weights'),
        ({**selected, 'schema_version': report.RUN_RESULT_SCHEMA, 'status': 'success'},
         root / 'runs' / 'global_k8' / 'run_result_seed_42.json', b'selected weights are different'),
    ]:
        checkpoint = root / 'checkpoints' / result['run_id']
        checkpoint.mkdir(parents=True)
        model = checkpoint / 'model.pt'
        model.write_bytes(payload)
        (checkpoint / 'unrelated.bin').write_bytes(b'x' * 113)
        pred = root / 'runs' / 'predictions' / (result['run_id'] + '.parquet')
        if result['run_id'] != 'none_seed42':
            frame = pd.read_parquet(baseline_pred)
            frame['run_id'] = result['run_id']
            frame.to_parquet(pred, index=False)
        result['hashes'] = {'model_pt_sha256': _sha(model), 'tfidf_features_sha256': 'features',
                            'predictions_sha256': _sha(pred)}
        result['artifacts'] = {'checkpoint_dir': str(checkpoint), 'tfidf_artifact': 'tfidf',
                               'selection_artifact': 'selection'}
        result['comparison_val_eval'] = {'predictions_path': str(pred), 'l3_macro_f1': 1 / 3,
                                         'l3_accuracy': .5}
        _write_json(result_path, result)

    stages = pd.DataFrame([
        {'run_id': 'none_seed42', 'stage': 'mlp_train', 'status': 'reused', 'artefact_bytes': None},
        {'run_id': 'global_k8_seed42', 'stage': 'mlp_train', 'status': 'success', 'artefact_bytes': None},
        {'run_id': 'none_seed43', 'stage': 'mlp_train', 'status': 'failed', 'artefact_bytes': None},
    ])
    stages.to_csv(root / 'runs' / 'stages.csv', index=False)
    return root


@pytest.mark.parametrize('run_id', ['none_seed42', 'global_k8_seed42'])
def test_report_mlp_stages_show_serialized_model_size_for_selected_and_baseline(model_size_inputs, run_id):
    root = model_size_inputs
    report.build_s06_report(str(root))
    stages = pd.read_csv(root / 'report' / 'stages.csv').set_index('run_id')
    expected = (root / 'checkpoints' / run_id / 'model.pt').stat().st_size
    assert stages.loc[run_id, 'artefact_bytes'] == expected


def test_failed_mlp_stage_has_no_fabricated_size(model_size_inputs):
    root = model_size_inputs
    report.build_s06_report(str(root))
    stages = pd.read_csv(root / 'report' / 'stages.csv').set_index('run_id')
    assert pd.isna(stages.loc['none_seed43', 'artefact_bytes'])
