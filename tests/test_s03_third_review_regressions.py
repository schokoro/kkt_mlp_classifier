"""Regression checks for damaged saved stages and unavailable RSS samples."""
import builtins
import json
import io
import threading

import pandas as pd
import pytest
import torch

from kkt_clf import experiment as exp
from tests.test_s03_experiment_contract import saved_experiment


@pytest.mark.parametrize('damaged', [42, 'broken', [None, 42, 'broken', {}, {'stage': 17}]])
def test_full_report_survives_corrupt_saved_stages(saved_experiment, damaged):
    path = saved_experiment / 'runs/class_aware_50000/run_result.json'
    result = json.loads(path.read_text(encoding='utf8'))
    result['stages'] = damaged
    path.write_text(json.dumps(result), encoding='utf8')
    original = {p: p.read_bytes() for p in saved_experiment.rglob('run_result.json')}

    paths = exp.build_report(str(saved_experiment))

    assert all(p.read_bytes() == content for p, content in original.items())
    assert all((saved_experiment / 'report' / name).is_file() for name in
               ('report.html', 'runs.csv', 'per_class.parquet', 'stages.csv', 'manifest.json'))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert len(runs) == 7
    assert runs.loc['none', 'l3_macro_f1'] == pytest.approx(.5)
    assert runs.loc['global_50000', 'l3_macro_f1'] == pytest.approx(.75)
    assert runs.loc['class_aware_50000', 'status'] == 'failed'
    stages = pd.read_csv(paths['stages_csv'])
    assert set(stages.query("run_id == 'class_aware_50000'").stage) >= {
        'selection', 'selected_tfidf_refit', 'mlp_train', 'validation_inference'}
    assert 'class_aware_50000' in (saved_experiment / 'report/report.html').read_text(encoding='utf8')


def _explains_missing_rss(note):
    note = note.lower()
    return 'rss' in note and any(word in note for word in (
        'недоступ', 'отсутств', 'не удалось', 'пропущ', 'частич',
        'unavailable', 'missing', 'failed', 'partial'))


def test_unavailable_rss_is_null_and_explained(monkeypatch):
    original_open = builtins.open
    def unavailable(path, *args, **kwargs):
        if str(path) == '/proc/self/status':
            raise OSError('RSS unavailable')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', unavailable)
    with exp._Measurement(torch.device('cpu'), 'test') as measurement:
        pass
    row = measurement.stage_row('run', 'mlp_train', 'success')
    assert row['peak_rss_bytes'] is None
    assert _explains_missing_rss(row['measurement_note'])


def test_partially_unavailable_rss_preserves_samples_and_explains_gap(monkeypatch):
    sampled_gap = threading.Event()
    original_open = builtins.open
    calls = 0
    def open_status(path, *args, **kwargs):
        nonlocal calls
        if str(path) != '/proc/self/status':
            return original_open(path, *args, **kwargs)
        calls += 1
        if calls == 1:
            return io.StringIO('VmRSS:\t123 kB\n')
        sampled_gap.set()
        raise OSError('RSS temporarily unavailable')
    monkeypatch.setattr(builtins, 'open', open_status)
    with exp._Measurement(torch.device('cpu'), 'test', rss_interval_s=.001) as measurement:
        assert sampled_gap.wait(2), 'RSS sampler did not run'
    row = measurement.stage_row('run', 'mlp_train', 'success')
    assert row['peak_rss_bytes'] == 123 * 1024
    assert _explains_missing_rss(row['measurement_note'])
