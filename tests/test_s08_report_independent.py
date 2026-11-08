"""S08 blocked-state honesty and isolated deterministic table-only rebuild."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from kkt_clf import s08, s08_report as report
from tests.test_s08_integrity_independent import synthetic, forbid, guard_closed_reads
from tests.test_s08_fixtures import sources, file_sha, write_json, reseal
from tests.test_s08_smoke_independent import executed


TEST_ONLY_FILES = {'paired_test.csv', 'per_class.parquet', 'support_bins.csv',
                   'predictions_test.parquet', 'confusion_pairs.csv'}


def test_pending_report_has_honest_nulls_and_no_test_files(synthetic, monkeypatch):
    cfg, root, src = synthetic
    s08.write_plan(cfg, root)
    guard_closed_reads(monkeypatch, src['parts']['sealed_test']['file'])
    for name in ('train_model', 'select_features', 'refit_selected_vectorizer'):
        monkeypatch.setattr(s08, name, forbid)
    result = report.build_report(root)
    assert result['gate'] == 'awaiting_user_approval'
    assert result['training_complete'] is result['confirmatory_complete'] is False
    expected = {'report.html', 'manifest.json', 'runs.csv', 'stages.csv', 'ni_decision.json'}
    assert {p.name for p in (root / 'report').iterdir()} == expected
    assert not any(p.name in TEST_ONLY_FILES for p in root.rglob('*'))
    runs = pd.read_csv(root / 'report' / 'runs.csv')
    assert len(runs) == 9 and (runs.status == 'pending').all()
    assert runs[['validation_macro_f1', 'test_macro_f1', 'model_bytes']].isna().all().all()
    decision = json.loads((root / 'report' / 'ni_decision.json').read_text())
    for name in ('delta_NI', 'delta', 'lower_bound', 'decision'):
        assert decision[name] is None
    assert decision['status'] == 'not_run_awaiting_approval'
    assert decision['margin_status'] == decision['primary_status'] == 'unapproved'
    assert decision['sealed_inference'] is False
    manifest = json.loads((root / 'report' / 'manifest.json').read_text())
    assert manifest['approval'] is None and manifest['confirmatory_complete'] is False
    feasibility = manifest['feasibility']
    assert feasibility['exploratory'] is True and feasibility['sealed_quality_used'] is False
    assert feasibility['delta_NI'] is None
    assert len(feasibility['scenarios']) >= 1
    for scenario in feasibility['scenarios']:
        assert scenario['recipe']['B'] >= 5000
        assert isinstance(scenario['lower_bound'], float)


def test_html_rebuild_is_byte_identical_in_isolated_tables_only_directory(synthetic, tmp_path, monkeypatch):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    report.build_report(root)
    original_html = (root / 'report' / 'report.html').read_bytes()
    isolated = tmp_path / 'tables_only'
    isolated.mkdir()
    for name in ('manifest.json', 'runs.csv', 'stages.csv', 'ni_decision.json'):
        shutil.copyfile(root / 'report' / name, isolated / name)
    for name in ('load_plan', 'verify_checkpoint', 'verify_selected', 'verify_training_freeze',
                 'load_vectorizer_artifact'):
        monkeypatch.setattr(report, name, forbid)
    path_open, builtin_open = Path.open, builtins.open
    def check_path(path):
        if not isinstance(path, int):
            assert Path(path).resolve().is_relative_to(isolated.resolve()), f'External rebuild access: {path}'
    def guarded_path_open(path, *args, **kwargs):
        check_path(path)
        return path_open(path, *args, **kwargs)
    def guarded_builtin_open(path, *args, **kwargs):
        check_path(path)
        return builtin_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_path_open)
    monkeypatch.setattr(builtins, 'open', guarded_builtin_open)
    report.render_report(isolated, isolated / 'rebuilt.html')
    report.render_report(isolated, isolated / 'rebuilt_again.html')
    assert (isolated / 'rebuilt.html').read_bytes() == original_html
    assert (isolated / 'rebuilt_again.html').read_bytes() == original_html
    html = original_html.decode()
    assert 'Holdout CLOSED' in html and 'UNAPPROVED' in html
    assert 'src="http' not in html and 'href="http' not in html


@pytest.mark.parametrize('name', ['runs.csv', 'stages.csv', 'ni_decision.json'])
def test_table_tamper_is_rejected_on_rebuild(synthetic, name):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    report.build_report(root)
    with (root / 'report' / name).open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises((RuntimeError, ValueError)):
        report.render_report(root / 'report', root / 'tampered.html')


@pytest.mark.parametrize('damage', ['fake_approval', 'fake_ni', 'fake_metric', 'fake_complete', 'fake_status'])
def test_semantic_pending_report_corruption_rejected_even_after_resealing(synthetic, damage):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    report.build_report(root)
    out = root / 'report'
    manifest = json.loads((out / 'manifest.json').read_text())
    if damage == 'fake_approval':
        manifest['approval'] = {'approved': True}
    elif damage == 'fake_complete':
        manifest['confirmatory_complete'] = True
    elif damage == 'fake_ni':
        decision = json.loads((out / 'ni_decision.json').read_text())
        decision.update(delta_NI=.01, margin_status='approved', status='non_inferior')
        write_json(out / 'ni_decision.json', reseal(decision))
        manifest['tables']['ni_decision.json'] = s08.fingerprint(out / 'ni_decision.json')
    else:
        runs = pd.read_csv(out / 'runs.csv')
        if damage == 'fake_metric':
            runs.loc[0, 'test_macro_f1'] = .99
        else:
            runs.loc[0, 'status'] = 'success'
        runs.to_csv(out / 'runs.csv', index=False)
        manifest['tables']['runs.csv'] = s08.fingerprint(out / 'runs.csv')
    write_json(out / 'manifest.json', reseal(manifest))
    with pytest.raises((RuntimeError, ValueError)):
        report.render_report(out, root / 'misleading.html')


def test_failed_training_remains_failed_and_not_success(synthetic, monkeypatch):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    def fail_training(*args, **kwargs):
        raise RuntimeError('independent synthetic training interruption')
    monkeypatch.setattr(s08, 'train_model', fail_training)
    outcome = s08.execute(root)
    assert len(outcome['failed']) == 1 and len(outcome['pending']) == 8
    assert not (root / 'training_freeze.json').exists()
    result = report.build_report(root)
    assert result['statuses']['failed'] == 1 and result['statuses']['success'] == 0
    runs = pd.read_csv(root / 'report' / 'runs.csv')
    assert runs.test_macro_f1.isna().all() and runs.validation_macro_f1.isna().all()
    assert 'independent synthetic training interruption' in runs.loc[runs.status == 'failed', 'reason'].iloc[0]


def test_unfinished_stale_success_cannot_publish(synthetic):
    cfg, root, _ = synthetic
    plan = s08.write_plan(cfg, root)
    path = root / 'runs' / plan['runs'][0]['run_id'] / 'result.json'
    path.parent.mkdir(parents=True)
    write_json(path, reseal({'status': 'success'}))
    with pytest.raises((RuntimeError, ValueError)):
        report.build_report(root)


def test_completed_training_is_still_pending_ni_and_shared_refits_charged_once(executed):
    _, root, _, _ = executed
    result = report.build_report(root)
    assert result['training_complete'] is True and result['confirmatory_complete'] is False
    assert result['statuses']['success'] == 9 and result['gate'] == 'awaiting_user_approval'
    stages = pd.read_csv(root / 'report' / 'stages.csv')
    refits = stages[(stages.stage == 'refit') & (stages.scope == 'shared')]
    assert len(refits) == 2 and set(refits.method) == {'class_aware', 'global'}
    assert (refits.charge_count == 1).all()
    assert (stages[stages.stage == 'train'].scope == 'run').all()
    assert not set(p.name for p in (root / 'report').iterdir()) & TEST_ONLY_FILES


def test_cli_rebuild_with_only_four_report_inputs(synthetic, tmp_path):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    report.build_report(root)
    isolated = tmp_path / 'cli_tables_only'
    isolated.mkdir()
    for name in ('manifest.json', 'runs.csv', 'stages.csv', 'ni_decision.json'):
        shutil.copyfile(root / 'report' / name, isolated / name)
    completed = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts' / 'run_s08.py'),
        'rebuild', '--report-dir', str(isolated), '--output-html', str(isolated / 'cli.html')],
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert (isolated / 'cli.html').read_bytes() == (root / 'report' / 'report.html').read_bytes()
