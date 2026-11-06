"""Independent S05 report contract checks using disposable saved results only."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from kkt_clf import s05_report as report


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / 'input'
    root.mkdir()
    rows = []
    for run_id, config, seed, status, f1, acc, reason in [
        ('seed_42', 'A', 42, 'success', .4, .5, None),
        ('seed_43', 'A', 43, 'success', .8, .9, None),
        ('seed_44', 'B', 44, 'success', .2, .3, None),
        ('seed_45', 'A', 45, 'failed', None, None, 'out of memory'),
    ]:
        row = dict.fromkeys(report.RUNS_REQUIRED)
        row.update(run_id=run_id, config_id=config, train_seed=seed, split_seed=7,
                   status=status, reason=reason, l3_macro_f1=f1, l3_accuracy=acc,
                   metric_evaluated_on='comparison_val')
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / 'runs.csv', index=False)
    pd.DataFrame([
        ('seed_42', 1, 'train', .8, .3, .4, .01),
        ('seed_42', 1, 'monitor', .9, .2, .3, .01),
        ('seed_42', 2, 'train', .4, .6, .7, .01),
        ('seed_42', 2, 'monitor', .7, .5, .6, .01),
    ], columns=report.HISTORY_REQUIRED).to_csv(root / 'history.csv', index=False)
    pd.DataFrame(columns=report.STAGES_REQUIRED).to_csv(root / 'stages.csv', index=False)
    pd.DataFrame([
        ('seed_42', 'known', 3, .8, .6, .7),
        ('seed_42', 'unsupported', 0, None, None, None),
    ], columns=report.PER_CLASS_REQUIRED).to_parquet(root / 'per_class.parquet', index=False)
    (root / 'manifest.json').write_text(json.dumps({'seeds': {'planned': [42, 43, 44, 45, 46]}}))
    return root


def build(saved, tmp_path):
    return report.build_s05_report(str(saved), str(tmp_path / 'report'))


def test_saved_results_cli_and_repeat_render_without_training(saved, tmp_path, monkeypatch):
    import kkt_clf.s05_baseline as baseline
    for name in ('run_selection', 'run_refit'):
        if hasattr(baseline, name):
            monkeypatch.setattr(baseline, name, lambda *a, **k: pytest.fail('training invoked'))
    result = build(saved, tmp_path)
    first = Path(result['report_html']).read_text()
    assert all((tmp_path / 'report' / name).is_file() for name in report.REPORT_FILES)
    assert '<svg' in first and 'seed_42' in first
    assert 'cdn' not in first.lower()
    cli_dir = tmp_path / 'cli_report'
    cli = subprocess.run([sys.executable, 'scripts/run_s05_report.py', '--output', str(saved),
                          '--report-dir', str(cli_dir)], capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    assert 'seed_42' in (cli_dir / 'report.html').read_text()
    # The exported report tables themselves must be sufficient for another render.
    restored = tmp_path / 'restored'
    again = subprocess.run([sys.executable, 'scripts/run_s05_report.py', '--output', str(tmp_path / 'report'),
                            '--report-dir', str(restored)], capture_output=True, text=True)
    assert again.returncode == 0, again.stderr
    assert 'seed_42' in (restored / 'report.html').read_text()


def test_config_statistics_gap_and_failed_skipped_null(saved, tmp_path):
    build(saved, tmp_path)
    out = tmp_path / 'report'
    manifest = json.loads((out / 'manifest.json').read_text())
    stats = {x['config_id']: x for x in manifest['seed_stats_comparison_val']}
    assert set(stats) == {'A', 'B'}
    for metric, mean in [('l3_macro_f1', .6), ('l3_accuracy', .7)]:
        assert stats['A'][metric]['mean'] == pytest.approx(mean)
        assert stats['A'][metric]['std'] == pytest.approx(.2)
        assert stats['A'][metric]['min'] == pytest.approx(mean - .2)
        assert stats['A'][metric]['max'] == pytest.approx(mean + .2)
        assert stats['A'][metric]['range'] == pytest.approx(.4)
    assert stats['B']['l3_macro_f1']['mean'] == pytest.approx(.2)
    runs = pd.read_csv(out / 'runs.csv').set_index('run_id')
    for rid, status in [('seed_45', 'failed'), ('seed_46', 'skipped')]:
        assert runs.loc[rid, 'status'] == status
        assert isinstance(runs.loc[rid, 'reason'], str) and runs.loc[rid, 'reason']
        assert pd.isna(runs.loc[rid, 'l3_macro_f1'])
        assert pd.isna(runs.loc[rid, 'l3_accuracy'])
    gap = report._train_monitor_gap(pd.read_csv(saved / 'history.csv'), pd.read_csv(saved / 'runs.csv'))
    assert gap.iloc[0]['gap_monitor_minus_train'] == pytest.approx(.3)
    assert gap.iloc[0]['final_monitor_l3_macro_f1'] == pytest.approx(.5)
    html = (out / 'report.html').read_text()
    assert 'out of memory' in html and '0.3000' in html


def test_unsupported_class_does_not_acquire_metrics(saved, tmp_path):
    build(saved, tmp_path)
    pc = pd.read_parquet(tmp_path / 'report/per_class.parquet').set_index('L3')
    assert pc.loc['unsupported', 'support'] == 0
    assert pc.loc['unsupported', ['precision', 'recall', 'f1']].isna().all()
    html = (tmp_path / 'report/report.html').read_text()
    assert 'unsupported' in html


@pytest.mark.parametrize('filename,contents', [
    ('runs.csv', ''), ('history.csv', 'not,a,valid,table\n1,2,3,4\n'),
    ('stages.csv', ''), ('per_class.parquet', 'not parquet'),
])
def test_damaged_required_tables_fail_cleanly_without_partial_output(saved, tmp_path, filename, contents):
    (saved / filename).write_text(contents)
    with pytest.raises(report.S05ReportError):
        build(saved, tmp_path)
    assert not (tmp_path / 'report').exists()


def test_empty_well_formed_tables_produce_diagnostic_report(saved, tmp_path):
    for filename, columns in [('runs.csv', report.RUNS_REQUIRED), ('history.csv', report.HISTORY_REQUIRED),
                              ('stages.csv', report.STAGES_REQUIRED)]:
        pd.DataFrame(columns=columns).to_csv(saved / filename, index=False)
    pd.DataFrame(columns=report.PER_CLASS_REQUIRED).to_parquet(saved / 'per_class.parquet', index=False)
    build(saved, tmp_path)
    manifest = json.loads((tmp_path / 'report/manifest.json').read_text())
    assert manifest['gate']['decision'] != 'ready'


def test_test_metrics_cannot_be_relabelled_as_comparison_validation(saved, tmp_path):
    runs = pd.read_csv(saved / 'runs.csv')
    runs.loc[0, 'metric_evaluated_on'] = 'sealed_test'
    runs.loc[0, 'l3_macro_f1'] = .1234567
    runs.to_csv(saved / 'runs.csv', index=False)
    try:
        build(saved, tmp_path)
    except report.S05ReportError:
        return
    html = (tmp_path / 'report/report.html').read_text()
    exported = pd.read_csv(tmp_path / 'report/runs.csv')
    assert not ((exported['metric_evaluated_on'] == 'sealed_test') & exported['l3_macro_f1'].notna()).any()
    assert '0.1235' not in html


def test_gate_does_not_claim_ready_without_learning_evidence(saved, tmp_path):
    runs = pd.read_csv(saved / 'runs.csv').iloc[:3].copy()
    runs['config_id'] = 'A'
    runs.to_csv(saved / 'runs.csv', index=False)
    (saved / 'manifest.json').write_text(json.dumps({'seeds': {'planned': [42, 43, 44]}}))
    pd.DataFrame(columns=report.HISTORY_REQUIRED).to_csv(saved / 'history.csv', index=False)
    (saved / 'audit').mkdir()
    pd.DataFrame({'split': ['train']}).to_csv(saved / 'audit/split_audit_corrected.csv', index=False)
    (saved / 'majority_baseline.json').write_text(json.dumps({'l3_macro_f1': .1}))
    (saved / 'tfidf_verification.json').write_text(json.dumps({'decision': 'ok'}))
    (saved / 'selected_checkpoint.json').write_text(json.dumps({'roundtrip': {'status': 'success'}}))
    build(saved, tmp_path)
    gate = json.loads((tmp_path / 'report/manifest.json').read_text())['gate']
    assert gate['decision'] != 'ready', 'empty history cannot demonstrate learning or train-monitor gap'
