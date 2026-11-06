"""Regression checks for S04 comparison identity and shared resource costs."""
import json

import pandas as pd
import pytest

from kkt_clf import experiment as exp
from tests.test_s03_experiment_contract import saved_experiment  # noqa: F401


IDENTITY = {
    'val_sha256': 'val', 'label_map_sha256': 'labels', 'train_sha256': 'train',
    'full_tfidf_sha256': 'tfidf', 'model_config_sha256': 'model',
    'eval_protocol_sha256': 'protocol', 'seed': 42,
}


def _write_fixture_identities(path, schema, missing=None):
    meta_path = path / 'experiment.json'
    meta = json.loads(meta_path.read_text())
    meta['schema_version'] = schema
    meta_path.write_text(json.dumps(meta))
    for run in ('none', 'global_50000'):
        result_path = path / 'runs' / run / 'run_result.json'
        result = json.loads(result_path.read_text())
        result.update(IDENTITY)
        if missing:
            result.pop(missing)
        result_path.write_text(json.dumps(result))


@pytest.mark.parametrize('missing', list(IDENTITY))
def test_s04_missing_identity_on_both_sides_disables_deltas(saved_experiment, missing):
    _write_fixture_identities(saved_experiment, 's04-experiment@1', missing)
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert not bool(runs.loc['global_50000', 'comparable_with_baseline'])
    assert pd.isna(runs.loc['global_50000', 'delta_macro_f1'])
    assert pd.isna(runs.loc['global_50000', 'delta_accuracy'])
    pc = pd.read_parquet(paths['per_class_parquet']).query("run_id == 'global_50000'")
    assert pc[['baseline_f1', 'baseline_recall', 'delta_f1', 'delta_recall']].isna().all().all()


def test_explicit_s03_legacy_keeps_val_and_label_comparison(saved_experiment):
    _write_fixture_identities(saved_experiment, 's03-experiment@1')
    for run in ('none', 'global_50000'):
        path = saved_experiment / 'runs' / run / 'run_result.json'
        result = json.loads(path.read_text())
        for key in ('train_sha256', 'full_tfidf_sha256', 'model_config_sha256', 'eval_protocol_sha256'):
            result.pop(key)
        path.write_text(json.dumps(result))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert runs.loc['global_50000', 'delta_macro_f1'] == pytest.approx(.25)


def test_saved_tables_without_comparability_do_not_plot_s04_runs(saved_experiment, monkeypatch):
    _write_fixture_identities(saved_experiment, 's04-experiment@1')
    paths = exp.build_report(str(saved_experiment))
    runs_path = saved_experiment / 'report/runs.csv'
    runs = pd.read_csv(runs_path).drop(columns='comparable_with_baseline')
    runs.to_csv(runs_path, index=False)
    manifest_path = saved_experiment / 'report/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for run in manifest['planned_runs']:
        run.pop('comparable_with_baseline', None)
    manifest_path.write_text(json.dumps(manifest))
    (saved_experiment / 'experiment.json').unlink()
    seen = []
    original = exp._svg_line_chart
    def capture(title, ks, series, **kwargs):
        seen.extend(series)
        return original(title, ks, series, **kwargs)
    monkeypatch.setattr(exp, '_svg_line_chart', capture)
    exp.build_report(str(saved_experiment))
    html = open(paths['report_html'], encoding='utf8').read()
    assert not seen
    assert 'нет данных о сопоставимости' in html


def test_resource_costs_charge_shared_only_to_gxi_and_once_in_total():
    runs = pd.DataFrame([
        {'run_id': name, 'method': method, 'requested_k': k, 'l3_macro_f1': .5}
        for name, method, k in [('none', 'none', None), ('global_50', 'global', 50),
                                ('class_50', 'class_aware', 50),
                                ('g1', 'gradient_x_input', 50), ('g2', 'gradient_x_input', 100),
                                ('g3', 'gradient_x_input', 200)]])
    stages = pd.DataFrame([{'run_id': run, 'status': 'success', 'seconds': seconds}
                           for run, seconds in [('none', 1), ('global_50', 2), ('class_50', 3),
                                                ('g1', 4), ('g2', 5), ('g3', 6),
                                                ('shared_attribution', 11), ('shared_attribution', 13)]])
    rows, totals = exp._scenario_cost_rows(stages, runs)
    rows = {r['run_id']: r for r in rows}
    totals = {r['method']: r for r in totals}
    for run, seconds in [('none', 1), ('global_50', 2), ('class_50', 3)]:
        assert rows[run]['shared_seconds'] is None
        assert rows[run]['scenario_total_seconds'] == seconds
    assert [rows[k]['scenario_total_seconds'] for k in ('g1', 'g2', 'g3')] == [28, 29, 30]
    assert totals['gradient_x_input']['n_budgets'] == 3
    assert totals['gradient_x_input']['scenario_total_seconds'] == 4 + 5 + 6 + 11 + 13


def test_resource_costs_unknown_shared_or_budget_time_stays_null():
    runs = pd.DataFrame([{'run_id': f'g{i}', 'method': 'gradient_x_input',
                          'requested_k': i, 'l3_macro_f1': .5} for i in (1, 2, 3)])
    stages = pd.DataFrame([{'run_id': run, 'status': 'success', 'seconds': seconds}
                           for run, seconds in [('g1', 1), ('g2', 2), ('g3', None),
                                                ('shared_attribution', None)]])
    rows, totals = exp._scenario_cost_rows(stages, runs)
    assert all(row['scenario_total_seconds'] is None for row in rows)
    assert totals[0]['scenario_total_seconds'] is None


@pytest.mark.parametrize('unknown_run', ['g1', 'shared_attribution'])
def test_partially_unknown_stage_time_does_not_become_partial_total(unknown_run):
    runs = pd.DataFrame([{'run_id': 'g1', 'method': 'gradient_x_input',
                          'requested_k': 50, 'l3_macro_f1': .5}])
    stages = pd.DataFrame([
        {'run_id': 'g1', 'status': 'success', 'seconds': 4},
        {'run_id': 'shared_attribution', 'status': 'success', 'seconds': 11},
        {'run_id': unknown_run, 'status': 'success', 'seconds': None},
    ])
    rows, totals = exp._scenario_cost_rows(stages, runs)
    assert rows[0]['scenario_total_seconds'] is None
    assert totals[0]['scenario_total_seconds'] is None
