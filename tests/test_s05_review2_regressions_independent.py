"""Review 2 regressions, using disposable synthetic data only."""
import dataclasses
from pathlib import Path

import pandas as pd
import pytest
import torch

from kkt_clf import s05_baseline as baseline, s05_report as report
from tests.test_s05_baseline_independent import synthetic
from tests.test_s05_report_independent import saved


def test_selection_loss_is_reused_or_mismatch_rejected(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    cfg.fixed_epochs = None
    cfg.loss = {'class_weight_alpha': 0.17}
    selection = baseline.run_selection(cfg, manifest, loaded, torch.device('cpu'))
    cfg.loss = {'class_weight_alpha': 0.83}
    seen = []
    original = baseline.train_model
    def capture(*args, **kwargs):
        seen.append(dataclasses.asdict(args[5]))
        return original(*args, **kwargs)
    monkeypatch.setattr(baseline, 'train_model', capture)
    try:
        baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'), selection=selection,
                           tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    except (baseline.S05BaselineError, ValueError):
        assert not seen, 'loss mismatch must be rejected before training'
    else:
        expected = selection['candidates'][0]['loss']
        assert seen and all(loss == expected for loss in seen)


def test_plateau_selection_refit_schedule_is_fixed_and_identical(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    cfg.fixed_epochs = None
    cfg.train['scheduler'] = 'plateau'
    seen = []
    original = baseline.train_model
    def capture(*args, **kwargs):
        seen.append((args[6].scheduler, args[6].lr))
        return original(*args, **kwargs)
    monkeypatch.setattr(baseline, 'train_model', capture)
    try:
        selection = baseline.run_selection(cfg, manifest, loaded, torch.device('cpu'))
        baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'), selection=selection,
                           tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    except (baseline.S05BaselineError, ValueError):
        assert not seen, 'unsupported adaptive schedule must be rejected before selection training'
    else:
        assert seen and len(set(seen)) == 1 and seen[0][0] != 'plateau'


@pytest.mark.parametrize('verification', [None, {}, {'decision': 'failed'},
                                           {'decision': 'ok', 'full_tfidf_id': 'wrong'}])
def test_refit_api_fails_closed_without_verified_matching_tfidf(synthetic, monkeypatch, verification):
    cfg, manifest, loaded, _, _ = synthetic
    monkeypatch.setattr(baseline, 'train_model', lambda *a, **k: pytest.fail('training without verified TF-IDF'))
    with pytest.raises((baseline.S05BaselineError, ValueError)):
        baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'), tfidf_verification=verification)


def test_validation_transform_failure_keeps_diagnostics(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    original = baseline._transform
    def transform(loaded_arg, names, *args, **kwargs):
        if '' in names:
            raise MemoryError('synthetic validation transform failure')
        return original(loaded_arg, names, *args, **kwargs)
    monkeypatch.setattr(baseline, '_transform', transform)
    try:
        baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                           tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    except (baseline.S05BaselineError, RuntimeError, MemoryError, ValueError):
        pass
    out = Path(cfg.output_dir)
    for name in ('runs.csv', 'history.csv', 'stages.csv', 'per_class.parquet', 'manifest.json'):
        assert (out / name).is_file(), name
    runs = pd.read_csv(out / 'runs.csv')
    assert set(runs.train_seed) == set(cfg.seeds)
    assert not (runs.status == 'success').any()
    assert runs.l3_macro_f1.isna().all()
    report.build_s05_report(str(out), str(out / 'report'))


def test_selection_curves_and_gap_are_rendered(saved, tmp_path):
    history = pd.read_csv(saved / 'history.csv')
    extra = pd.DataFrame([('selection:unique_candidate', 1, 'train', .21, .7, .8, .01),
                          ('selection:unique_candidate', 1, 'monitor', .58, .5, .6, .01)],
                         columns=report.HISTORY_REQUIRED)
    pd.concat([history, extra]).to_csv(saved / 'history.csv', index=False)
    gap = report._train_monitor_gap(pd.concat([history, extra]), pd.read_csv(saved / 'runs.csv'))
    row = gap[gap.run_id == 'selection:unique_candidate']
    assert len(row) == 1 and row.iloc[0].gap_monitor_minus_train == pytest.approx(.37)
    report.build_s05_report(str(saved), str(tmp_path / 'report'))
    html = (tmp_path / 'report/report.html').read_text()
    assert 'selection:unique_candidate' in html and '0.3700' in html


def test_report_uses_corrected_split_support(saved, tmp_path):
    audit = saved / 'audit'
    audit.mkdir()
    pd.DataFrame({'L3': ['stale'], 'support_all': [999], 'kept_after_min_count': [True]}).to_csv(audit / 'class_support_split.csv', index=False)
    pd.DataFrame({'L3': ['a', 'b', 'rare'], 'support_all': [10, 10, 3],
                  'kept_after_min_count': [True, True, False]}).to_csv(audit / 'class_support_split_corrected.csv', index=False)
    report.build_s05_report(str(saved), str(tmp_path / 'report'))
    html = (tmp_path / 'report/report.html').read_text()
    assert 'Классов после min_count: 2; исключено редких L3: 1' in html


def gate_inputs(saved):
    runs = pd.read_csv(saved / 'runs.csv').iloc[:3].copy()
    runs['config_id'] = 'A'
    runs['full_tfidf_id'] = 'full-id'
    history = pd.DataFrame([('selection:base', 1, 'train', .4, .6, .7, .01),
                            ('selection:base', 1, 'monitor', .5, .5, .6, .01),
                            ('seed_42', 1, 'monitor', .5, .5, .6, .01)],
                           columns=report.HISTORY_REQUIRED)
    audit = pd.DataFrame([{'part': part, 'n_rows': 2, 'n_row_ids_unique': 2,
                           report.SPLIT_AUDIT_OVERLAP_COLUMNS[0]: 0}
                          for part in ('outer_train', 'comparison_val', 'sealed_test')])
    checks = {key: True for key in ('seed_matches', 'n_features_matches', 'n_classes_match',
              'label_maps_match', 'feature_contract_matches', 'full_tfidf_id_matches',
              'train_fingerprint_matches', 'fixed_epochs_matches')}
    checks.update(restored_best=False, epochs_completed=1, protocol='fixed_refit')
    selected = {'run_id': 'seed_42', 'provenance_checks': checks,
                'roundtrip': {'status': 'success', 'metrics_match': True}}
    selection = {'chosen': {'name': 'base', 'fixed_epochs': 1},
                 'candidates': [{'name': 'base', 'best_epoch': 1}]}
    return [runs, history, {}, {'decision': 'ok', 'full_tfidf_id': 'full-id'}, selected,
            {'l3_macro_f1': .1, 'evaluated_on': 'comparison_val'},
            report._train_monitor_gap(history, runs), audit, selection, True]


def test_gate_accepts_false_restored_best_as_successful_provenance(saved):
    args = gate_inputs(saved)
    gate = report._decide_gate(*args)
    assert not any('provenance' in reason.lower() for reason in gate['hard_reasons'])


@pytest.mark.parametrize('missing', ['selection', 'provenance', 'tfidf_id', 'audit', 'history'])
def test_gate_never_ready_with_incomplete_evidence(saved, missing):
    args = gate_inputs(saved)
    # Isolate incomplete evidence from the separate restored_best regression.
    args[4]['provenance_checks'].pop('restored_best')
    if missing == 'selection':
        args[8] = {}
    elif missing == 'provenance':
        args[4]['provenance_checks'] = {}
    elif missing == 'tfidf_id':
        args[3] = {'decision': 'ok'}
        args[0]['full_tfidf_id'] = None
    elif missing == 'audit':
        args[7] = args[7].iloc[:0]
    else:
        args[1] = pd.DataFrame([('seed_42', 1, 'monitor', None, None, None, None)], columns=report.HISTORY_REQUIRED)
    assert report._decide_gate(*args)['decision'] != 'ready'
