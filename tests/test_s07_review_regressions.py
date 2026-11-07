"""Independent regressions for frozen scoring, partial provenance and cost reporting."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from kkt_clf import s07, s07_report as report
from kkt_clf.features import FeatureBuilder
from kkt_clf.vectorizer import load_vectorizer_artifact
from tests.test_s07_integrity_independent import synthetic, _json, _seal
from tests.test_s07_report_independent import _tables


def test_evaluation_uses_verified_frozen_texts_not_caller_substitutions(synthetic, monkeypatch):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    plan = s07.load_plan(root)
    run = plan['runs'][0]
    parts = s07.load_split(root, plan, run['split_id'])
    frozen = parts['comparison_val'].copy(deep=True)
    supplied = copy.deepcopy(parts)
    supplied['comparison_val']['name'] = 'CORRUPTED caller text outside frozen validation'
    pd.testing.assert_frame_equal(supplied['comparison_val'].drop(columns='name'), frozen.drop(columns='name'))
    full = load_vectorizer_artifact(cfg.full_tfidf)
    fid = plan['sources']['features']['sha256']
    s07._train_one(root, plan, run, full.builder, fid, parts, torch.device('cpu'))
    transformed, inferred = [], []
    transform = FeatureBuilder.transform
    def capture_transform(builder, names, *args, **kwargs):
        transformed.append(list(names))
        return transform(builder, names, *args, **kwargs)
    def text_sensitive_predict(model, X, *args, **kwargs):
        expected = transform(full.builder, frozen.name.tolist())
        assert X.shape == expected.shape and (X != expected).nnz == 0
        inferred.append(True)
        # Deliberately imperfect predictions make both accuracy and F1 checkable.
        return SimpleNamespace(l3_top1=np.array([0, 1, 0, 1])), None
    monkeypatch.setattr(FeatureBuilder, 'transform', capture_transform)
    monkeypatch.setattr(s07, 'predict', text_sensitive_predict)
    try:
        s07._evaluate_one(root, plan, run, full.builder, fid, supplied, torch.device('cpu'))
    except s07.S07Error:
        assert not inferred, 'Caller corruption must be rejected before inference'
        return
    assert inferred and transformed[-1] == frozen.name.tolist()
    base = root / 'runs' / run['run_id']
    predictions = pd.read_parquet(base / 'predictions_val.parquet')
    assert predictions.row_id.tolist() == frozen.row_id.tolist()
    assert predictions.true_L3.tolist() == frozen.L3.tolist()
    assert predictions.predicted_L3.tolist() == ['alpha', 'beta', 'alpha', 'beta']
    from sklearn.metrics import accuracy_score, f1_score
    result = s07.read_sealed(base / 'result.json')
    assert result['metrics']['l3_accuracy'] == accuracy_score(frozen.L3, predictions.predicted_L3)
    assert result['metrics']['l3_macro_f1'] == f1_score(
        frozen.L3, predictions.predicted_L3, labels=plan['label_maps']['L3'], average='macro', zero_division=0)
    assert report.build_report(root)['statuses']['success'] == 1


def _non_success(root, plan, run, status, fid):
    path = root / 'runs' / run['run_id'] / 'result.json'
    path.parent.mkdir(parents=True)
    _json(path, _seal({'status': status, 'reason': 'Synthetic interrupted dependency/model stage',
                       'provenance': s07.run_provenance(plan, run, fid)}))
    assert not (root / 'execution_finished.json').exists()


@pytest.mark.parametrize('method', s07.METHODS[1:])
@pytest.mark.parametrize('status', ['skipped', 'pending', 'failed'])
def test_unavailable_selected_dependency_rejects_self_supplied_feature_id(synthetic, method, status):
    cfg, root = synthetic
    plan = s07.write_plan(cfg, root)
    run = next(r for r in plan['runs'] if r['method'] == method)
    _non_success(root, plan, run, status, 'a' * 64)
    with pytest.raises(s07.S07Error):
        report.build_report(root)


@pytest.mark.parametrize('method', s07.METHODS[1:])
def test_honest_skipped_unavailable_dependency_stays_reportable(synthetic, method):
    cfg, root = synthetic
    plan = s07.write_plan(cfg, root)
    run = next(r for r in plan['runs'] if r['method'] == method)
    _non_success(root, plan, run, 'skipped', 'unavailable')
    result = report.build_report(root)
    manifest, tables = report.load_report_tables(root / 'report')
    assert result['statuses'] == {'success': 0, 'failed': 0, 'skipped': 1, 'pending': 35}
    assert not result['complete'] and not manifest['complete']
    row = tables['runs.csv'].set_index('run_id').loc[run['run_id']]
    assert row.status == 'skipped' and pd.notna(row.reason)
    assert row[['l3_macro_f1', 'l3_accuracy', 'delta_macro_f1', 'delta_accuracy']].isna().all()
    assert tables['predictions_val.parquet'].empty and tables['per_class.parquet'].empty


@pytest.mark.parametrize('fid', ['unavailable', 'a' * 64])
def test_failed_selected_run_without_verified_dependency_fails_closed(synthetic, fid):
    cfg, root = synthetic
    plan = s07.write_plan(cfg, root)
    run = next(r for r in plan['runs'] if r['method'] == 'global')
    _non_success(root, plan, run, 'failed', fid)
    with pytest.raises(s07.S07Error):
        report.build_report(root)


@pytest.mark.parametrize('identity', ['verified', 'unavailable', 'forged'])
def test_failed_selected_run_requires_available_dependency_identity(synthetic, identity):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    plan = s07.load_plan(root)
    full = load_vectorizer_artifact(cfg.full_tfidf)
    run = next(r for r in plan['runs'] if r['method'] == 'global')
    parts = s07.load_split(root, plan, run['split_id'])
    s07._prepare_selected(root, plan, run['split_id'], 'global', full, parts, torch.device('cpu'))
    fid = s07.sha(root / 'shared' / run['split_id'] / 'global' / 'tfidf' / 'features.joblib')
    supplied_fid = {'verified': fid, 'unavailable': 'unavailable', 'forged': 'a' * 64}[identity]
    _non_success(root, plan, run, 'failed', supplied_fid)
    if identity != 'verified':
        with pytest.raises(s07.S07Error):
            report.build_report(root)
    else:
        result = report.build_report(root)
        assert not result['complete'] and result['statuses']['failed'] == 1
        _, tables = report.load_report_tables(root / 'report')
        row = tables['runs.csv'].set_index('run_id').loc[run['run_id']]
        assert pd.isna(row.l3_macro_f1) and tables['predictions_val.parquet'].empty


def test_gain_counts_distinguish_seed_pairs_from_split_means(synthetic):
    runs, pcs, preds = _tables()
    runs, _, _, _ = report._comparison_tables(runs, pcs, preds)
    summary = report.paired_summary(runs)
    group = summary[summary.method == 'class_aware']
    within = group[group.level == 'within_split']
    across = group[group.level == 'across_split_means'].iloc[0]
    assert (within.gains.sum(), within.losses.sum(), within.ties.sum()) == (6, 3, 0)
    assert (across.gains, across.losses, across.ties) == (2, 1, 0)
    assert within.paired_count.sum() == across.paired_count == 9
    assert across.split_count == 3
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    report.build_report(root)
    manifest, _ = report.load_report_tables(root / 'report')
    documentation = ' '.join(str(v) for v in manifest['formulas'].values()).lower()
    assert 'seed pairs' in documentation and 'split means' in documentation


def _capture_render_frames(monkeypatch):
    frames = []
    original = pd.DataFrame.to_html
    def capture(frame, *args, **kwargs):
        frames.append(frame.copy(deep=True))
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, 'to_html', capture)
    return frames


@pytest.mark.parametrize('all_unpaired', [False, True])
def test_partial_class_extremes_do_not_display_unpaired_nan_deltas(synthetic, monkeypatch, all_unpaired):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    report.build_report(root)
    manifest, tables = report.load_report_tables(root / 'report')
    # Presentation oracle with known tails plus a class lacking its baseline.
    tables['per_class.parquet'] = pd.DataFrame([
        {'method': 'global', 'L3': 'unpaired', 'delta_f1': np.nan},
        {'method': 'global', 'L3': 'loss', 'delta_f1': -.25},
        {'method': 'class_aware', 'L3': 'gain', 'delta_f1': .125}])
    if all_unpaired:
        tables['per_class.parquet']['delta_f1'] = np.nan
    monkeypatch.setattr(report, 'load_report_tables', lambda path: (manifest, tables))
    frames = _capture_render_frames(monkeypatch)
    report.render_report(root / 'report', root / 'partial_extremes.html')
    extremes = [f for f in frames if {'L3', 'delta_f1'} <= set(f.columns)]
    assert len(extremes) == 1
    assert extremes[0].delta_f1.notna().all()
    assert set(extremes[0].L3) == (set() if all_unpaired else {'loss', 'gain'})


def test_checkpoint_transform_and_training_share_cost_role_and_gxi_totals(synthetic, monkeypatch):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    sid = 'split_101'
    events = [('none', 42, 'train_transform', 2.), ('none', 42, 'train', 7.),
              ('none', 42, 'inference', 3.), ('none', 43, 'train', 11.),
              ('gradient_x_input', None, 'ranking', 13.), ('gradient_x_input', None, 'refit', 17.),
              ('gradient_x_input', 42, 'train_transform', 19.),
              ('gradient_x_input', 42, 'train', 23.), ('gradient_x_input', 42, 'inference', 29.)]
    for method, seed, stage, seconds in events:
        def stage_row(rid, name, status, seconds=seconds):
            return dict(run_id=rid, stage=name, status=status, seconds=seconds,
                        peak_rss_bytes=1024, peak_vram_bytes=None, measurement_note='synthetic')
        measurement = SimpleNamespace(_rss=True, stage_row=stage_row)
        s07._record_stage(root, 'shared' if seed is None else 'run', method, sid,
                          None if seed is None else f'{sid}__{method}__seed_{seed}', stage, measurement)
    frames = _capture_render_frames(monkeypatch)
    assert report.build_report(root)['complete'] is False
    _, tables = report.load_report_tables(root / 'report')
    stages = tables['stages.csv']
    checkpoint = stages[stages.cost_role == 'attribution_checkpoint_and_full_comparator']
    assert set(checkpoint.stage) == {'train', 'train_transform'}
    assert checkpoint.seconds.sum() == 9
    assert (checkpoint.checkpoint_incremental_training_seconds == 0).all()
    assert (stages.charge_count == 1).all() and stages.seconds.sum() == 124
    assert stages.peak_vram_bytes.isna().all()
    allocation = [f for f in frames if 'gxi_total_measured_seconds' in f.columns]
    assert len(allocation) == 1
    row = allocation[0].iloc[0]
    assert row.gxi_total_measured_seconds == 101
    assert row.full_seed42_checkpoint_training_and_transform_seconds == 9
    assert row.gxi_standalone_scenario_seconds == 110
    # Baseline and other-seed training/inference must not be charged to GxI.
    assert stages[stages.method == 'gradient_x_input'].seconds.sum() == 101
