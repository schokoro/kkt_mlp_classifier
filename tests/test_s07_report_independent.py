"""S07 reporting oracles use hand-calculated data and synthetic execution only."""
import itertools
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kkt_clf import s07, s07_report as report
from kkt_clf.vectorizer import load_vectorizer_artifact
from tests.test_s07_integrity_independent import synthetic, _json, _seal


def _tables():
    runs, pcs, preds = [], [], []
    for split, seed, method in itertools.product(range(3), (42, 43, 44), s07.METHODS):
        baseline = .612345678912345 + split * .01 + (seed - 42) * .001
        effect = 0 if method == 'none' else (.01, -.02, .04)[split] + (seed - 43) * .003
        rid = f'split_{split}__{method}__{seed}'
        runs.append({'run_id': rid, 'split_id': f'split_{split}', 'method': method,
                     'seed': seed, 'status': 'success', 'reason': None,
                     'l3_macro_f1': baseline + effect, 'l3_accuracy': baseline + effect / 2,
                     'delta_macro_f1': None, 'delta_accuracy': None})
        for label, support, f1 in [('rare_a', 2, .1), ('rare_b', 4, .9), ('common', 20, .7)]:
            pcs.append({'run_id': rid, 'split_id': f'split_{split}', 'method': method, 'seed': seed,
                        'L3': label, 'support': support, 'precision': f1, 'recall': f1,
                        'f1': f1 + effect})
        preds.extend([{'run_id': rid, 'split_id': f'split_{split}', 'row_id': i,
                       'true_L3': 'rare_a', 'predicted_L3': v, 'empty_features': v is None}
                      for i, v in enumerate(['rare_b', None])])
    return pd.DataFrame(runs), pd.DataFrame(pcs), pd.DataFrame(preds)


def test_unrounded_same_split_seed_pairing_and_separate_variability():
    runs, pcs, predictions = _tables()
    runs, _, _, _ = report._comparison_tables(runs, pcs, predictions)
    full = runs[runs.method == 'none'].set_index(['split_id', 'seed'])
    for _, row in runs.iterrows():
        expected = row.l3_macro_f1 - full.loc[(row.split_id, row.seed), 'l3_macro_f1']
        assert row.delta_macro_f1 == expected
    summary = report.paired_summary(runs)
    method = summary[summary.method == 'class_aware']
    within = method[method.level == 'within_split'].sort_values('split_id')
    assert within.paired_count.tolist() == [3, 3, 3]
    np.testing.assert_allclose(within.mean_delta_macro_f1, [.01, -.02, .04], atol=1e-15)
    np.testing.assert_allclose(within.within_split_seed_std, np.std([-.003, 0, .003]), atol=1e-15)
    assert within.between_split_mean_std.isna().all()
    across = method[method.level == 'across_split_means'].iloc[0]
    assert across.split_count == 3
    assert across.mean_delta_macro_f1 == pytest.approx(.01)
    assert across.between_split_mean_std == pytest.approx(np.std([.01, -.02, .04]))
    assert pd.isna(across.within_split_seed_std)
    assert (across.gains, across.losses, across.ties) == (2, 1, 0)


@pytest.mark.parametrize('status', ['failed', 'pending', 'skipped'])
def test_incomplete_pairs_never_enter_complete_split_summary(status):
    runs, pcs, preds = _tables()
    runs.loc[(runs.split_id == 'split_1') & (runs.method == 'none') & (runs.seed == 43), 'status'] = status
    runs, _, _, _ = report._comparison_tables(runs, pcs, preds)
    selected = runs[(runs.split_id == 'split_1') & (runs.method == 'global') & (runs.seed == 43)].iloc[0]
    assert pd.isna(selected.delta_macro_f1)
    summary = report.paired_summary(runs)
    across = summary[(summary.method == 'global') & (summary.level == 'across_split_means')].iloc[0]
    assert across.split_count == 2
    assert across.paired_count == 6


def test_support_bins_count_classes_once_and_keep_empty_bins():
    runs, pcs, preds = _tables()
    _, _, bins, _ = report._comparison_tables(runs, pcs, preds)
    rare = bins[(bins.run_id == runs.iloc[0].run_id) & (bins.bin == '2-4')].iloc[0]
    assert rare.class_count == 2
    assert rare.example_support == 6
    assert rare.macro_f1 == pytest.approx(.5)  # Class mean, not example-weighted mean.
    assert len(bins) == 36 * 4
    empty = bins[bins.bin.isin(['5-9', '10-19'])]
    assert (empty.class_count == 0).all() and (empty.example_support == 0).all()
    assert empty.macro_f1.isna().all() and empty.delta_macro_f1.isna().all()


def test_confusion_counts_empty_feature_prediction_as_an_error():
    runs, pcs, preds = _tables()
    _, _, _, confusion = report._comparison_tables(runs, pcs, preds)
    first = confusion[confusion.run_id == runs.iloc[0].run_id]
    assert first['count'].sum() == 2
    assert first[first.predicted_L3.isna()]['count'].sum() == 1


def test_feature_jaccard_uses_names_despite_reordered_local_columns():
    a = np.array(['flour', 'milk', 'tea'])
    b = np.array(['tea', 'milk', 'rice'])
    overlap = report.feature_jaccard(a, b)
    assert overlap == {'size_a': 3, 'size_b': 3, 'intersection': 2, 'union': 4, 'jaccard': .5}
    assert report.feature_jaccard(a, a[::-1])['jaccard'] == 1
    assert report.feature_jaccard(['x'], ['y'])['jaccard'] == 0


def test_pending_matrix_reports_all_ten_files_and_rebuilds_with_only_tables(synthetic, tmp_path, monkeypatch):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    result = report.build_report(root)
    assert result['statuses'] == {'success': 0, 'failed': 0, 'skipped': 0, 'pending': 36}
    assert result['complete'] is False
    expected = {'report.html', 'manifest.json', 'runs.csv', 'per_class.parquet',
                'predictions_val.parquet', 'support_bins.csv', 'confusion_pairs.csv',
                'feature_stability.csv', 'stages.csv', 'paired_summary.csv'}
    source = root / 'report'
    assert {p.name for p in source.iterdir()} == expected
    isolated = tmp_path / 'table_only'
    isolated.mkdir()
    for name in expected - {'report.html'}:
        shutil.copyfile(source / name, isolated / name)
    def forbidden(*a, **k):
        pytest.fail('table-only rebuild accessed experiment API')
    for name in ('load_plan', '_verify_checkpoint', 'verify_selected', 'load_split', '_verified_results'):
        monkeypatch.setattr(report, name, forbidden)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.resolve().is_relative_to(isolated.resolve()), f'external rebuild read: {path}'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    report.render_report(isolated, isolated / 'rebuilt.html')
    html = (isolated / 'rebuilt.html').read_text()
    assert 'PARTIAL' in html and 'synthetic' in html
    assert 'src="http' not in html and 'href="http' not in html
    manifest, tables = report.load_report_tables(isolated)
    assert len(tables['runs.csv']) == 36
    assert manifest['complete'] is False


@pytest.mark.parametrize('status', ['failed', 'skipped', 'pending'])
def test_saved_non_success_status_keeps_reason_without_metrics(synthetic, status):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    plan = s07.load_plan(root)
    run = plan['runs'][0]
    path = root / 'runs' / run['run_id'] / 'result.json'
    path.parent.mkdir(parents=True)
    _json(path, _seal({'status': status, 'reason': 'fixture interruption',
                       'provenance': s07.run_provenance(plan, run, plan['sources']['features']['sha256'])}))
    result = report.build_report(root)
    rows = pd.read_csv(root / 'report' / 'runs.csv')
    row = rows[rows.run_id == run['run_id']].iloc[0]
    assert row.status == status and row.reason == 'fixture interruption'
    assert pd.isna(row.l3_macro_f1)
    assert not result['complete'] and result['statuses']['success'] == 0


def test_shared_resource_event_charged_once_and_vram_unavailable(synthetic):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    plan = s07.load_plan(root)
    event = {'plan_sha256': s07.digest(plan), 'split_id': 'split_101', 'method': 'global', 'run_id': None, 'scope': 'shared',
             'stage': 'refit', 'status': 'success', 'seconds': 7.25, 'peak_rss_bytes': 1024,
             'peak_vram_bytes': None, 'charge_count': 1, 'cost_role': 'ordinary',
             'checkpoint_incremental_training_seconds': None, 'measurement_note': 'synthetic'}
    (root / 'stages').mkdir()
    _json(root / 'stages' / 'one.json', _seal(event))
    report.build_report(root)
    stages = pd.read_csv(root / 'report' / 'stages.csv')
    assert len(stages) == 1 and stages.seconds.sum() == 7.25
    assert stages.peak_vram_bytes.isna().all()
    _json(root / 'stages' / 'duplicate.json', _seal(event))
    with pytest.raises(s07.S07Error, match='[Dd]uplicate'):
        report.build_report(root, root / 'report_duplicate')


def test_tiny_real_execution_refit_reload_and_selected_provenance(synthetic, tmp_path):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    result = s07.execute(root)
    assert len(result['success']) == 36, result
    assert not result['failed'] and not result['pending']
    plan = s07.load_plan(root)
    full = load_vectorizer_artifact(cfg.full_tfidf)
    assert len(list((root / 'shared').glob('*/*/dependency_receipt.json'))) == 9
    corpus_names = pd.read_parquet(cfg.corpus).name.tolist()
    for sid, method in itertools.product(plan['splits'], s07.METHODS[1:]):
        loaded = s07.verify_selected(root, plan, sid, method, full)
        X = loaded.builder.transform(corpus_names)
        for branch, offset, count in [('char', 0, 6), ('word', 6, 2)]:
            reduced = getattr(loaded.builder, branch)
            original = getattr(full.builder, branch)
            names = reduced.get_feature_names_out()
            assert len(names) == count
            indices = np.array([original.vocabulary_[name] for name in names])
            np.testing.assert_allclose(reduced.idf_, original.idf_[indices], atol=1e-6)
            norm = np.asarray(X[:, offset:offset+count].power(2).sum(axis=1)).ravel()
            np.testing.assert_allclose(norm[norm > 0], 1, atol=1e-6)
        path = root / 'shared' / sid / method / 'tfidf' / 'features.joblib'
        before = path.read_bytes()
        with path.open('ab') as f:
            f.write(b'changed')
        with pytest.raises(s07.S07Error):
            s07.verify_selected(root, plan, sid, method, full)
        # Restore only our generated fixture, to permit remaining independent checks.
        path.write_bytes(before)
    summary = report.build_report(root)
    assert summary['complete']
    runs = pd.read_csv(root / 'report' / 'runs.csv', float_precision='round_trip')
    assert len(runs) == 36 and (runs.status == 'success').all()


@pytest.mark.parametrize('method', ['global', 'class_aware', 'gradient_x_input'])
def test_split_local_selection_refit_roundtrip_and_negative_inputs(synthetic, method, monkeypatch):
    import torch
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    plan = s07.load_plan(root)
    full = load_vectorizer_artifact(cfg.full_tfidf)
    sid = 'split_101'
    parts = s07.load_split(root, plan, sid)
    if method == 'gradient_x_input':
        run = next(r for r in plan['runs'] if r['run_id'] == f'{sid}__none__seed_42')
        s07._train_one(root, plan, run, full.builder, plan['sources']['features']['sha256'], parts, torch.device('cpu'))
    expected_y = np.array([plan['label_maps']['L3'].index(label) for label in parts['outer_train'].L3])
    expected_X = full.builder.transform(parts['outer_train'].name.tolist())
    called = []
    selection_function = 'compute_margin_attributions' if method == 'gradient_x_input' else 'select_features'
    original_function = getattr(s07, selection_function)
    def checked_ranking(*args, **kwargs):
        X, y = (args[1], args[2]) if method == 'gradient_x_input' else (args[0], args[1])
        assert X.shape == expected_X.shape
        assert (X != expected_X).nnz == 0
        np.testing.assert_array_equal(y, expected_y)
        called.append(True)
        return original_function(*args, **kwargs)
    monkeypatch.setattr(s07, selection_function, checked_ranking)
    s07._prepare_selected(root, plan, sid, method, full, parts, torch.device('cpu'))
    assert called == [True]
    loaded = s07.verify_selected(root, plan, sid, method, full)
    names = pd.read_parquet(cfg.corpus).name.tolist()
    transformed = loaded.builder.transform(names)
    for branch, start, count in [('char', 0, 6), ('word', 6, 2)]:
        reduced, original = getattr(loaded.builder, branch), getattr(full.builder, branch)
        indices = [original.vocabulary_[name] for name in reduced.get_feature_names_out()]
        assert len(indices) == count
        np.testing.assert_allclose(reduced.idf_, original.idf_[indices], atol=1e-6)
        norm = np.asarray(transformed[:, start:start+count].power(2).sum(axis=1)).ravel()
        np.testing.assert_allclose(norm[norm > 0], 1, atol=1e-6)
        sliced = original.transform(names)[:, indices]
        assert np.any(np.abs(np.asarray(sliced.power(2).sum(axis=1)).ravel() - norm) > 1e-5)
    base = root / 'shared' / sid / method
    for filename in ['selection/selection.json', 'selection/ratings.npy', 'tfidf/features.joblib']:
        # Pick actual selection files via the immutable receipt rather than assuming a format.
        target = (base / filename if filename.startswith('tfidf') else
                  next((base / 'selection').rglob('*.npy' if filename.endswith('.npy') else '*.json')))
        old = target.read_bytes()
        target.write_bytes(old + b'alteration')
        with pytest.raises(s07.S07Error):
            s07.verify_selected(root, plan, sid, method, full)
        target.write_bytes(old)
    receipt_path = base / 'dependency_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt.pop('integrity_sha256')
    receipt['provenance']['dependency']['split_id'] = 'split_202'
    _json(receipt_path, _seal(receipt))
    with pytest.raises(s07.S07Error):
        s07.verify_selected(root, plan, sid, method, full)


@pytest.fixture
def baseline_result(synthetic):
    import torch
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    plan = s07.load_plan(root)
    full = load_vectorizer_artifact(cfg.full_tfidf)
    run = plan['runs'][0]
    parts = s07.load_split(root, plan, run['split_id'])
    feature_id = plan['sources']['features']['sha256']
    s07._train_one(root, plan, run, full.builder, feature_id, parts, torch.device('cpu'))
    s07._evaluate_one(root, plan, run, full.builder, feature_id, parts, torch.device('cpu'))
    return root, plan, run


@pytest.mark.parametrize('damage', ['checkpoint', 'training', 'result_labels', 'prediction_hash',
                                     'prediction_rows', 'prediction_labels', 'prediction_split', 'prediction_run'])
def test_report_rejects_corrupt_checkpoint_training_or_predictions(baseline_result, damage):
    root, plan, run = baseline_result
    base = root / 'runs' / run['run_id']
    result = json.loads((base / 'result.json').read_text())
    result.pop('integrity_sha256')
    if damage == 'checkpoint':
        with (base / 'checkpoint' / 'model.pt').open('ab') as f:
            f.write(b'changed')
    elif damage == 'training':
        with (base / 'training.json').open('ab') as f:
            f.write(b'changed')
    elif damage == 'result_labels':
        result['provenance']['label_maps_sha256'] = '0' * 64
        _json(base / 'result.json', _seal(result))
    else:
        path = base / 'predictions_val.parquet'
        frame = pd.read_parquet(path)
        if damage == 'prediction_rows':
            frame.loc[0, 'row_id'] = 900  # Sealed structural identity must never be accepted as val.
        elif damage == 'prediction_labels':
            frame.loc[0, 'true_L3'] = 'beta' if frame.loc[0, 'true_L3'] == 'alpha' else 'alpha'
        elif damage == 'prediction_split':
            frame.loc[0, 'split_id'] = 'split_202'
        else:
            frame.loc[0, 'run_id'] = 'unknown_run'
        frame.to_parquet(path, index=False)
        if damage != 'prediction_hash':
            result['predictions'] = s07.fingerprint(path)
            _json(base / 'result.json', _seal(result))
    with pytest.raises(s07.S07Error):
        report.build_report(root)


def test_verified_none_baseline_metrics_match_saved_predictions(baseline_result):
    from sklearn.metrics import accuracy_score, f1_score
    root, plan, run = baseline_result
    outcome = report.build_report(root)
    assert outcome['statuses']['success'] == 1 and outcome['statuses']['pending'] == 35
    pred = pd.read_parquet(root / 'runs' / run['run_id'] / 'predictions_val.parquet')
    table = pd.read_csv(root / 'report' / 'runs.csv', float_precision='round_trip')
    row = table[table.run_id == run['run_id']].iloc[0]
    assert row.l3_accuracy == accuracy_score(pred.true_L3, pred.predicted_L3)
    assert row.l3_macro_f1 == f1_score(pred.true_L3, pred.predicted_L3, labels=['alpha', 'beta'], average='macro', zero_division=0)
    assert row.delta_macro_f1 == 0


def test_emitted_stability_table_resolves_feature_names(synthetic, monkeypatch):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    original = report._verified_results
    def synthetic_selections(root, plan):
        rows, pcs, preds, _ = original(root, plan)
        selections = {}
        for sid in plan['splits']:
            for method in s07.METHODS[1:]:
                chars = {'split_101': ['a', 'b', 'c'], 'split_202': ['c', 'b', 'd'],
                         'split_303': ['d', 'e', 'f']}[sid]
                words = {'split_101': ['one', 'two'], 'split_202': ['two', 'one'],
                         'split_303': ['three', 'four']}[sid]
                # All objects use dense local 0..N columns; names determine actual overlap.
                selections[(sid, method)] = SimpleNamespace(builder=SimpleNamespace(
                    char=SimpleNamespace(get_feature_names_out=lambda names=chars: np.array(names)),
                    word=SimpleNamespace(get_feature_names_out=lambda names=words: np.array(names))))
        return rows, pcs, preds, selections
    monkeypatch.setattr(report, '_verified_results', synthetic_selections)
    report.build_report(root)
    stability = pd.read_csv(root / 'report' / 'feature_stability.csv')
    assert len(stability) == 18
    pair = stability[(stability.split_a == 'split_101') & (stability.split_b == 'split_202')]
    assert (pair[pair.branch == 'char'].intersection == 2).all()
    assert (pair[pair.branch == 'char'].jaccard == .5).all()
    assert (pair[pair.branch == 'word'].jaccard == 1).all()
    different = stability[(stability.split_a == 'split_101') & (stability.split_b == 'split_303')]
    assert (different.jaccard == 0).all()


def test_report_table_hash_tampering_rejected(synthetic):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    report.build_report(root)
    path = root / 'report' / 'runs.csv'
    with path.open('ab') as f:
        f.write(b'\nchanged')
    with pytest.raises(s07.S07Error):
        report.render_report(root / 'report', root / 'tampered.html')


def test_cli_table_only_rebuild(synthetic, tmp_path):
    import subprocess
    import sys
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    report.build_report(root)
    isolated = tmp_path / 'cli_tables'
    isolated.mkdir()
    for path in (root / 'report').iterdir():
        if path.name != 'report.html':
            shutil.copyfile(path, isolated / path.name)
    output = isolated / 'cli.html'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts' / 'run_s07.py'),
               'rebuild', '--report-dir', str(isolated), '--output-html', str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert output.is_file() and 'PARTIAL' in output.read_text()


def test_empty_feature_predictions_preserve_s06_raw_bias_path(baseline_result, monkeypatch):
    import torch
    from kkt_clf.features import FeatureBuilder
    root, plan, first = baseline_result
    run = next(r for r in plan['runs'] if r['run_id'] == 'split_101__none__seed_43')
    full = load_vectorizer_artifact(plan['config']['full_tfidf'])
    parts = s07.load_split(root, plan, 'split_101')
    feature_id = plan['sources']['features']['sha256']
    s07._train_one(root, plan, run, full.builder, feature_id, parts, torch.device('cpu'))
    original = FeatureBuilder.transform
    validation_names = parts['comparison_val'].name.tolist()
    def one_empty_row(builder, names, *args, **kwargs):
        matrix = original(builder, names, *args, **kwargs)
        if list(names) == validation_names:
            matrix.data[matrix.indptr[0]:matrix.indptr[1]] = 0
            matrix.eliminate_zeros()
        return matrix
    monkeypatch.setattr(FeatureBuilder, 'transform', one_empty_row)
    raw_predictions = []
    original_predict = s07.predict
    def capture_raw(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        raw_predictions.extend(result[0].l3_top1.tolist())
        return result
    monkeypatch.setattr(s07, 'predict', capture_raw)
    s07._evaluate_one(root, plan, run, full.builder, feature_id, parts, torch.device('cpu'))
    predictions = pd.read_parquet(root / 'runs' / run['run_id'] / 'predictions_val.parquet')
    assert predictions.empty_features.iloc[0]
    assert predictions.predicted_L3.tolist() == [plan['label_maps']['L3'][i] for i in raw_predictions]
    summary = report.build_report(root)
    assert summary['statuses']['success'] == 2
    saved = pd.read_parquet(root / 'report' / 'predictions_val.parquet')
    assert len(saved[saved.run_id == run['run_id']]) == len(validation_names)
    assert saved[saved.run_id == run['run_id']].empty_features.sum() == 1


@pytest.mark.parametrize('method', ['global', 'class_aware', 'gradient_x_input'])
def test_changed_partition_fails_before_ranking(synthetic, method, monkeypatch):
    import torch
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    plan = s07.load_plan(root)
    full = load_vectorizer_artifact(cfg.full_tfidf)
    parts = s07.load_split(root, plan, 'split_101')
    parts['outer_train'] = parts['outer_train'].assign(L3='changed_label')
    def forbidden(*a, **k):
        pytest.fail('changed partition reached supervised ranking')
    monkeypatch.setattr(s07, 'select_features', forbidden)
    monkeypatch.setattr(s07, 'compute_margin_attributions', forbidden)
    with pytest.raises(s07.S07Error, match='frozen split'):
        s07._prepare_selected(root, plan, 'split_101', method, full, parts, torch.device('cpu'))
