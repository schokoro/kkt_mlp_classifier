"""S07 verified split-aware tables and table-only, self-contained HTML rendering.

Rebuilding HTML loads no models and reads only manifest plus the eight data
files in the supplied report directory. Original report directories are immutable.
"""
from __future__ import annotations

import html
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .s07 import (METHODS, MODEL_SEEDS, S07Config, WARNINGS, _l3_val_metrics,
                  _verify_checkpoint, dependency_provenance, digest, fingerprint,
                  fresh_root, load_plan, load_split, now, read_sealed, require,
                  row_identity, run_provenance, sealed, sha, stage_cost_role, tree, verify_selected,
                  verify_tree, write_new)

TABLES = {'runs.csv': ['run_id', 'split_id', 'method', 'seed', 'status', 'reason',
                       'l3_macro_f1', 'l3_accuracy', 'delta_macro_f1', 'delta_accuracy',
                       'n_char', 'n_word', 'model_bytes'],
          'per_class.parquet': ['run_id', 'split_id', 'method', 'seed', 'L3', 'support',
                                'precision', 'recall', 'f1', 'delta_precision', 'delta_recall', 'delta_f1'],
          'predictions_val.parquet': ['run_id', 'split_id', 'row_id', 'true_L3', 'predicted_L3', 'empty_features'],
          'support_bins.csv': ['run_id', 'split_id', 'method', 'seed', 'bin', 'class_count',
                               'example_support', 'macro_f1', 'delta_macro_f1'],
          'confusion_pairs.csv': ['run_id', 'split_id', 'method', 'seed', 'true_L3', 'predicted_L3',
                                  'count', 'baseline_count', 'delta_count'],
          'feature_stability.csv': ['method', 'split_a', 'split_b', 'branch', 'status', 'reason',
                                    'size_a', 'size_b', 'intersection', 'union', 'jaccard'],
          'stages.csv': ['event_id', 'split_id', 'method', 'run_id', 'scope', 'stage', 'status',
                         'seconds', 'peak_rss_bytes', 'peak_vram_bytes', 'charge_count', 'cost_role',
                         'checkpoint_incremental_training_seconds', 'measurement_note'],
          'paired_summary.csv': ['level', 'split_id', 'method', 'paired_count', 'split_count',
                                 'mean_delta_macro_f1', 'within_split_seed_std', 'between_split_mean_std',
                                 'min_delta_macro_f1', 'max_delta_macro_f1', 'gains', 'losses', 'ties',
                                 'mean_delta_accuracy', 'mean_macro_f1', 'within_split_metric_std',
                                 'between_split_metric_mean_std']}
BINS = ('2-4', '5-9', '10-19', '>=20')


def support_bin(support: int) -> str:
    require(support >= 2, 'Comparison validation support below 2')
    if support < 5:
        return '2-4'
    if support < 10:
        return '5-9'
    if support < 20:
        return '10-19'
    return '>=20'


def feature_jaccard(a, b) -> dict:
    """Stable feature identities only, independently for each TF-IDF branch."""
    a, b = set(a), set(b)
    intersection, union = len(a & b), len(a | b)
    return {'size_a': len(a), 'size_b': len(b), 'intersection': intersection,
            'union': union, 'jaccard': intersection / union if union else None}


def paired_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Within-split population seed SD and between-split population mean SD.

    No pooling of nine correlated pairs for an independence-based uncertainty
    estimate. Partial pair/split counts remain explicit and can never imply 3/3.
    """
    rows = []
    for method in METHODS:
        group = runs[(runs.method == method) & (runs.status == 'success')]
        for sid in runs.split_id.unique():
            g = group[group.split_id == sid]
            paired = g.dropna(subset=['delta_macro_f1'])
            d = paired.delta_macro_f1.to_numpy(float)
            rows.append({'level': 'within_split', 'split_id': sid, 'method': method,
                         'paired_count': len(d), 'split_count': int(bool(len(d))),
                         'mean_delta_macro_f1': float(d.mean()) if len(d) else None,
                         'within_split_seed_std': float(d.std(ddof=0)) if len(d) else None,
                         'between_split_mean_std': None,
                         'min_delta_macro_f1': float(d.min()) if len(d) else None,
                         'max_delta_macro_f1': float(d.max()) if len(d) else None,
                         'gains': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'ties': int((d == 0).sum()),
                         'mean_delta_accuracy': float(paired.delta_accuracy.mean()) if len(d) else None,
                         'mean_macro_f1': float(g.l3_macro_f1.mean()) if len(g) else None,
                         'within_split_metric_std': float(g.l3_macro_f1.std(ddof=0)) if len(g) else None,
                         'between_split_metric_mean_std': None})
        # All three matched seeds are required before a split enters across-split means.
        within = pd.DataFrame([r for r in rows if r['method'] == method and r['level'] == 'within_split'])
        complete = within[within.paired_count == len(MODEL_SEEDS)]
        d = complete.mean_delta_macro_f1.to_numpy(float)
        means = complete.mean_macro_f1.to_numpy(float)
        rows.append({'level': 'across_split_means', 'split_id': None, 'method': method,
                     'paired_count': int(complete.paired_count.sum()), 'split_count': len(complete),
                     'mean_delta_macro_f1': float(d.mean()) if len(d) else None,
                     'within_split_seed_std': None,
                     'between_split_mean_std': float(d.std(ddof=0)) if len(d) else None,
                     'min_delta_macro_f1': float(d.min()) if len(d) else None,
                     'max_delta_macro_f1': float(d.max()) if len(d) else None,
                     'gains': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'ties': int((d == 0).sum()),
                     'mean_delta_accuracy': float(complete.mean_delta_accuracy.mean()) if len(d) else None,
                     'mean_macro_f1': float(means.mean()) if len(d) else None,
                     'within_split_metric_std': None,
                     'between_split_metric_mean_std': float(means.std(ddof=0)) if len(d) else None})
    return pd.DataFrame(rows, columns=TABLES['paired_summary.csv'])


def _verified_results(root: Path, plan: dict) -> tuple[list, list, list, dict]:
    """Fail closed before reporting any successful row or deserializing its model."""
    import torch
    from .vectorizer import load_vectorizer_artifact

    cfg = S07Config(**plan['config'])
    full = load_vectorizer_artifact(cfg.full_tfidf)
    all_runs, all_pc, all_preds, selections = [], [], [], {}
    finished_path = root / 'execution_finished.json'
    finished = read_sealed(finished_path) if finished_path.exists() else None
    if finished is not None:
        require(finished['plan_sha256'] == digest(plan), 'Execution/plan mismatch')
        actual = {r['run_id']: fingerprint(root / 'runs' / r['run_id'] / 'result.json')
                  for r in plan['runs'] if (root / 'runs' / r['run_id'] / 'result.json').exists()}
        require(actual == finished['results'], 'Missing/modified result in execution ledger')
        listed = finished['summary']
        require(set(listed['success'] + listed['failed'] + listed['skipped'] + listed['pending'])
                == {r['run_id'] for r in plan['runs']}, 'Execution statuses do not cover frozen plan')
    for sid in plan['splits']:
        for method in METHODS[1:]:
            base = root / 'shared' / sid / method
            if (base / 'dependency_receipt.json').exists():
                selections[(sid, method)] = verify_selected(root, plan, sid, method, full)
    for run in plan['runs']:
        sid, method, rid = run['split_id'], run['method'], run['run_id']
        path = root / 'runs' / rid / 'result.json'
        row = {**run, 'status': 'pending', 'reason': 'No completed result',
               'l3_macro_f1': None, 'l3_accuracy': None, 'delta_macro_f1': None, 'delta_accuracy': None,
               'model_bytes': None}
        if not path.exists():
            all_runs.append(row)
            continue
        result = read_sealed(path)
        status = result['status']
        require(status in ('success', 'failed', 'skipped', 'pending'), 'Unknown run status')
        row.update(status=status, reason=result.get('reason'))
        require(status == 'success' or bool(result.get('reason')), 'Non-success status requires reason')
        if method == 'none':
            feature_id = plan['sources']['features']['sha256']
        elif (sid, method) in selections:
            feature_id = sha(root / 'shared' / sid / method / 'tfidf' / 'features.joblib')
        else:
            require(status in ('skipped', 'pending'),
                    'Successful/failed selected run lacks verified dependencies')
            # The executor uses this sentinel when shared preparation fails.
            # Never trust a result's own feature identity, even without a ledger.
            feature_id = 'unavailable'
        require(result['provenance'] == run_provenance(plan, run, feature_id), 'Result/plan provenance mismatch')
        if finished:
            require(rid in finished['summary'][status], 'Result/execution status mismatch')
        if status != 'success':
            all_runs.append(row)
            continue
        require(method == 'none' or (sid, method) in selections, 'Successful selected run lacks verified dependencies')
        builder = full.builder if method == 'none' else selections[(sid, method)].builder
        require(result['checkpoint_receipt_sha256'] == sha(path.parent / 'checkpoint_receipt.json'),
                'Checkpoint receipt changed')
        require(result['training_sha256'] == sha(path.parent / 'training.json'), 'Training record changed')
        training = read_sealed(path.parent / 'training.json')
        require(training['provenance'] == result['provenance']
                and len(training['history']) == plan['training']['train']['epochs']
                and not training['best_monitor_diagnostic_only'].get('restored_best'), 'Final training rule failed')
        # Verify metadata/files and deserialized feature/model contracts, no inference.
        art = _verify_checkpoint(root, plan, run, builder, feature_id, torch.device('cpu'))
        del art
        pred_path = path.parent / 'predictions_val.parquet'
        require(fingerprint(pred_path) == result['predictions'], 'Prediction checksum/size mismatch')
        df = pd.read_parquet(pred_path)
        val = load_split(root, plan, sid)['comparison_val']
        require(set(df.columns) == set(TABLES['predictions_val.parquet']), 'Prediction schema mismatch')
        require(len(df) == len(val) and df.row_id.is_unique and set(df.row_id) == set(val.row_id),
                'Prediction validation row identity mismatch')
        require(set(df.run_id) == {rid} and set(df.split_id) == {sid}, 'Prediction run/split mismatch')
        ordered = df.set_index('row_id').loc[val.row_id].reset_index()
        require(ordered.true_L3.tolist() == val.L3.tolist(), 'Prediction labels not from active comparison validation')
        labels = plan['label_maps']['L3']
        require(ordered.predicted_L3.isin(labels).all(), 'Unknown/null predicted label')
        require(ordered.empty_features.isin([True, False]).all(), 'Invalid empty feature flag')
        ids = {label: i for i, label in enumerate(labels)}
        truth = ordered.true_L3.map(ids).to_numpy(np.int64)
        pred = ordered.predicted_L3.map(ids).to_numpy(np.int64)
        f1, acc, pc = _l3_val_metrics(truth, pred, labels)
        require(f1 == result['metrics']['l3_macro_f1'] and acc == result['metrics']['l3_accuracy']
                and pc == result['per_class'], 'Saved predictions do not reproduce unrounded metrics')
        row.update(l3_macro_f1=f1, l3_accuracy=acc, model_bytes=(path.parent / 'checkpoint' / 'model.pt').stat().st_size)
        all_runs.append(row)
        all_pc.extend({**c, **{k: run[k] for k in ('run_id', 'split_id', 'method', 'seed')}} for c in pc)
        all_preds.extend(ordered.to_dict('records'))
    return all_runs, all_pc, all_preds, selections


def confusion_counts(predictions: pd.DataFrame) -> dict:
    wrong = predictions[predictions.true_L3 != predictions.predicted_L3].copy()
    # Missing labels represent an explicit error in diagnostic inputs, never a dropped group.
    sentinel = '__S07_MISSING_PREDICTION__'
    wrong['predicted_L3'] = wrong.predicted_L3.fillna(sentinel)
    counts = wrong.groupby(['true_L3', 'predicted_L3'], dropna=False).size().to_dict()
    return {(true, None if predicted == sentinel else predicted): int(count)
            for (true, predicted), count in counts.items()}


def _comparison_tables(runs: pd.DataFrame, pcs: pd.DataFrame, predictions: pd.DataFrame) -> tuple:
    successful_ids = set(runs.loc[runs.status == 'success', 'run_id'])
    pcs = pcs[pcs.run_id.isin(successful_ids)].copy()
    predictions = predictions[predictions.run_id.isin(successful_ids)].copy()
    baseline = runs[(runs.method == 'none') & (runs.status == 'success')].set_index(['split_id', 'seed'])
    for i, row in runs[runs.status == 'success'].iterrows():
        key = (row.split_id, row.seed)
        if key in baseline.index:
            runs.at[i, 'delta_macro_f1'] = row.l3_macro_f1 - baseline.loc[key, 'l3_macro_f1']
            runs.at[i, 'delta_accuracy'] = row.l3_accuracy - baseline.loc[key, 'l3_accuracy']
    for name in ('precision', 'recall', 'f1'):
        pcs['delta_' + name] = np.nan
    if not pcs.empty:
        base_pc = pcs[pcs.method == 'none'].set_index(['split_id', 'seed', 'L3'])
        for i, row in pcs.iterrows():
            key = (row.split_id, row.seed, row.L3)
            if key in base_pc.index:
                require(row.support == base_pc.loc[key, 'support'], 'Paired per-class support mismatch')
                for name in ('precision', 'recall', 'f1'):
                    pcs.at[i, 'delta_' + name] = row[name] - base_pc.loc[key, name]
    bins, confusion = [], []
    for _, run in runs[runs.status == 'success'].iterrows():
        g = pcs[pcs.run_id == run.run_id].copy()
        g['bin'] = g.support.map(support_bin)
        for b in BINS:
            subset = g[g.bin == b]
            bins.append({**{k: run[k] for k in ('run_id', 'split_id', 'method', 'seed')},
                         'bin': b, 'class_count': len(subset), 'example_support': int(subset.support.sum()),
                         'macro_f1': float(subset.f1.mean()) if len(subset) else None,
                         'delta_macro_f1': float(subset.delta_f1.mean())
                         if len(subset) and subset.delta_f1.notna().all() else None})
        pred = predictions[predictions.run_id == run.run_id]
        wrong = confusion_counts(pred)
        base_run = baseline.loc[(run.split_id, run.seed), 'run_id'] if (run.split_id, run.seed) in baseline.index else None
        bp = predictions[predictions.run_id == base_run]
        base_wrong = confusion_counts(bp)
        for true, predicted in sorted(set(wrong) | set(base_wrong), key=lambda pair: tuple(str(x) for x in pair)):
            count = int(wrong.get((true, predicted), 0))
            bc = int(base_wrong.get((true, predicted), 0)) if base_run is not None else None
            confusion.append({**{k: run[k] for k in ('run_id', 'split_id', 'method', 'seed')},
                              'true_L3': true, 'predicted_L3': predicted, 'count': count,
                              'baseline_count': bc, 'delta_count': count - bc if bc is not None else None})
    return (runs, pcs, pd.DataFrame(bins, columns=TABLES['support_bins.csv']),
            pd.DataFrame(confusion, columns=TABLES['confusion_pairs.csv']))


def build_report(output_root: str | Path, report_dir: str | Path | None = None) -> dict:
    """Read verified saved results only; never train, refit, or run inference."""
    root = Path(output_root)
    plan = load_plan(root)
    rows, pc, preds, selections = _verified_results(root, plan)
    runs = pd.DataFrame(rows, columns=TABLES['runs.csv'])
    per_class = pd.DataFrame(pc, columns=TABLES['per_class.parquet'])
    predictions = pd.DataFrame(preds, columns=TABLES['predictions_val.parquet'])
    runs, per_class, bins, confusion = _comparison_tables(runs, per_class, predictions)
    stability = []
    for method in METHODS[1:]:
        for a, b in itertools.combinations(plan['splits'], 2):
            for branch in ('char', 'word'):
                row = {'method': method, 'split_a': a, 'split_b': b, 'branch': branch,
                       'status': 'pending', 'reason': 'Both split-local verified refits required'}
                if (a, method) in selections and (b, method) in selections:
                    fa = getattr(selections[(a, method)].builder, branch).get_feature_names_out()
                    fb = getattr(selections[(b, method)].builder, branch).get_feature_names_out()
                    row.update(status='success', reason=None, **feature_jaccard(fa, fb))
                stability.append(row)
    stages = []
    events = root / 'stages'
    finished = read_sealed(root / 'execution_finished.json') if (root / 'execution_finished.json').exists() else None
    if events.exists():
        if finished:
            require(tree(events) == finished['stages'], 'Stage event ledger changed')
        for path in sorted(events.glob('*.json')):
            event = read_sealed(path)
            require(event['plan_sha256'] == digest(plan), 'Stage/plan provenance mismatch')
            require(event['split_id'] in plan['splits'] and event['method'] in METHODS, 'Invalid stage identity')
            require(event['charge_count'] == 1, 'Shared/run stage must be charged exactly once')
            require(event['scope'] in ('shared', 'run') and event['status'] in ('success', 'failed', 'skipped', 'pending'),
                    'Invalid stage scope/status')
            event['event_id'] = path.stem
            stages.append(event)
    elif finished:
        require(not finished['stages'], 'Missing stage ledger')
    stages_df = pd.DataFrame(stages, columns=TABLES['stages.csv'])
    if not stages_df.empty:
        require(not stages_df.duplicated(['split_id', 'method', 'run_id', 'scope', 'stage']).any(),
                'Duplicate stage cost event')
    summary = paired_summary(runs)
    tables = {'runs.csv': runs, 'per_class.parquet': per_class, 'predictions_val.parquet': predictions,
              'support_bins.csv': bins, 'confusion_pairs.csv': confusion,
              'feature_stability.csv': pd.DataFrame(stability, columns=TABLES['feature_stability.csv']),
              'stages.csv': stages_df, 'paired_summary.csv': summary}
    report = fresh_root(report_dir or root / 'report')
    for name, data in tables.items():
        if name.endswith('.parquet'):
            data.to_parquet(report / name, index=False)
        else:
            data.to_csv(report / name, index=False, float_format='%.17g')
    statuses = {status: int((runs.status == status).sum()) for status in ('success', 'failed', 'skipped', 'pending')}
    manifest = {'schema': 's07-report@1', 'created_at': now(), 'plan': plan, 'plan_sha256': digest(plan),
                'source_fingerprints': plan['sources'], 'statuses': statuses,
                'complete': statuses['success'] == 36, 'study_kind': plan['study_kind'],
                'warnings': WARNINGS + ([] if statuses['success'] == 36 else ['PARTIAL experiment; no completed-study conclusion.']),
                'measurement_sources': {'S07': 'Verified saved predictions and stage event measurements',
                                        'S06': 'Frozen protocol and full unsupervised TF-IDF references only; no reused quality metrics'},
                'formulas': {'paired_delta': 'unrounded method(split,seed) minus none(split,seed)',
                             'seed_variability': 'population SD of three paired deltas within a split (ddof=0)',
                             'split_variability': 'population SD of complete three-seed split mean deltas (ddof=0)',
                             'effect_counts': 'within_split gains/losses/ties count seed pairs; across_split_means counts complete split means; split_count counts contributing splits (0/1 within a split)',
                             'support_bins': 'each L3 counted once per bin; example_support=sum(validation support)',
                             'jaccard': 'branch feature-name intersection / union, same full vocabulary provenance',
                             'cost': 'sum distinct event seconds once; shared ranking/refit not multiplied by seeds',
                             'attribution_checkpoint': 'full seed42 training and input transform charged once to full comparator; GxI standalone scenario adds that same cost, incremental comparison adds 0'},
                'tables': {name: fingerprint(report / name) for name in TABLES}}
    write_new(report / 'manifest.json', sealed(manifest))
    render_report(report, report / 'report.html')
    return {'report_dir': str(report), 'statuses': statuses, 'complete': manifest['complete']}


def load_report_tables(report_dir: str | Path) -> tuple[dict, dict]:
    """Only this directory is read; no source/output-root resolution in rebuild."""
    report = Path(report_dir)
    manifest = read_sealed(report / 'manifest.json')
    require(manifest.get('schema') == 's07-report@1', 'Unknown report schema')
    require(set(manifest['tables']) == set(TABLES), 'Incomplete table-only report manifest')
    require(manifest['plan_sha256'] == digest(manifest['plan']), 'Saved report plan mismatch')
    tables = {}
    for name, columns in TABLES.items():
        require(fingerprint(report / name) == manifest['tables'][name], f'Report table changed: {name}')
        data = pd.read_parquet(report / name) if name.endswith('.parquet') else pd.read_csv(
            report / name, float_precision='round_trip', keep_default_na=True)
        require(data.columns.tolist() == columns, f'Report column schema changed: {name}')
        tables[name] = data
    runs = tables['runs.csv']
    require(len(runs) == 36 and runs.run_id.is_unique
            and set(runs.run_id) == {r['run_id'] for r in manifest['plan']['runs']}, 'Report run matrix mismatch')
    require(runs.status.isin(['success', 'failed', 'skipped', 'pending']).all(), 'Invalid table statuses')
    require(manifest['statuses'] == {s: int((runs.status == s).sum()) for s in manifest['statuses']},
            'Table/manifest statuses disagree')
    require(manifest['complete'] == bool((runs.status == 'success').all()), 'Incorrect completeness claim')
    validate_report_tables(manifest, tables)
    return manifest, tables


def validate_report_tables(manifest: dict, tables: dict) -> None:
    """Semantic cross-checks using table-only identities and the saved immutable plan."""
    plan, runs = manifest['plan'], tables['runs.csv']
    index = runs.set_index('run_id')
    labels = plan['label_maps']['L3']
    require(set(manifest['statuses']) == {'success', 'failed', 'skipped', 'pending'}, 'Incomplete status universe')
    for planned in plan['runs']:
        row = index.loc[planned['run_id']]
        require(all(row[k] == planned[k] for k in ('split_id', 'method', 'seed', 'n_char', 'n_word')),
                'Report run/config identity mismatch')
        if row.status != 'success':
            require(pd.isna(row.l3_macro_f1) and pd.isna(row.l3_accuracy)
                    and pd.isna(row.delta_macro_f1) and pd.isna(row.delta_accuracy),
                    'Non-success table row contains quality metrics')
            require(pd.notna(row.reason), 'Non-success report row needs reason')
    pcs, predictions = tables['per_class.parquet'], tables['predictions_val.parquet']
    successful = set(runs.loc[runs.status == 'success', 'run_id'])
    require(set(pcs.run_id) <= successful and set(predictions.run_id) <= successful,
            'Predictions/per-class rows belong to non-success/unplanned run')
    for rid in successful:
        row = index.loc[rid]
        pred = predictions[predictions.run_id == rid]
        per_class = pcs[pcs.run_id == rid]
        split_spec = plan['splits'][row.split_id]['parts']['comparison_val']
        require(len(pred) == split_spec['rows'] and pred.row_id.is_unique
                and digest(sorted(pred.row_id.astype(int).tolist())) == split_spec['row_ids_sha256'],
                'Table prediction row identities differ from frozen validation')
        require(digest(pred[['row_id', 'true_L3']].sort_values('row_id').values.tolist())
                == split_spec['row_l3_sha256'], 'Table true labels differ from frozen validation')
        require(set(pred.split_id) == {row.split_id} and pred.predicted_L3.isin(labels).all(),
                'Table prediction split/label map mismatch')
        require(len(per_class) == len(labels) and per_class.L3.is_unique and set(per_class.L3) == set(labels),
                'Table per-class universe mismatch')
        ids = {label: i for i, label in enumerate(labels)}
        f1, accuracy, expected_pc = _l3_val_metrics(pred.true_L3.map(ids).to_numpy(np.int64),
                                                  pred.predicted_L3.map(ids).to_numpy(np.int64), labels)
        require(f1 == row.l3_macro_f1 and accuracy == row.l3_accuracy, 'Table prediction metrics mismatch')
        expected_pc = pd.DataFrame(expected_pc).set_index('L3')
        actual_pc = per_class.set_index('L3').loc[labels]
        require(np.array_equal(actual_pc.support, expected_pc.support), 'Table per-class support mismatch')
        for metric in ('precision', 'recall', 'f1'):
            require(np.allclose(actual_pc[metric], expected_pc[metric], rtol=0, atol=1e-15),
                    'Table per-class metric mismatch')
    expected_runs, expected_pc, expected_bins, expected_confusion = _comparison_tables(
        runs.copy(), pcs.copy(), predictions.copy())
    def equal_frame(actual, expected, columns):
        require(len(actual) == len(expected), 'Inconsistent table row count')
        left, right = actual[columns].reset_index(drop=True).copy(), expected[columns].reset_index(drop=True).copy()
        for column in columns:
            if right[column].dtype == object:
                left[column] = left[column].where(left[column].notna(), np.nan)
                right[column] = right[column].where(right[column].notna(), np.nan)
        try:
            pd.testing.assert_frame_equal(left, right,
                                          check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-15)
        except AssertionError as exc:
            require(False, 'Inconsistent derived report tables: ' + str(exc)[:200])
    equal_frame(runs, expected_runs, ['delta_macro_f1', 'delta_accuracy'])
    equal_frame(pcs, expected_pc, ['delta_precision', 'delta_recall', 'delta_f1'])
    equal_frame(tables['support_bins.csv'], expected_bins, TABLES['support_bins.csv'])
    equal_frame(tables['confusion_pairs.csv'], expected_confusion, TABLES['confusion_pairs.csv'])
    equal_frame(tables['paired_summary.csv'], paired_summary(runs), TABLES['paired_summary.csv'])
    stability = tables['feature_stability.csv']
    expected_keys = {(m, a, b, branch) for m in METHODS[1:] for a, b in itertools.combinations(plan['splits'], 2)
                     for branch in ('char', 'word')}
    require(len(stability) == 18 and set(stability[['method', 'split_a', 'split_b', 'branch']].itertuples(
        index=False, name=None)) == expected_keys, 'Incomplete feature stability table')
    for _, row in stability[stability.status == 'success'].iterrows():
        require(row.union == row.size_a + row.size_b - row.intersection
                and 0 <= row.intersection <= min(row.size_a, row.size_b)
                and abs(row.jaccard - row.intersection / row.union) <= 1e-15, 'Invalid Jaccard table values')
    stages = tables['stages.csv']
    require(stages.empty or (stages.charge_count == 1).all(), 'Repeated stage cost charging')
    for _, row in stages.iterrows():
        role = stage_cost_role(row.method, None if pd.isna(row.run_id) else row.run_id, row.stage)
        require(row.cost_role == role, 'Inconsistent stage checkpoint cost role')
        require(row.checkpoint_incremental_training_seconds == 0.0
                if role == 'attribution_checkpoint_and_full_comparator'
                else pd.isna(row.checkpoint_incremental_training_seconds),
                'Inconsistent checkpoint incremental cost')


def render_report(report_dir: str | Path, output_html: str | Path) -> str:
    """Rebuild from manifest + eight tables only; refuses to overwrite HTML."""
    manifest, tables = load_report_tables(report_dir)
    esc = lambda value: html.escape(str(value))
    pieces = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>S07 robustness study</title>',
              '<style>body{font:15px system-ui;margin:2em;color:#182334}table{border-collapse:collapse;font-size:12px}'
              'td,th{border:1px solid #ccd;padding:5px;text-align:left}section{overflow:auto;margin:2em 0}'
              '.warning{background:#fff2cc;padding:1em}code{word-break:break-all}</style>',
              '<body><h1>S07 development-split robustness</h1>',
              f'<p class="warning">{"COMPLETE" if manifest["complete"] else "PARTIAL"}; '
              f'{esc(manifest["study_kind"])}; {esc(manifest["statuses"])}</p>',
              '<ul>' + ''.join(f'<li>{esc(w)}</li>' for w in manifest['warnings']) + '</ul>',
              '<p>All effects are paired by split and model seed. Seed variation and split-mean variation '
              'are shown separately; no confidence interval assumes nine independent samples. '
              'In paired_summary.csv, within_split gains/losses/ties count seed pairs; '
              'across_split_means gains/losses/ties count complete split means. '
              'split_count counts contributing splits (0 or 1 for within_split rows). '
              'Support-bin class counts and example counts are displayed, including empty bins.</p>']
    for name in ('paired_summary.csv', 'runs.csv', 'support_bins.csv', 'feature_stability.csv'):
        pieces.append(f'<section><h2>{esc(name)}</h2>{tables[name].to_html(index=False, escape=True)}</section>')
    comparisons = []
    run_table = tables['runs.csv']
    successes = run_table[run_table.status == 'success'].set_index(['split_id', 'seed', 'method'])
    for sid in run_table.split_id.unique():
        for other in ('global', 'class_aware'):
            differences = []
            for seed in MODEL_SEEDS:
                gx, alternative = (sid, seed, 'gradient_x_input'), (sid, seed, other)
                if gx in successes.index and alternative in successes.index:
                    differences.append(successes.loc[gx, 'l3_macro_f1'] - successes.loc[alternative, 'l3_macro_f1'])
            comparisons.append({'split_id': sid, 'contrast': 'gradient_x_input minus ' + other,
                                'paired_seeds': len(differences),
                                'mean_delta_macro_f1': np.mean(differences) if differences else None,
                                'within_split_seed_std': np.std(differences) if differences else None})
    pieces.append('<section><h2>Incremental GxI effects versus Chi²</h2>' + pd.DataFrame(comparisons).to_html(
        index=False, escape=True) + '<p>Compare these split-local effects with the checkpoint allocation and measured '
        'shared ranking/refit costs below. There is no predeclared universal cost-benefit threshold.</p></section>')
    pc = tables['per_class.parquet']
    if not pc.empty:
        extremes = pc[(pc.method != 'none') & pc.delta_f1.notna()].sort_values('delta_f1')
        # Full per-class data remains embedded; this view highlights both tails.
        pieces.append('<section><h2>Largest per-class losses and gains (validation support shown)</h2>' +
                      pd.concat([extremes.head(30), extremes.tail(30)]).to_html(index=False, escape=True) + '</section>')
    confusion = tables['confusion_pairs.csv']
    pieces.append('<section><h2>Confusion changes</h2>' + confusion.sort_values(
        'delta_count', ascending=False).head(60).to_html(index=False, escape=True) + '</section>')
    stages = tables['stages.csv']
    if not stages.empty:
        costs = stages.groupby(['split_id', 'method', 'scope'], dropna=False).seconds.sum(min_count=1).reset_index()
        pieces.append('<section><h2>Measured costs (distinct events charged once)</h2>' +
                      costs.to_html(index=False, escape=True) + '</section>')
        attribution_costs = []
        for sid, g in stages.groupby('split_id'):
            reference = g[g.cost_role == 'attribution_checkpoint_and_full_comparator']
            checkpoint_seconds = reference.seconds.sum(min_count=1)
            own = g[g.method == 'gradient_x_input'].seconds.sum(min_count=1)
            attribution_costs.append({'split_id': sid, 'full_seed42_checkpoint_training_and_transform_seconds': checkpoint_seconds,
                                      'gxi_total_measured_seconds': own,
                                      'gxi_standalone_scenario_seconds': own + checkpoint_seconds})
        pieces.append('<section><h2>GxI checkpoint cost allocation</h2><p>Full seed42 is both comparator and '
                      'attribution checkpoint. Its training and input transform are charged once in measured totals; the standalone '
                      'GxI scenario includes it explicitly and is not added to measured totals.</p>' +
                      pd.DataFrame(attribution_costs).to_html(index=False, escape=True) + '</section>')
    pieces.append('<section><h2>Stage resources (unavailable VRAM = null)</h2>' + stages.to_html(
        index=False, escape=True, na_rep='null') + '</section>')
    pieces.append('<section><h2>Interpretation</h2><p>A research recommendation requires a complete matrix, '
                  'paired split effects, rare-class support and resource costs. Positive development deltas '
                  'alone do not establish transferable or industrial gains. No S07 outcome is inferred from '
                  'the historical S06 +0.0139 observation.</p></section>')
    pieces.append('<section><h2>Provenance and formulas</h2><pre>' +
                  esc(json.dumps({k: manifest[k] for k in ('plan_sha256', 'measurement_sources', 'formulas')},
                                 ensure_ascii=False, indent=2)) + '</pre></section>')
    # Self-contained full tables, safely encoded: no external assets/scripts/data requests.
    embedded = {name: json.loads(df.to_json(orient='records', force_ascii=False, double_precision=15))
                for name, df in tables.items()}
    pieces.append('<script type="application/json" id="report-data">' + json.dumps(
        embedded, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026') + '</script>')
    pieces.append('</body></html>')
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        f.write('\n'.join(pieces))
    return str(output)
