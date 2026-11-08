"""Detached consent boundary and one-shot S08 evaluation.

A SHA256 authenticates bytes only relative to the trusted caller's anchor. The
caller (orchestrator) is responsible for obtaining actual consent. Filesystem
seals/claims assume a trusted local operator; they cannot resist ledger deletion
or a hostile process controlling this account. No issuer/service is assumed.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .s07 import digest, fingerprint, read_sealed, require, sealed, sha, write_new, row_identity, group_keys
from .s08_stats import paired_bootstrap, ni_decision


def approval_bindings(output_root):
    """Read-only identity template; NOT an approval record or consent digest."""
    from .s08 import load_plan, verify_training_freeze
    root = Path(output_root)
    plan = load_plan(root)
    freeze = verify_training_freeze(root)
    return {'protocol': fingerprint(root / 'protocol.json'), 'plan': fingerprint(root / 'plan.json'),
            'training_freeze': fingerprint(root / 'training_freeze.json'),
            'sources': plan['sources'], 'runs': freeze['runs'], 'shared': freeze['shared'],
            'label_maps_sha256': digest(plan['label_maps'])}


def _approval(root, record, anchor):
    require(isinstance(anchor, str) and re.fullmatch('[0-9a-f]{64}', anchor) is not None,
            'Unauthenticated approval rejected: separate trusted SHA256 anchor required')
    require(isinstance(record, (str, Path)), 'Separate approval record file required')
    path = Path(record)
    fingerprint(path)  # rejects symlinks
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == anchor, 'Approval authentication digest mismatch')
    decision = json.loads(data)
    require(isinstance(decision, dict) and set(decision) == {
        'schema', 'scope', 'primary_candidate', 'delta_NI', 'practical_rationale',
        'owner', 'consent_evidence', 'one_time_use', 'bindings'}, 'Wrong approval schema/fields')
    require(decision['schema'] == 's08-detached-decision@1'
            and decision['scope'] == 'S08_original_S06_sealed_test_once'
            and decision['primary_candidate'] == 'class_aware'
            and decision['one_time_use'] is True, 'Wrong approval scope/candidate/one-time-use')
    margin = decision['delta_NI']
    require(type(margin) in (int, float) and math.isfinite(margin) and 0 < margin < 1,
            'Exact positive NI margin required; no default')
    for key in ('practical_rationale', 'owner', 'consent_evidence'):
        require(isinstance(decision[key], str) and bool(decision[key].strip()),
                f'Missing approval {key}')
    require(decision['bindings'] == approval_bindings(root), 'Approval frozen identity mismatch')
    return decision, data


def _exclusive_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _parquet_new(frame, path):
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    _exclusive_bytes(path, buffer.getvalue())


def _csv_new(frame, path):
    _exclusive_bytes(path, frame.to_csv(index=False).encode('utf-8'))


def _guard(root, plan, bindings):
    from .s08 import verify_frozen_runtime
    from .s07 import verify_tree
    verify_frozen_runtime(root, plan)
    require(fingerprint(root / 'plan.json') == bindings['plan']
            and fingerprint(root / 'protocol.json') == bindings['protocol'], 'Protocol/plan changed after claim')
    require(fingerprint(root / 'training_freeze.json') == bindings['training_freeze'], 'Freeze changed after claim')
    freeze = read_sealed(root / 'training_freeze.json')
    require(fingerprint(root / 'execution_finished.json') == freeze['execution'], 'Execution receipt changed after claim')
    execution = read_sealed(root / 'execution_finished.json')
    require(fingerprint(root / 'pre_registration.md') == plan['pre_registration']
            and fingerprint(root / 'feasibility.json') == plan['feasibility'], 'Frozen document changed after claim')
    verify_tree(root / 'stages', execution['stages'])
    verify_tree(root / 'runs', bindings['runs'])
    verify_tree(root / 'shared', bindings['shared'])


def _snapshot_checkpoints(root, out, plan, bindings):
    """Inference only uses copies of individually verified bytes, never live files."""
    for run in plan['runs']:
        prefix = run['run_id'] + '/checkpoint/'
        files = {key[len(prefix):]: value for key, value in bindings['runs'].items() if key.startswith(prefix)}
        require(bool(files), 'Missing checkpoint snapshot')
        for name, expected in files.items():
            data = (root / 'runs' / run['run_id'] / 'checkpoint' / name).read_bytes()
            require({'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} == expected,
                    'Checkpoint changed while snapshotting')
            _exclusive_bytes(out / 'checkpoint_snapshots' / run['run_id'] / name, data)


def _aligned_predictions(path, plan, expected_rows, expected_truth, model_hashes):
    """Prove pairing from persisted keys before any positional statistics."""
    frame = pd.read_parquet(path)
    require(list(frame.columns) == ['row_id', 'truth', 'prediction', 'method', 'seed', 'model_sha256'],
            'Prediction table schema mismatch')
    require(len(frame) == 9 * len(expected_rows), 'Incomplete predictions')
    arrays = {}
    for run in plan['runs']:
        part = frame[(frame.method == run['method']) & (frame.seed == run['seed'])]
        require(part.row_id.is_unique and part.row_id.tolist() == expected_rows,
                'Persisted prediction rowIDs/order mismatch')
        require(part.truth.tolist() == expected_truth, 'Persisted prediction truth mismatch')
        require(part.model_sha256.eq(model_hashes[run['run_id']]).all(), 'Prediction model identity mismatch')
        require(part.prediction.isin(plan['label_maps']['L3']).all(), 'Missing/out-of-universe predictions')
        arrays[run['run_id']] = part.prediction.to_numpy()
    return frame, arrays


def _tables(out, plan, truth, row_ids, groups, arrays, margin):
    labels = plan['label_maps']['L3']
    from .s08 import METHODS, SEEDS
    scores, per_class, confusion = {}, [], []
    for run in plan['runs']:
        pred = arrays[run['run_id']]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, pred, labels=labels, zero_division=0)
        scores[run['run_id']] = {'macro_f1': float(f1.mean()), 'accuracy': float(accuracy_score(truth, pred))}
        for i, label in enumerate(labels):
            count = int(support[i])
            per_class.append({'method': run['method'], 'seed': run['seed'], 'L3': label,
                              'precision': float(precision[i]), 'recall': float(recall[i]),
                              'f1': float(f1[i]), 'support': count,
                              'support_bin': '0-1' if count < 2 else '2-4' if count <= 4 else '5-9' if count <= 9 else '10+'})
        errors = pd.DataFrame({'truth': truth, 'prediction': pred})
        errors = errors[errors.truth != errors.prediction].groupby(['truth', 'prediction']).size()
        confusion.extend({'method': run['method'], 'seed': run['seed'], 'truth': a, 'prediction': b, 'count': int(n)}
                         for (a, b), n in errors.items())
    paired = []
    for method in METHODS[1:]:
        for seed in SEEDS:
            full, cand = scores[f'none__seed_{seed}'], scores[f'{method}__seed_{seed}']
            paired.append({'method': method, 'seed': seed, 'full_macro_f1': full['macro_f1'],
                           'candidate_macro_f1': cand['macro_f1'], 'delta': cand['macro_f1']-full['macro_f1'],
                           'full_accuracy': full['accuracy'], 'candidate_accuracy': cand['accuracy'],
                           'role': 'primary' if method == 'class_aware' else 'descriptive_only'})
    _csv_new(pd.DataFrame(paired), out / 'paired_test.csv')
    classes = pd.DataFrame(per_class)
    _parquet_new(classes, out / 'per_class.parquet')
    bins = classes.groupby(['method', 'seed', 'support_bin'], sort=True).agg(
        precision=('precision', 'mean'), recall=('recall', 'mean'), f1=('f1', 'mean'),
        support=('support', 'sum'), classes=('L3', 'size')).reset_index()
    _csv_new(bins, out / 'support_bins.csv')
    _csv_new(pd.DataFrame(confusion, columns=['method', 'seed', 'truth', 'prediction', 'count']), out / 'confusion_pairs.csv')
    result = paired_bootstrap(truth, np.stack([arrays[f'none__seed_{s}'] for s in SEEDS]),
                              np.stack([arrays[f'class_aware__seed_{s}'] for s in SEEDS]),
                              labels=labels, row_ids=row_ids, groups=groups, fixture=plan['config']['fixture'])
    return {**result, **ni_decision(result, margin)}, scores


def evaluate_authorized(output_root, approval, trusted_anchor, *, report_dir=None):
    from .s08 import load_plan, verify_training_freeze
    from .s08_report import _run_rows, RUN_COLUMNS, STAGE_COLUMNS
    from .artifact import load_artifact
    from .experiment import _Measurement
    from .train import predict

    root = Path(output_root)
    # Authenticate before loading any source, including structural data.
    decision, approval_bytes = _approval(root, approval, trusted_anchor)
    plan = load_plan(root)
    verify_training_freeze(root)  # all nine models/maps/features before any label read
    rows = _run_rows(root, plan)
    for row in rows:
        row.update(inference_status='pending', inference_reason=None)
    stages = [read_sealed(p) for p in sorted((root / 'stages').glob('*.json'))]
    out = Path(report_dir) if report_dir else root / ('report-evaluation-' + uuid.uuid4().hex)
    require(not out.exists() and out.resolve() != (root / 'report').resolve(), 'Use fresh versioned evaluation report')
    bindings = decision['bindings']
    # O_EXCL is the concurrency boundary. Never remove/replace this claim,
    # including after failure; a second invocation must never read labels.
    claim = {'schema': 's08-use-claim@1', 'approval_sha256': trusted_anchor,
             'bindings_sha256': digest(bindings), 'report_dir': str(out.resolve()),
             'use_consumed': True, 'retry_or_resume': False}
    _exclusive_bytes(root / 'evaluation_claim.json', json.dumps(sealed(claim), allow_nan=False).encode())
    # Anything after this point permanently consumes use, even an interrupted write.
    active_run = None
    try:
        out.mkdir(parents=True, exist_ok=False)
        _exclusive_bytes(out / 'approval_decision.json', approval_bytes)
        _guard(root, plan, bindings)
        _snapshot_checkpoints(root, out, plan, bindings)
        _guard(root, plan, bindings)
        # Read a verified byte snapshot, so mutation between hash/read cannot substitute labels.
        info = plan['sources']['sealed_test']
        _guard(root, plan, bindings)
        data = Path(info['path']).read_bytes()
        require({'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} ==
                {key: info[key] for key in ('sha256', 'bytes')}, 'Sealed source changed during read')
        frame = pd.read_parquet(BytesIO(data), columns=['row_id', 'name', 'L3'])
        require(row_identity(frame[['row_id', 'name']]) ==
                {k: info[k] for k in row_identity(frame[['row_id', 'name']])}, 'Sealed row identity changed')
        require(frame.row_id.is_unique and not frame[['row_id', 'name', 'L3']].isna().any().any(), 'Invalid sealed rows')
        require(frame.L3.isin(plan['label_maps']['L3']).all(), 'Sealed taxonomy mismatch')
        row_ids, truth = frame.row_id.tolist(), frame.L3.tolist()
        groups = group_keys(frame)
        predictions, model_hashes = [], {}
        for run in plan['runs']:
            active_run = next(row for row in rows if row['run_id'] == run['run_id'])
            _guard(root, plan, bindings)
            snap = out / 'checkpoint_snapshots' / run['run_id']
            # verify snapshot bytes immediately before loading; loaded objects are isolated
            from .s07 import verify_tree
            prefix = run['run_id'] + '/checkpoint/'
            expected = {key[len(prefix):]: value for key, value in bindings['runs'].items() if key.startswith(prefix)}
            verify_tree(snap, expected)
            art = load_artifact(str(snap), torch.device('cpu'))
            verify_tree(snap, expected)
            require(art.maps.to_dict() == plan['label_maps'], 'Snapshot maps mismatch')
            require(art.meta['s08']['feature_id'] == read_sealed(root / 'runs' / run['run_id'] / 'checkpoint_receipt.json')['provenance']['feature_id'], 'Snapshot feature mismatch')
            model_hash = expected['model.pt']['sha256']
            model_hashes[run['run_id']] = model_hash
            _guard(root, plan, bindings)
            measurement = _Measurement(torch.device('cpu'), 'S08 frozen raw top-1 inference; transform included')
            inference_status = 'failed'
            try:
                with measurement:
                    X = art.features.transform(frame.name.tolist(), chunk_rows=plan['config']['chunk_rows'])
                    raw, _ = predict(art.model, X, torch.device('cpu'),
                                     plan['training']['train']['eval_batch_size'], 1)
                    pred = art.maps.decode('L3', raw.l3_top1)
                inference_status = 'success'
            finally:
                stage = measurement.stage_row(run['run_id'], 'inference', inference_status)
                stage.update(method=run['method'], scope='run', charge_count=1)
                stages.append(stage)
            _guard(root, plan, bindings)
            part = pd.DataFrame({'row_id': row_ids, 'truth': truth, 'prediction': pred,
                                 'method': run['method'], 'seed': run['seed'], 'model_sha256': model_hash})
            part_path = out / 'prediction_parts' / (run['run_id'] + '.parquet')
            _parquet_new(part, part_path)
            predictions.append((part_path, fingerprint(part_path)))
            active_run['inference_status'] = 'success'
            active_run = None
        parts = []
        for path, expected in predictions:
            require(fingerprint(path) == expected, 'Per-run predictions changed')
            parts.append(pd.read_parquet(path))
            require(fingerprint(path) == expected, 'Per-run predictions changed during read')
        _parquet_new(pd.concat(parts, ignore_index=True), out / 'predictions_test.parquet')
        _guard(root, plan, bindings)
        stash_hash = fingerprint(out / 'predictions_test.parquet')
        _, arrays = _aligned_predictions(out / 'predictions_test.parquet', plan, row_ids, truth, model_hashes)
        require(fingerprint(out / 'predictions_test.parquet') == stash_hash, 'Prediction stash changed during reload')
        result, scores = _tables(out, plan, truth, row_ids, groups, arrays, decision['delta_NI'])
        for row in rows:
            row['test_macro_f1'] = scores[row['run_id']]['macro_f1']
        result.update(gate='evaluated', primary_candidate='class_aware', margin_status='approved_detached',
                      primary_status='approved_detached', sealed_inference=True, decision=result['status'])
        _guard(root, plan, bindings)
    except Exception as exc:
        for row in rows:
            row['test_macro_f1'] = None
        if active_run is not None:
            active_run.update(inference_status='failed', inference_reason=f'{type(exc).__name__}: {exc}')
        result = {'status': 'evaluation_blocked', 'gate': 'failed_consumed', 'decision': None,
                  'delta_NI': decision['delta_NI'], 'delta': None, 'lower_bound': None,
                  'error': f'{type(exc).__name__}: {exc}', 'use_consumed': True, 'retry_or_resume': False,
                  'sealed_access_may_have_occurred': True, 'primary_candidate': 'class_aware'}
        if not out.is_dir():
            raise  # claim itself is the durable failure receipt
    _csv_new(pd.DataFrame(rows, columns=RUN_COLUMNS + ['inference_status', 'inference_reason']), out / 'runs.csv')
    _csv_new(pd.DataFrame(stages).reindex(columns=STAGE_COLUMNS), out / 'stages.csv')
    write_new(out / 'ni_decision.json', sealed(result))
    from .s08 import TEST_FILES, LIMITATIONS
    table_names = ['runs.csv', 'stages.csv', 'ni_decision.json', 'approval_decision.json']
    table_names += [f for f in TEST_FILES if (out / f).exists()]
    manifest = {'schema': 's08-evaluated-report@1', 'gate': result['gate'],
                'approval': decision, 'approval_sha256': trusted_anchor, 'claim': claim,
                'training_complete': True, 'confirmatory_complete': result['gate'] == 'evaluated',
                'study_kind': 'synthetic' if plan['config']['fixture'] else 'ODS_internal_holdout',
                'expected_runs': plan['runs'], 'limitations': [v for v in LIMITATIONS if not v.startswith('Primary class_aware')],
                'tables': {name: fingerprint(out / name) for name in table_names}}
    write_new(out / 'manifest.json', sealed(manifest))
    render_evaluated(out, out / 'report.html', manifest)
    return {'report_dir': str(out), 'gate': result['gate'], 'status': result['status'],
            'use_consumed': True, 'failed': result['gate'] != 'evaluated'}


def render_evaluated(base, output_html, manifest):
    """Deterministic offline rebuild; reads only manifest-listed reporting tables."""
    from .s08 import TEST_FILES
    base = Path(base)
    required = {'runs.csv', 'stages.csv', 'ni_decision.json', 'approval_decision.json'}
    require(required <= set(manifest['tables']) <= required | set(TEST_FILES), 'Wrong evaluated table set')
    for name, expected in manifest['tables'].items():
        require(fingerprint(base / name) == expected, 'Report table integrity mismatch')
    require(sha(base / 'approval_decision.json') == manifest['approval_sha256'], 'Approval anchor mismatch')
    from .s07 import read_json
    approval = read_json(base / 'approval_decision.json')
    require(approval == manifest['approval'] and approval['bindings'] == manifest['approval']['bindings'], 'Detached decision changed')
    decision = read_sealed(base / 'ni_decision.json')
    require(decision['delta_NI'] == approval['delta_NI'], 'Decision margin mismatch')
    success = manifest['gate'] == 'evaluated'
    require(manifest['confirmatory_complete'] is success and manifest['training_complete'] is True,
            'Wrong evaluated completion state')
    require(decision['gate'] == manifest['gate'], 'Decision gate mismatch')
    if success:
        require(set(manifest['tables']) == required | set(TEST_FILES), 'Incomplete evaluated report')
        require(decision['status'] == ni_decision(decision, approval['delta_NI'])['status']
                and decision['decision'] == decision['status'], 'Invalid NI decision')
    else:
        require(manifest['gate'] == 'failed_consumed' and decision['status'] == 'evaluation_blocked'
                and decision['delta'] is None and decision['lower_bound'] is None,
                'Invalid failed/consumed decision')
    from .s08_report import RUN_COLUMNS, STAGE_COLUMNS
    runs = pd.read_csv(base / 'runs.csv')
    stages = pd.read_csv(base / 'stages.csv')
    require(list(runs.columns) == RUN_COLUMNS + ['inference_status', 'inference_reason']
            and list(stages.columns) == STAGE_COLUMNS, 'Evaluated schema mismatch')
    require(len(runs) == 9 and runs.run_id.is_unique and runs.status.eq('success').all()
            and runs.validation_macro_f1.isna().all(), 'Wrong frozen training rows')
    for row, expected in zip(runs.to_dict('records'), manifest['expected_runs']):
        require(all(row[k] == expected[k] for k in ('run_id', 'method', 'seed', 'n_char', 'n_word')),
                'Evaluated matrix changed')
        require(row['inference_status'] in {'success', 'failed', 'pending'}, 'Invalid inference status')
    require(stages.charge_count.eq(1).all(), 'Repeated stage charging')
    require(manifest['claim']['use_consumed'] is True and manifest['claim']['retry_or_resume'] is False
            and manifest['claim']['approval_sha256'] == manifest['approval_sha256']
            and manifest['claim']['bindings_sha256'] == digest(approval['bindings']), 'Invalid claim binding')
    if success:
        require(runs.inference_status.eq('success').all() and runs.test_macro_f1.notna().all(),
                'Incomplete evaluated runs')
        paired = pd.read_csv(base / 'paired_test.csv')
        require(len(paired) == 6 and not paired[['method', 'seed']].duplicated().any(), 'Invalid paired table')
        for row in paired.to_dict('records'):
            require(row['method'] in ('class_aware', 'global') and row['seed'] in (42,43,44), 'Wrong paired method/seed')
            full = float(runs.loc[(runs.method == 'none') & (runs.seed == row['seed']), 'test_macro_f1'].iloc[0])
            candidate = float(runs.loc[(runs.method == row['method']) & (runs.seed == row['seed']), 'test_macro_f1'].iloc[0])
            require(np.isclose(row['full_macro_f1'], full, rtol=0, atol=1e-14)
                    and np.isclose(row['candidate_macro_f1'], candidate, rtol=0, atol=1e-14)
                    and np.isclose(row['delta'], candidate-full, rtol=0, atol=1e-14), 'Unpaired metric table')
            require(row['role'] == ('primary' if row['method'] == 'class_aware' else 'descriptive_only'), 'Wrong inferential role')
        require(np.isclose(paired.loc[paired.method == 'class_aware', 'delta'].mean(), decision['delta'],
                           rtol=0, atol=1e-14), 'Primary delta mismatch')
        from .s08_stats import BOOTSTRAP
        require(all(decision['recipe'][key] == value for key, value in BOOTSTRAP.items()), 'Bootstrap contract changed')
    else:
        require(runs.test_macro_f1.isna().all(), 'Failed evaluation cannot publish final quality')
    sections = []
    for name in sorted(manifest['tables']):
        if name.endswith('.csv'):
            table = pd.read_csv(base / name)
        elif name.endswith('.parquet'):
            table = pd.read_parquet(base / name)
        else:
            continue
        sections.append('<h2>' + html.escape(name) + '</h2>' + table.to_html(index=False, escape=True, na_rep='null'))
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>S08 evaluation</title>'
                '<h1>S08 ' + html.escape(decision['status']) + '</h1><p>One-time use consumed permanently. '
                'Global is descriptive only. Future adaptive confirmation requires new independent labels.</p>'
                '<pre>' + html.escape(json.dumps(decision, sort_keys=True, indent=2)) + '</pre><ul>'
                + ''.join('<li>' + html.escape(v) + '</li>' for v in manifest['limitations']) + '</ul>'
                + ''.join(sections) + '</html>')
    _exclusive_bytes(output_html, document.encode('utf-8'))
    return str(output_html)
