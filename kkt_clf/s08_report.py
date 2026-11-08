"""Honest pre-unseal S08 tables and offline table-only HTML regeneration."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .s07 import (digest, fingerprint, read_sealed, require, sealed, tree, verify_tree, write_new)
from .s08 import (LIMITATIONS, METHODS, TEST_FILES, S08Config, _provenance, load_plan,
                  verify_checkpoint, verify_selected, verify_training_freeze)
from .vectorizer import load_vectorizer_artifact

RUN_COLUMNS = ['run_id', 'method', 'seed', 'n_char', 'n_word', 'status', 'reason',
               'model_bytes', 'validation_macro_f1', 'test_macro_f1']
STAGE_COLUMNS = ['run_id', 'method', 'scope', 'stage', 'status', 'seconds',
                 'peak_rss_bytes', 'peak_vram_bytes', 'charge_count']


def _run_rows(root, plan):
    rows = []
    finished_path = root / 'execution_finished.json'
    finished = read_sealed(finished_path) if finished_path.exists() else None
    if finished:
        require(finished['plan_sha256'] == digest(plan), 'Execution receipt plan mismatch')
        for name in ('runs', 'shared', 'stages'):
            actual = tree(root / name) if (root / name).exists() else {}
            require(actual == finished[name], 'Execution outputs changed after completion')
    full = None
    for run in plan['runs']:
        row = {k: run[k] for k in ('run_id', 'method', 'seed', 'n_char', 'n_word')}
        row.update(status='pending', reason=None, model_bytes=None,
                   validation_macro_f1=None, test_macro_f1=None)
        base = root / 'runs' / run['run_id']
        if (base / 'result.json').exists():
            result = read_sealed(base / 'result.json')
            status = result.get('status')
            require(status in ('success', 'failed', 'skipped'), 'Invalid result status')
            require(finished is not None, 'Unfinished execution cannot publish successful/stale results')
            require(run['run_id'] in finished['summary'][status], 'Result/summary status disagreement')
            row.update(status=status, reason=result.get('reason'))
            if status == 'success':
                if full is None:
                    full = load_vectorizer_artifact(S08Config(**plan['config']).full_tfidf)
                features = full if run['method'] == 'none' else verify_selected(root, plan, run['method'], full)
                fid = plan['sources']['features']['sha256'] if run['method'] == 'none' else fingerprint(
                    root / 'shared' / run['method'] / 'tfidf' / 'features.joblib')['sha256']
                require(result['provenance'] == _provenance(plan, run, fid), 'Stale run provenance')
                require(result['checkpoint_receipt'] == fingerprint(base / 'checkpoint_receipt.json')
                        and result['training'] == fingerprint(base / 'training.json'), 'Run evidence changed')
                require(result['test_macro_f1'] is None and result['validation_macro_f1'] is None,
                        'Unexpected pre-unseal quality metric')
                verify_checkpoint(root, plan, run, features.builder, fid)
                actual_bytes = (base / 'checkpoint' / 'model.pt').stat().st_size
                require(actual_bytes == result['model_bytes'], 'Model bytes mismatch')
                row['model_bytes'] = actual_bytes
            else:
                require(result['plan_sha256'] == digest(plan), 'Failed/skipped run plan mismatch')
        elif finished:
            require(run['run_id'] in finished['summary']['pending'], 'Missing result falsely marked completed')
        elif base.exists():
            row.update(status='incomplete', reason='Interrupted/in-progress execution; no success receipt')
        rows.append(row)
    return rows


def build_report(output_root, report_dir=None):
    root = Path(output_root)
    require(not (root / 'evaluation_claim.json').exists(),
            'Evaluation use already consumed; rebuild its versioned report, never publish a pending state')
    plan = load_plan(root)
    # Reject unexpected sealed outputs rather than accidentally legitimizing them.
    for p in [root / 'report', Path(report_dir) if report_dir else root / 'report']:
        require(not any((p / f).exists() for f in TEST_FILES), 'Unauthorized test-dependent output detected')
    rows = _run_rows(root, plan)
    complete = all(r['status'] == 'success' for r in rows)
    freeze = verify_training_freeze(root) if complete else None
    stages = [read_sealed(p) for p in sorted((root / 'stages').glob('*.json'))]
    require(all(s['charge_count'] == 1 and s['method'] in METHODS for s in stages), 'Invalid stage accounting')
    out = Path(report_dir) if report_dir else root / 'report'
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows, columns=RUN_COLUMNS).to_csv(out / 'runs.csv', index=False)
    pd.DataFrame(stages).reindex(columns=STAGE_COLUMNS).to_csv(out / 'stages.csv', index=False)
    decision = {'status': 'not_run_awaiting_approval', 'delta_NI': None, 'margin_status': 'unapproved',
                'primary_candidate': 'class_aware', 'primary_status': 'unapproved', 'delta': None,
                'lower_bound': None, 'decision': None, 'sealed_inference': False,
                'estimator': 'mean of seed-paired fixed-L3 raw top-1 macro-F1 differences',
                'gate': 'awaiting_user_approval', 'approval_authentication': 'not_provisioned'}
    write_new(out / 'ni_decision.json', sealed(decision))
    manifest = {'schema': 's08-report@1', 'gate': 'awaiting_user_approval', 'approval': None,
                'training_complete': complete, 'confirmatory_complete': False,
                'plan_sha256': digest(plan), 'protocol': plan['protocol'], 'plan': fingerprint(root / 'plan.json'),
                'training_freeze': fingerprint(root / 'training_freeze.json') if freeze else None,
                'study_kind': 'synthetic' if plan['config']['fixture'] else 'ODS_internal_holdout',
                'expected_runs': plan['runs'], 'limitations': LIMITATIONS,
                'statuses': {s: sum(r['status'] == s for r in rows)
                             for s in ('success', 'failed', 'skipped', 'pending', 'incomplete')},
                'tables': {f: fingerprint(out / f) for f in ('runs.csv', 'stages.csv', 'ni_decision.json')},
                'feasibility': read_sealed(root / 'feasibility.json')}
    write_new(out / 'manifest.json', sealed(manifest))
    render_report(out, out / 'report.html')
    return {'report_dir': str(out), 'training_complete': complete, 'confirmatory_complete': False,
            'gate': 'awaiting_user_approval', 'statuses': manifest['statuses']}


def render_report(report_dir, output_html):
    """Only manifest, two CSV tables and ni_decision.json; no run/shared/source reads."""
    base = Path(report_dir)
    manifest = read_sealed(base / 'manifest.json')
    require(manifest['schema'] in ('s08-report@1', 's08-evaluated-report@1'), 'Wrong report schema')
    if manifest['schema'] == 's08-evaluated-report@1':
        from .s08_evaluation import render_evaluated
        return render_evaluated(base, output_html, manifest)
    require(set(manifest['tables']) == {'runs.csv', 'stages.csv', 'ni_decision.json'}, 'Wrong pending report table set')
    for name, expected in manifest['tables'].items():
        require(fingerprint(base / name) == expected, 'Report table integrity mismatch')
    require(not any((base / f).exists() for f in TEST_FILES), 'Pending report has test-dependent files')
    runs, stages = pd.read_csv(base / 'runs.csv'), pd.read_csv(base / 'stages.csv')
    decision = read_sealed(base / 'ni_decision.json')
    require(list(runs.columns) == RUN_COLUMNS and list(stages.columns) == STAGE_COLUMNS, 'Report schema mismatch')
    require(len(runs) == 9 and runs.run_id.is_unique, 'Wrong/duplicate report runs')
    for row, expected in zip(runs.to_dict('records'), manifest['expected_runs']):
        require(all(row[k] == expected[k] for k in ('run_id', 'method', 'seed', 'n_char', 'n_word')),
                'Report matrix changed')
        require(row['status'] in {'success', 'failed', 'skipped', 'pending', 'incomplete'}, 'Unknown run status')
    require(all(int((runs.status == s).sum()) == n for s, n in manifest['statuses'].items()), 'Status totals mismatch')
    require(runs[['test_macro_f1', 'validation_macro_f1']].isna().all().all(), 'Pre-unseal metrics must be null')
    require(manifest['training_complete'] == bool((runs.status == 'success').all())
            and manifest['confirmatory_complete'] is False, 'Misleading completion state')
    require(decision['status'] == 'not_run_awaiting_approval' and decision['delta_NI'] is None
            and decision['delta'] is None and decision['lower_bound'] is None and decision['decision'] is None
            and decision['margin_status'] == decision['primary_status'] == 'unapproved'
            and decision['sealed_inference'] is False, 'Invalid pending NI decision')
    require(manifest['gate'] == decision['gate'] == 'awaiting_user_approval' and manifest['approval'] is None,
            'Misleading approval state')
    require(manifest['limitations'] == LIMITATIONS, 'Mandatory limitations changed')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>S08 prepared study</title>'
                '<style>body{font:15px system-ui;margin:2em;max-width:1200px}table{border-collapse:collapse;font-size:12px}'
                'td,th{border:1px solid #ccc;padding:.4em}pre{white-space:pre-wrap}</style>'
                '<h1>S08 — awaiting user approval</h1><p>Holdout CLOSED. NI not run. '
                'Primary class_aware and practical margin remain UNAPPROVED. delta_NI=null. '
                'Training completion does not establish non-inferiority.</p>'
                f'<p>Study kind: {html.escape(manifest["study_kind"])}</p>'
                '<h2>Limitations</h2><ul>' + ''.join('<li>'+html.escape(x)+'</li>' for x in manifest['limitations'])
                + '</ul><h2>Training runs</h2>' + runs.to_html(index=False, na_rep='null', escape=True)
                + '<h2>Measured stages (shared charged once)</h2>' + stages.to_html(index=False, na_rep='null', escape=True)
                + '<h2>Exploratory feasibility (simulated, not holdout scores)</h2><pre>'
                + html.escape(__import__('json').dumps(manifest['feasibility'], ensure_ascii=False, indent=2))
                + '</pre></html>')
    target = Path(output_html)
    with target.open('x', encoding='utf-8') as f:
        f.write(document)
    return str(target)
