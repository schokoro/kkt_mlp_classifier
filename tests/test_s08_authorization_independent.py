"""Detached authorization mechanics on a SEPARATE generated labeled holdout.

No test creates production consent or reads the original S06 holdout. Anchors in
this file represent a synthetic trusted caller, never actual stakeholder consent.
"""
from __future__ import annotations

import builtins
import dataclasses
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

from kkt_clf import s08, s08_report
from tests.test_s08_fixtures import sources, file_sha, write_json, reseal
from tests.test_s08_integrity_independent import synthetic, forbid
from tests.test_s08_statistics_independent import oracle


@pytest.fixture
def labeled_study(synthetic):
    """New sources and pins, before planning; preserve the label-free sealed file."""
    cfg, root, src = synthetic
    holdout = pd.DataFrame({'row_id': list(range(910, 918)),
        'name': [f'{label} unseen synthetic item{j}' for label in ('alpha', 'beta') for j in range(4)],
        'L3': ['alpha'] * 4 + ['beta'] * 4})
    path = root.parent / 'generated_labeled_holdout.parquet'
    holdout.to_parquet(path, index=False)
    manifest = json.loads(src['manifest'].read_text())
    manifest['parts']['sealed_test'] = {'file': str(path), 'sha256': file_sha(path),
        'bytes': path.stat().st_size, 'n_rows': len(holdout), 'n_groups': len(holdout)}
    mp = root.parent / 'generated_manifest.json'
    write_json(mp, manifest)
    protocol = json.loads(src['source_protocol'].read_text())
    protocol['splits'] = manifest['parts']
    pp = root.parent / 'generated_protocol.json'
    write_json(pp, protocol)
    pins = {**src['pins'], 'manifest': file_sha(mp), 'protocol': file_sha(pp)}
    cfg = dataclasses.replace(cfg, manifest=str(mp), protocol=str(pp), fixture_pins=pins)
    plan = s08.write_plan(cfg, root)
    summary = s08.execute(root)
    assert len(summary['success']) == 9, summary
    # Original negative fixture stays label-free, byte-identical and separate.
    assert list(pd.read_parquet(src['parts']['sealed_test']['file']).columns) == ['row_id', 'name']
    return cfg, root, src, plan, holdout, path


def decision_record(study):
    """Independent decision construction from frozen identities, not a consent API."""
    _, root, _, plan, _, _ = study
    freeze = json.loads((root / 'training_freeze.json').read_text())
    return {'schema': 's08-detached-decision@1', 'scope': 'S08_original_S06_sealed_test_once',
        'primary_candidate': 'class_aware', 'delta_NI': .023,
        'practical_rationale': 'Generated mechanics test only; no production practical judgment',
        'owner': 'synthetic independent test caller',
        'consent_evidence': 'synthetic fixture; NOT original S06 authorization', 'one_time_use': True,
        'bindings': {'protocol': s08.fingerprint(root / 'protocol.json'),
            'plan': s08.fingerprint(root / 'plan.json'),
            'training_freeze': s08.fingerprint(root / 'training_freeze.json'),
            'sources': plan['sources'], 'runs': freeze['runs'], 'shared': freeze['shared'],
            'label_maps_sha256': s08.digest(plan['label_maps'])}}


def save_decision(study, record=None):
    root = study[1]
    path = root.parent / 'SYNTHETIC_decision_only.json'
    write_json(path, decision_record(study) if record is None else record)
    return path, file_sha(path)


def forbid_quality(monkeypatch):
    import kkt_clf.train as train
    monkeypatch.setattr(train, 'predict', forbid)
    original = pd.read_parquet
    def guarded(path, *args, **kwargs):
        columns = kwargs.get('columns')
        if not isinstance(path, (str, Path)) or Path(path).name == 'generated_labeled_holdout.parquet':
            if columns != ['row_id', 'name']:
                pytest.fail('Unauthorized quality/label read')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', guarded)


@pytest.mark.parametrize('anchor', [None, '', '0' * 64, True, 'g' * 64])
def test_valid_looking_record_requires_separate_exact_trusted_anchor(labeled_study, monkeypatch, anchor):
    path, _ = save_decision(labeled_study)
    forbid_quality(monkeypatch)
    root = labeled_study[1]
    before = {str(p): file_sha(p) for p in root.rglob('*') if p.is_file()}
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        s08.evaluate(root, approval=path, trusted_anchor=anchor)
    assert not (root / 'evaluation_claim.json').exists()
    assert before == {str(p): file_sha(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['margin_absent', 'margin_null', 'margin_zero', 'margin_bool',
    'primary', 'scope', 'one_time', 'owner', 'rationale', 'evidence', 'plan', 'protocol',
    'training_freeze', 'source', 'model_weights', 'label_maps', 'feature_vocab'])
def test_authentic_but_semantically_wrong_decisions_cannot_read_labels(labeled_study, monkeypatch, damage):
    record = decision_record(labeled_study)
    if damage == 'margin_absent':
        record.pop('delta_NI')
    elif damage.startswith('margin_'):
        record['delta_NI'] = {'margin_null': None, 'margin_zero': 0, 'margin_bool': True}[damage]
    elif damage in ('primary', 'scope', 'one_time', 'owner', 'rationale', 'evidence'):
        key, value = {'primary': ('primary_candidate', 'global'), 'scope': ('scope', 'other'),
            'one_time': ('one_time_use', False), 'owner': ('owner', ''),
            'rationale': ('practical_rationale', ' '), 'evidence': ('consent_evidence', '')}[damage]
        record[key] = value
    else:
        bindings = record['bindings']
        if damage in ('plan', 'protocol', 'training_freeze'):
            bindings[damage]['sha256'] = '0' * 64
        elif damage == 'source':
            bindings['sources']['sealed_test']['sha256'] = '0' * 64
        elif damage == 'label_maps':
            bindings['label_maps_sha256'] = '0' * 64
        else:
            group = 'runs' if damage == 'model_weights' else 'shared'
            key = next(k for k in bindings[group] if k.endswith('model.pt' if group == 'runs' else 'features.joblib'))
            bindings[group][key]['sha256'] = '0' * 64
    path, anchor = save_decision(labeled_study, record)
    forbid_quality(monkeypatch)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        s08.evaluate(labeled_study[1], approval=path, trusted_anchor=anchor)
    assert not (labeled_study[1] / 'evaluation_claim.json').exists()


@pytest.mark.parametrize('target', ['plan.json', 'protocol.json', 'training_freeze.json',
    'runs/none__seed_42/checkpoint/model.pt', 'runs/none__seed_42/checkpoint/label_maps.json',
    'shared/class_aware/tfidf/features.joblib'])
def test_approved_identity_does_not_authorize_changed_frozen_files(labeled_study, monkeypatch, target):
    path, anchor = save_decision(labeled_study)
    root = labeled_study[1]
    with (root / target).open('ab') as stream:
        stream.write(b'changed after synthetic decision')
    forbid_quality(monkeypatch)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        s08.evaluate(root, approval=path, trusted_anchor=anchor)
    assert not (root / 'evaluation_claim.json').exists()


def assert_evaluated(study, outcome):
    _, root, _, plan, holdout, _ = study
    assert outcome['gate'] == 'evaluated', outcome
    out = Path(outcome['report_dir'])
    frame = pd.read_parquet(out / 'predictions_test.parquet')
    assert len(frame) == 9 * len(holdout)
    arrays = {}
    paired = pd.read_csv(out / 'paired_test.csv')
    per_class = pd.read_parquet(out / 'per_class.parquet')
    bins = pd.read_csv(out / 'support_bins.csv')
    confusion = pd.read_csv(out / 'confusion_pairs.csv')
    labels = plan['label_maps']['L3']
    for run in plan['runs']:
        part = frame[(frame.method == run['method']) & (frame.seed == run['seed'])]
        assert part.row_id.tolist() == holdout.row_id.tolist() and part.row_id.is_unique
        assert part.truth.tolist() == holdout.L3.tolist()
        assert set(part.model_sha256) == {file_sha(root / 'runs' / run['run_id'] / 'checkpoint' / 'model.pt')}
        pred = part.prediction.to_numpy()
        arrays[run['run_id']] = pred
        cls = per_class[(per_class.method == run['method']) & (per_class.seed == run['seed'])]
        p, r, f, support = precision_recall_fscore_support(holdout.L3, pred, labels=labels, zero_division=0)
        assert cls.L3.tolist() == labels
        for name, expected in [('precision', p), ('recall', r), ('f1', f), ('support', support)]:
            np.testing.assert_allclose(cls[name], expected, atol=1e-15)
        b = bins[(bins.method == run['method']) & (bins.seed == run['seed'])]
        assert b.support.sum() == len(holdout) and b.classes.sum() == len(labels)
        assert b.f1.iloc[0] == pytest.approx(f.mean())
        errors = confusion[(confusion.method == run['method']) & (confusion.seed == run['seed'])]
        assert errors['count'].sum() == int(np.sum(pred != holdout.L3.to_numpy()))
        for row in errors.itertuples():
            assert row.count == int(np.sum((holdout.L3.to_numpy() == row.truth) & (pred == row.prediction)))
    for row in paired.itertuples():
        full = arrays[f'none__seed_{row.seed}']
        candidate = arrays[f'{row.method}__seed_{row.seed}']
        f = f1_score(holdout.L3, full, labels=labels, average='macro', zero_division=0)
        c = f1_score(holdout.L3, candidate, labels=labels, average='macro', zero_division=0)
        assert row.full_macro_f1 == pytest.approx(f)
        assert row.candidate_macro_f1 == pytest.approx(c)
        assert row.delta == pytest.approx(c - f)
        assert row.full_accuracy == pytest.approx(accuracy_score(holdout.L3, full))
        assert row.candidate_accuracy == pytest.approx(accuracy_score(holdout.L3, candidate))
        assert row.role == ('primary' if row.method == 'class_aware' else 'descriptive_only')
    full = np.stack([arrays[f'none__seed_{s}'] for s in (42, 43, 44)])
    cand = np.stack([arrays[f'class_aware__seed_{s}'] for s in (42, 43, 44)])
    draws, differences = oracle(holdout.L3.to_numpy(), full, cand, labels)
    decision = json.loads((out / 'ni_decision.json').read_text())
    assert decision['delta_NI'] == .023  # no implicit .01 planning default
    assert decision['delta'] == pytest.approx(differences.mean(), abs=1e-15)
    assert decision['lower_bound'] == pytest.approx(np.quantile(draws, .025, method='linear'), abs=1e-15)
    assert decision['status'] == ('non_inferior' if decision['lower_bound'] > -.023 else 'not_established')
    stages = pd.read_csv(out / 'stages.csv')
    inference = stages[stages.stage == 'inference']
    assert len(inference) == 9 and (inference.status == 'success').all()
    assert (inference.seconds >= 0).all() and (inference.charge_count == 1).all()
    refits = stages[stages.stage == 'refit']
    assert len(refits) == 2 and (refits.charge_count == 1).all()
    assert len(confusion.columns) >= 5
    return out


def test_generated_authorized_evaluation_completes_and_replay_is_impossible(labeled_study, monkeypatch):
    path, anchor = save_decision(labeled_study)
    root = labeled_study[1]
    frozen = {str(p): file_sha(p) for p in root.rglob('*') if p.is_file()}
    outcome = s08.evaluate(root, approval=path, trusted_anchor=anchor)
    out = assert_evaluated(labeled_study, outcome)
    assert all(file_sha(p) == digest for p, digest in frozen.items())
    protocol = json.loads((root / 'protocol.json').read_text())
    assert protocol['delta_NI'] is None and protocol['primary_status'] == 'unapproved'
    prior = {str(p): file_sha(p) for p in out.rglob('*') if p.is_file()}
    forbid_quality(monkeypatch)
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        s08.evaluate(root, approval=path, trusted_anchor=anchor)
    assert prior == {str(p): file_sha(p) for p in out.rglob('*') if p.is_file()}
    with pytest.raises((RuntimeError, ValueError)):
        s08_report.build_report(root, root / 'misleading_new_pending_report')


def test_atomic_local_claim_has_exactly_one_winner(tmp_path):
    from kkt_clf.s08_evaluation import _exclusive_bytes
    path = tmp_path / 'claim.json'
    def compete(i):
        try:
            _exclusive_bytes(path, str(i).encode())
            return i
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        winners = [v for v in pool.map(compete, range(16)) if v is not None]
    assert len(winners) == 1 and path.read_text() == str(winners[0])


@pytest.mark.parametrize('damage', ['weights', 'holdout', 'failure_before_read'])
def test_post_claim_mutation_or_failure_consumes_use_and_prevents_label_read(labeled_study, monkeypatch, damage):
    import kkt_clf.s08_evaluation as evaluation
    path, anchor = save_decision(labeled_study)
    _, root, _, _, _, holdout_path = labeled_study
    original = evaluation._exclusive_bytes
    def claimed(target, data):
        original(target, data)
        if Path(target) == root / 'evaluation_claim.json':
            if damage == 'failure_before_read':
                return
            victim = holdout_path if damage == 'holdout' else root / 'runs/none__seed_42/checkpoint/model.pt'
            with victim.open('ab') as stream:
                stream.write(b'mutated after atomic claim')
    monkeypatch.setattr(evaluation, '_exclusive_bytes', claimed)
    if damage == 'failure_before_read':
        def interrupted(*args, **kwargs):
            raise RuntimeError('synthetic interruption before any labels')
        monkeypatch.setattr(evaluation, '_snapshot_checkpoints', interrupted)
    forbid_quality(monkeypatch)
    result = s08.evaluate(root, approval=path, trusted_anchor=anchor)
    assert result['gate'] == 'failed_consumed' and result['status'] == 'evaluation_blocked'
    assert (root / 'evaluation_claim.json').exists()
    out = Path(result['report_dir'])
    decision = json.loads((out / 'ni_decision.json').read_text())
    assert decision['delta'] is decision['lower_bound'] is None
    assert decision['use_consumed'] is True and decision['retry_or_resume'] is False
    assert not (out / 'predictions_test.parquet').exists()
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        s08.evaluate(root, approval=path, trusted_anchor=anchor)


@pytest.mark.parametrize('damage', ['reordered', 'missing', 'duplicate'])
def test_persisted_predictions_must_prove_row_identity_alignment(labeled_study, monkeypatch, damage):
    import kkt_clf.s08_evaluation as evaluation
    path, anchor = save_decision(labeled_study)
    original = evaluation._parquet_new
    def changed(frame, target):
        if Path(target).name == 'predictions_test.parquet':
            frame = frame.copy()
            if damage == 'reordered':
                frame = frame.iloc[::-1]
            elif damage == 'missing':
                frame = frame.iloc[1:]
            else:
                frame.iloc[1] = frame.iloc[0]
        return original(frame, target)
    monkeypatch.setattr(evaluation, '_parquet_new', changed)
    result = s08.evaluate(labeled_study[1], approval=path, trusted_anchor=anchor)
    assert result['gate'] == 'failed_consumed', 'Misaligned pairs must never silently yield a CI'
    decision = json.loads((Path(result['report_dir']) / 'ni_decision.json').read_text())
    assert decision['delta'] is decision['lower_bound'] is None


def test_evaluated_html_is_isolated_byte_identical_table_only_rebuild(labeled_study, tmp_path, monkeypatch):
    path, anchor = save_decision(labeled_study)
    result = s08.evaluate(labeled_study[1], approval=path, trusted_anchor=anchor)
    assert result['gate'] == 'evaluated', result
    out = Path(result['report_dir'])
    expected = (out / 'report.html').read_bytes()
    isolated = tmp_path / 'isolated_evaluated_tables'
    isolated.mkdir()
    manifest = json.loads((out / 'manifest.json').read_text())
    for name in ['manifest.json', *manifest['tables']]:
        shutil.copyfile(out / name, isolated / name)
    for name in ('load_plan', 'verify_checkpoint', 'verify_selected', 'verify_training_freeze', 'load_vectorizer_artifact'):
        monkeypatch.setattr(s08_report, name, forbid)
    po, bo = Path.open, builtins.open
    def check(path):
        if not isinstance(path, int):
            assert Path(path).resolve().is_relative_to(isolated.resolve()), f'Outside report tables: {path}'
    def path_open(path, *args, **kwargs):
        check(path)
        return po(path, *args, **kwargs)
    def builtin_open(path, *args, **kwargs):
        check(path)
        return bo(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', path_open)
    monkeypatch.setattr(builtins, 'open', builtin_open)
    s08_report.render_report(isolated, isolated / 'rebuilt.html')
    s08_report.render_report(isolated, isolated / 'again.html')
    assert (isolated / 'rebuilt.html').read_bytes() == expected == (isolated / 'again.html').read_bytes()


@pytest.mark.parametrize('damage', ['missing_report', 'fake_complete', 'fake_run_status', 'fake_metric'])
def test_evaluated_reporting_rejects_missing_or_resealed_fake_success(labeled_study, damage):
    path, anchor = save_decision(labeled_study)
    result = s08.evaluate(labeled_study[1], approval=path, trusted_anchor=anchor)
    assert result['gate'] == 'evaluated', result
    out = Path(result['report_dir'])
    manifest = json.loads((out / 'manifest.json').read_text())
    # Work only on isolated disposable copies; preserve all evaluation outputs.
    copy_out = out.parent / ('fake_tables_' + damage)
    copy_out.mkdir()
    for name in ['manifest.json', *manifest['tables']]:
        if not (damage == 'missing_report' and name == 'per_class.parquet'):
            shutil.copyfile(out / name, copy_out / name)
    if damage == 'fake_complete':
        manifest['confirmatory_complete'] = False
    elif damage in ('fake_run_status', 'fake_metric'):
        runs = pd.read_csv(copy_out / 'runs.csv')
        runs.loc[0, 'status' if damage == 'fake_run_status' else 'test_macro_f1'] = 'pending' if damage == 'fake_run_status' else .987654321
        runs.to_csv(copy_out / 'runs.csv', index=False)
        manifest['tables']['runs.csv'] = s08.fingerprint(copy_out / 'runs.csv')
    write_json(copy_out / 'manifest.json', reseal(manifest))
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        s08_report.render_report(copy_out, copy_out / 'fake.html')


def test_changed_payload_cannot_reuse_previous_trusted_margin_anchor(labeled_study, monkeypatch):
    path, anchor = save_decision(labeled_study)
    changed = json.loads(path.read_text())
    changed['delta_NI'] = .01
    write_json(path, changed)
    forbid_quality(monkeypatch)
    with pytest.raises((RuntimeError, ValueError)):
        s08.evaluate(labeled_study[1], approval=path, trusted_anchor=anchor)
    assert not (labeled_study[1] / 'evaluation_claim.json').exists()


def test_unfrozen_training_never_reaches_claim_or_labels(synthetic, monkeypatch):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    record = {'schema': 's08-detached-decision@1', 'scope': 'S08_original_S06_sealed_test_once',
        'primary_candidate': 'class_aware', 'delta_NI': .023, 'practical_rationale': 'synthetic',
        'owner': 'synthetic', 'consent_evidence': 'synthetic', 'one_time_use': True, 'bindings': {}}
    path = root.parent / 'unfrozen_SYNTHETIC_only.json'
    write_json(path, record)
    forbid_quality(monkeypatch)
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        s08.evaluate(root, approval=path, trusted_anchor=file_sha(path))
    assert not (root / 'evaluation_claim.json').exists()


def test_existing_output_directory_cannot_be_overwritten_even_with_synthetic_authority(labeled_study, monkeypatch):
    path, anchor = save_decision(labeled_study)
    out = labeled_study[1] / 'reserved_report'
    out.mkdir()
    marker = out / 'user_file.txt'
    marker.write_text('preserve synthetic user output')
    forbid_quality(monkeypatch)
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        s08.evaluate(labeled_study[1], approval=path, trusted_anchor=anchor, report_dir=out)
    assert marker.read_text() == 'preserve synthetic user output'
    assert not (labeled_study[1] / 'evaluation_claim.json').exists()


def test_mutation_between_final_guard_and_holdout_byte_read_is_detected(labeled_study, monkeypatch):
    path, anchor = save_decision(labeled_study)
    root, holdout = labeled_study[1], labeled_study[5]
    original = Path.read_bytes
    changed = []
    def read_bytes(target):
        if target == holdout and (root / 'evaluation_claim.json').exists():
            with target.open('ab') as stream:
                stream.write(b'changed exactly before snapshot read')
            changed.append(True)
        return original(target)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    forbid_quality(monkeypatch)
    result = s08.evaluate(root, approval=path, trusted_anchor=anchor)
    assert changed and result['gate'] == 'failed_consumed'
    decision = json.loads((Path(result['report_dir']) / 'ni_decision.json').read_text())
    assert 'changed' in decision['error'].lower()
    assert not (Path(result['report_dir']) / 'predictions_test.parquet').exists()


def test_two_simultaneous_authorized_callers_cannot_run_two_evaluations(labeled_study, monkeypatch):
    import kkt_clf.train as train
    path, anchor = save_decision(labeled_study)
    predict = train.predict
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return predict(*args, **kwargs)
    monkeypatch.setattr(train, 'predict', counted)
    def evaluate():
        try:
            return s08.evaluate(labeled_study[1], approval=path, trusted_anchor=anchor)
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: evaluate(), range(2)))
    success = [result for result in results if result is not None]
    assert len(success) == 1 and success[0]['gate'] == 'evaluated', results
    assert len(calls) == 9


def test_interrupted_fourth_inference_is_honest_and_permanently_consumed(labeled_study, monkeypatch):
    import kkt_clf.train as train
    path, anchor = save_decision(labeled_study)
    predict = train.predict
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(True)
        if len(calls) == 4:
            raise RuntimeError('synthetic fourth inference interruption')
        return predict(*args, **kwargs)
    monkeypatch.setattr(train, 'predict', interrupted)
    root = labeled_study[1]
    result = s08.evaluate(root, approval=path, trusted_anchor=anchor)
    assert result['gate'] == 'failed_consumed' and len(calls) == 4
    out = Path(result['report_dir'])
    decision = json.loads((out / 'ni_decision.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['confirmatory_complete'] is False
    assert decision['status'] == 'evaluation_blocked'
    assert decision['delta'] is decision['lower_bound'] is None
    assert 'synthetic fourth inference interruption' in decision['error']
    runs = pd.read_csv(out / 'runs.csv')
    assert runs.test_macro_f1.isna().all()
    assert runs.inference_status.tolist() == ['success'] * 3 + ['failed'] + ['pending'] * 5
    stages = pd.read_csv(out / 'stages.csv')
    inf = stages[stages.stage == 'inference']
    assert inf.status.tolist() == ['success'] * 3 + ['failed']
    assert (inf.charge_count == 1).all()
    assert not (out / 'paired_test.csv').exists()
    assert not (out / 'predictions_test.parquet').exists()
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        s08.evaluate(root, approval=path, trusted_anchor=anchor)
    assert len(calls) == 4


@pytest.mark.parametrize('name', s08.TEST_FILES)
def test_pending_report_rejects_unauthorized_quality_outputs_without_overwriting(synthetic, name):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    out = root / 'report'
    out.mkdir()
    target = out / name
    target.write_bytes(b'independent unauthorized synthetic output marker')
    before = file_sha(target)
    with pytest.raises((RuntimeError, ValueError)):
        s08_report.build_report(root)
    assert file_sha(target) == before
    assert not (out / 'manifest.json').exists()
