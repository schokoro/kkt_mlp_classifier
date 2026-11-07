"""Independent S07 fixtures: no production inputs, models, or sealed labels."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from kkt_clf import s07
from kkt_clf.config import BranchConfig, FeatureConfig, LossConfig, ModelConfig, TrainConfig
from kkt_clf.vectorizer import fit_and_save_vectorizer


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True), encoding='utf-8')


def _seal(obj):
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return {**obj, 'integrity_sha256': hashlib.sha256(raw.encode()).hexdigest()}


def _maps():
    return {'L1': ['root'], 'L2': ['branch'], 'L3': ['alpha', 'beta'],
            'parent_l2_of_l3': [0, 0], 'parent_l1_of_l2': [0]}


def _pool(n=10):
    return pd.DataFrame([{'row_id': c * 100 + i, 'name': f'{label} item{chr(97+i)} fresh',
                          'L1': 'root', 'L2': 'branch', 'L3': label}
                         for c, label in enumerate(('alpha', 'beta')) for i in range(n)])


@pytest.fixture
def synthetic(tmp_path):
    """Build original sources independently, with no S06 output reads."""
    pool = _pool()
    corpus = tmp_path / 'corpus.parquet'
    pd.DataFrame({'name': pool.name.tolist() + ['extra assorted grocery token']}).to_parquet(corpus)
    full = tmp_path / 'full'
    feat = FeatureConfig(char=BranchConfig((2, 3), 1, 24, True),
                         word=BranchConfig((1, 1), 1, 8, True), vectorizer_fit_rows=None)
    fit_and_save_vectorizer(str(full), feat, str(corpus), max_rows=100, seed=42)
    parts = {}
    frames = {'outer_train': pool.groupby('L3', sort=False).head(8),
              'comparison_val': pool.groupby('L3', sort=False).tail(2),
              # Deliberately no label columns: structural verification must suffice.
              'sealed_test': pd.DataFrame({'row_id': [900, 901], 'name': ['sealed one', 'sealed two']})}
    for part, frame in frames.items():
        path = tmp_path / f'{part}.parquet'
        frame.to_parquet(path, index=False)
        parts[part] = {'file': str(path), 'sha256': _hash(path), 'bytes': path.stat().st_size,
                       'n_rows': len(frame)}
    manifest = tmp_path / 'manifest.json'
    _json(manifest, {'policy': {'min_count': 12}, 'parts': parts, 'label_maps': _maps()})
    protocol = tmp_path / 'protocol.json'
    _json(protocol, {'frozen': True, 'fixed_epochs': 1, 'chosen': {'fixed_epochs': 1},
                     'full_tfidf_id': _hash(full / 'features.joblib'),
                     'corpus_fingerprint': {'sha256': _hash(corpus)}, 'splits': parts,
                     'resolved_sections': {'model': dataclasses.asdict(ModelConfig([4], [0.])),
                                           'loss': dataclasses.asdict(LossConfig()),
                                           'train': dataclasses.asdict(TrainConfig(epochs=1, scheduler='none',
                                                                                 batch_size=8, num_threads=1))}})
    pins = {key: _hash(path) for key, path in {'manifest': manifest, 'protocol': protocol,
                                             'corpus': corpus, 'features': full / 'features.joblib'}.items()}
    cfg = s07.S07Config(manifest=str(manifest), protocol=str(protocol), full_tfidf=str(full),
                        corpus=str(corpus), fixture=True, fixture_pins=pins,
                        fixture_full_counts=(24, 8), fixture_selected_counts=(6, 2), num_threads=1)
    return cfg, tmp_path / 'study'


def test_three_deterministic_splits_keep_under12_development_classes():
    pool = _pool()
    first, diagnostics = s07.split_development_pool(pool, _maps())
    second, _ = s07.split_development_pool(pool.sample(frac=1, random_state=99), _maps())
    assert set(first) == {'split_101', 'split_202', 'split_303'}
    assert all(d['feasible'] for d in diagnostics)
    assignments = []
    for sid, parts in first.items():
        assignments.append(tuple(parts['comparison_val'].row_id))
        leaf_ids = []
        groups = []
        for part, minimum in [('train_fit', 6), ('train_monitor', 2), ('comparison_val', 2)]:
            frame = parts[part]
            assert frame.L3.value_counts().to_dict() == {'alpha': minimum, 'beta': minimum}
            assert frame.equals(second[sid][part])
            leaf_ids.extend(frame.row_id)
            groups.append(set(re.sub(r'\s', '', x.lower()) for x in frame.name))
        assert sorted(leaf_ids) == sorted(pool.row_id)
        assert all(not a & b for a, b in itertools.combinations(groups, 2))
        assert set(parts['outer_train'].row_id) == set(parts['train_fit'].row_id) | set(parts['train_monitor'].row_id)
    assert len(set(assignments)) == 3


def test_duplicate_normalized_names_stay_together():
    pool = _pool(12)
    for label in ('alpha', 'beta'):
        ids = pool.index[pool.L3 == label][:2]
        pool.loc[ids, 'name'] = [f'{label} DUP', f'  {label.upper()}\tD U P ']
    splits, _ = s07.split_development_pool(pool, _maps())
    assert len(splits) == 3
    for parts in splits.values():
        for label in ('alpha', 'beta'):
            ids = set(pool[pool.L3 == label].head(2).row_id)
            assert sum(ids <= set(parts[p].row_id) for p in ('train_fit', 'train_monitor', 'comparison_val')) == 1


@pytest.mark.parametrize('sizes', [(9,), (8, 1, 1), (6, 4)])
def test_proven_group_infeasibility_preserves_class_and_diagnostics(sizes):
    pool = _pool(sum(sizes))
    for label in ('alpha', 'beta'):
        names = [f'{label} group{g}' for g, size in enumerate(sizes) for _ in range(size)]
        pool.loc[pool.L3 == label, 'name'] = names
    splits, diag = s07.split_development_pool(pool, _maps())
    assert not splits
    assert len(diag) == 3
    for d in diag:
        assert not d['feasible']
        assert {f['L3'] for f in d['failures']} == {'alpha', 'beta'}
        assert all(f['proven_infeasible'] for f in d['failures'])


def test_exhausted_search_is_not_a_proof(monkeypatch):
    monkeypatch.setattr(s07, 'assign_groups', lambda sizes, quota, minima, rng: (np.zeros(len(sizes), int), np.array([10, 0, 0])))
    monkeypatch.setattr(s07, 'search_feasible_assignment', lambda sizes, minima: (None, False, 7))
    splits, diag = s07.split_development_pool(_pool(), _maps())
    assert not splits
    assert all(not f['proven_infeasible'] and f['search_nodes'] == 7 for d in diag for f in d['failures'])


@pytest.mark.parametrize('damage', ['name', 'label', 'overlap', 'missing', 'outer'])
def test_partition_validation_rejects_corruption(damage):
    pool = _pool()
    splits, _ = s07.split_development_pool(pool, _maps())
    parts = copy.deepcopy(splits['split_101'])
    if damage == 'name':
        parts['train_fit'].loc[0, 'name'] = 'substitution'
    elif damage == 'label':
        parts['train_fit'].loc[0, 'L1'] = 'unknown'
    elif damage == 'overlap':
        parts['comparison_val'] = pd.concat([parts['comparison_val'], parts['train_fit'].head(1)])
    elif damage == 'missing':
        parts['train_fit'] = parts['train_fit'].iloc[1:]
    else:
        parts['outer_train'] = parts['outer_train'].iloc[1:]
    with pytest.raises(s07.S07Error):
        s07.validate_partition(pool, parts, _maps())


def test_sealed_structural_projection_and_no_training_during_plan(synthetic, monkeypatch):
    cfg, root = synthetic
    original = pd.read_parquet
    calls = []
    def guarded(path, *args, **kwargs):
        if str(path).endswith('sealed_test.parquet'):
            assert kwargs.get('columns') == ['row_id', 'name']
            calls.append(path)
        return original(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('Planning must not train, rank, refit or predict')
    monkeypatch.setattr(pd, 'read_parquet', guarded)
    for name in ('predict', 'train_model', 'select_features', 'compute_margin_attributions', 'refit_selected_vectorizer'):
        monkeypatch.setattr(s07, name, forbidden)
    plan = s07.write_plan(cfg, root)
    assert calls
    assert plan['study_kind'] == 'synthetic'
    assert not (root / 'runs').exists()
    keys = {(r['split_id'], r['method'], r['seed']) for r in plan['runs']}
    assert keys == set(itertools.product(('split_101', 'split_202', 'split_303'),
                                        ('none', 'global', 'class_aware', 'gradient_x_input'), (42, 43, 44)))
    assert len(plan['runs']) == len({r['run_id'] for r in plan['runs']}) == 36
    assert {r['initial_status'] for r in plan['runs']} == {'pending'}
    assert len(plan['dependencies']) == 9
    for dep in plan['dependencies']:
        sid, method = dep['split_id'], dep['method']
        consumers = [r for r in plan['runs'] if r['feature_dependency'] == dep['dependency_id']]
        assert {r['seed'] for r in consumers} == {42, 43, 44}
        assert {r['split_id'] for r in consumers} == {sid}
        assert dep['ranking_rows'] == plan['splits'][sid]['parts']['outer_train']['sha256']
        if method == 'gradient_x_input':
            assert dep['reference_checkpoint_run'] == f'{sid}__none__seed_42'
            assert dep['reference_seed'] == 42
    assert s07.load_plan(root) == json.loads(json.dumps(plan))


@pytest.mark.parametrize('source', ['manifest', 'protocol', 'corpus', 'features'])
def test_authoritative_source_pins_fail_closed(synthetic, source):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    paths = {'manifest': cfg.manifest, 'protocol': cfg.protocol, 'corpus': cfg.corpus,
             'features': str(Path(cfg.full_tfidf) / 'features.joblib')}
    with open(paths[source], 'ab') as f:
        f.write(b'changed')
    with pytest.raises(s07.S07Error):
        s07.load_plan(root)


@pytest.mark.parametrize('field', ['runs', 'training', 'label_maps', 'dependencies', 'config'])
def test_unsigned_plan_mutation_fails_closed(synthetic, field):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    path = root / 'plan.json'
    plan = json.loads(path.read_text())
    plan[field] = []
    _json(path, plan)
    with pytest.raises(s07.S07Error):
        s07.load_plan(root)


@pytest.mark.parametrize('damage', ['reference_seed', 'ranking_rows', 'cross_split', 'duplicate_dependency'])
def test_semantically_invalid_dependency_fails_even_with_consistent_seal(synthetic, damage):
    cfg, root = synthetic
    plan = s07.write_plan(cfg, root)
    dep = next(d for d in plan['dependencies'] if d['method'] == 'gradient_x_input')
    if damage == 'reference_seed':
        dep['reference_seed'] = 44
        dep['reference_checkpoint_run'] = f"{dep['split_id']}__none__seed_44"
    elif damage == 'ranking_rows':
        dep['ranking_rows'] = plan['splits'][dep['split_id']]['parts']['comparison_val']['sha256']
    elif damage == 'cross_split':
        dep['reference_checkpoint_run'] = 'split_202__none__seed_42'
    else:
        plan['dependencies'][0] = copy.deepcopy(plan['dependencies'][1])
    _json(root / 'plan.json', _seal(plan))
    with pytest.raises(s07.S07Error):
        s07.load_plan(root)


@pytest.mark.parametrize('target', ['split', 'supports'])
def test_saved_partition_files_are_immutable(synthetic, target):
    cfg, root = synthetic
    plan = s07.write_plan(cfg, root)
    split = plan['splits']['split_101']
    spec = split['parts']['train_fit'] if target == 'split' else split['supports']
    with (root / spec['file']).open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(s07.S07Error):
        s07.load_plan(root)


def test_populated_output_and_resume_are_rejected_without_changes(synthetic, monkeypatch):
    cfg, root = synthetic
    s07.write_plan(cfg, root)
    sentinel = root / 'user.txt'
    sentinel.write_text('preserve me')
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises((FileExistsError, s07.S07Error)):
        s07.write_plan(cfg, root)
    (root / 'execution_started.json').write_text('{}')
    monkeypatch.setattr(s07, 'train_model', lambda *a, **k: pytest.fail('unsafe resume trained'))
    with pytest.raises(s07.S07Error, match='[Rr]esume'):
        s07.execute(root)
    assert all((root / path).read_bytes() == data for path, data in before.items())


def test_production_profile_cannot_override_pins_or_budgets():
    cfg = s07.S07Config(fixture_full_counts=(24, 8), fixture_selected_counts=(6, 2))
    cfg.validate()
    assert cfg.full_counts == (300000, 100000)
    assert cfg.selected_counts == (75000, 25000)
    with pytest.raises(s07.S07Error):
        dataclasses.replace(cfg, fixture_pins={}).validate()


@pytest.mark.parametrize('damage', ['sealed_id', 'sealed_group', 'label_map', 'policy', 'row_hash', 'epochs'])
def test_invalid_references_rejected_even_when_source_files_are_repinned(synthetic, damage):
    cfg, root = synthetic
    manifest_path = Path(cfg.manifest)
    protocol_path = Path(cfg.protocol)
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if damage in ('sealed_id', 'sealed_group'):
        path = Path(manifest['parts']['sealed_test']['file'])
        frame = pd.read_parquet(path)
        pool = _pool()
        frame.loc[0, 'row_id' if damage == 'sealed_id' else 'name'] = (
            int(pool.row_id.iloc[0]) if damage == 'sealed_id' else pool.name.iloc[0].upper().replace(' ', '\t'))
        frame.to_parquet(path, index=False)
        manifest['parts']['sealed_test'].update(sha256=_hash(path), bytes=path.stat().st_size)
        protocol['splits'] = manifest['parts']
    elif damage == 'label_map':
        manifest['label_maps']['L3'].append('silently_missing_class')
    elif damage == 'policy':
        manifest['policy']['min_count'] = 10
    elif damage == 'row_hash':
        manifest['parts']['outer_train']['sha256'] = '0' * 64
        protocol['splits'] = manifest['parts']
    else:
        protocol['chosen']['fixed_epochs'] = 2
    _json(manifest_path, manifest)
    _json(protocol_path, protocol)
    pins = {**cfg.fixture_pins, 'manifest': _hash(manifest_path), 'protocol': _hash(protocol_path)}
    with pytest.raises(s07.S07Error):
        s07.write_plan(dataclasses.replace(cfg, fixture_pins=pins), root)
    assert not root.exists()


def test_failed_preflight_persists_honest_zero_workload(synthetic):
    cfg, root = synthetic
    cfg = dataclasses.replace(cfg, fixture_pins={**cfg.fixture_pins, 'corpus': '0' * 64})
    audit = s07.preflight(cfg, root)
    assert not audit['passed']
    assert audit['workload_runs'] == 0 and audit['sealed_labels_read'] is False
    assert audit['sealed_inference'] is False
    assert 'SHA256' in audit['error']
    assert (root / 'preflight.json').is_file()
    assert not (root / 'runs').exists()
