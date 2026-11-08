"""S08 security/provenance negatives and synthetic-only integration checks."""
from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kkt_clf import s08
from tests.test_s08_fixtures import sources, file_sha, write_json, reseal


@pytest.fixture
def synthetic(sources):
    cfg = s08.S08Config(manifest=str(sources['manifest']), protocol=str(sources['source_protocol']),
                        full_tfidf=str(sources['full']), corpus=str(sources['corpus']),
                        fixture=True, fixture_pins=sources['pins'], fixture_full_counts=(24, 8),
                        fixture_selected_counts=(6, 2), num_threads=1)
    return cfg, sources['root'], sources


def forbid(*args, **kwargs):
    pytest.fail('Forbidden inference/training/refit/label access reached')


def guard_closed_reads(monkeypatch, sealed_path):
    original = pd.read_parquet
    reads = []
    def guarded(path, *args, **kwargs):
        if Path(path).resolve() == Path(sealed_path).resolve():
            assert kwargs.get('columns') == ['row_id', 'name'], 'Sealed label read before authorization'
            reads.append(kwargs['columns'])
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', guarded)
    return reads


def test_plan_is_nine_fixed_seed_runs_on_entire_development_union(synthetic, monkeypatch):
    cfg, root, src = synthetic
    reads = guard_closed_reads(monkeypatch, src['parts']['sealed_test']['file'])
    for name in ('predict', 'train_model', 'select_features', 'refit_selected_vectorizer'):
        if hasattr(s08, name):
            monkeypatch.setattr(s08, name, forbid)
    plan = s08.write_plan(cfg, root)
    assert reads
    assert len(plan['runs']) == len({r['run_id'] for r in plan['runs']}) == 9
    assert {(r['method'], r['seed']) for r in plan['runs']} == set(itertools.product(
        ('none', 'global', 'class_aware'), (42, 43, 44)))
    protocol = json.loads((root / 'protocol.json').read_text())
    assert protocol['delta_NI'] is None
    assert protocol['margin_status'] == protocol['primary_status'] == 'unapproved'
    assert not (root / 'runs').exists()
    assert s08.load_plan(root) == json.loads(json.dumps(plan))
    # Exact epoch override and full pool are additionally inspected at training calls.


@pytest.mark.parametrize('source', ['manifest', 'protocol', 'corpus', 'features'])
def test_changed_authoritative_source_fails_closed(synthetic, source):
    cfg, root, src = synthetic
    s08.write_plan(cfg, root)
    paths = {'manifest': cfg.manifest, 'protocol': cfg.protocol,
             'corpus': cfg.corpus, 'features': str(Path(cfg.full_tfidf) / 'features.joblib')}
    with Path(paths[source]).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises((RuntimeError, ValueError)):
        s08.load_plan(root)


@pytest.mark.parametrize('part', ['outer_train', 'comparison_val', 'sealed_test', 'train_fit', 'train_monitor'])
def test_changed_partition_fails_before_training(synthetic, part, monkeypatch):
    cfg, root, src = synthetic
    s08.write_plan(cfg, root)
    with Path(src['parts'][part]['file']).open('ab') as handle:
        handle.write(b'changed')
    if hasattr(s08, 'train_model'):
        monkeypatch.setattr(s08, 'train_model', forbid)
    with pytest.raises((RuntimeError, ValueError)):
        s08.execute(root)


@pytest.mark.parametrize('document', ['plan.json', 'protocol.json'])
def test_frozen_documents_cannot_be_tampered(synthetic, document):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    path = root / document
    value = json.loads(path.read_text())
    value['delta_NI'] = .01
    value['primary_status'] = 'approved'
    write_json(path, value)
    with pytest.raises((RuntimeError, ValueError)):
        s08.load_plan(root)


@pytest.mark.parametrize('damage', ['sealed_row_overlap', 'sealed_group_overlap', 'dev_duplicate_row',
                                    'conflicting_hierarchy', 'label_maps'])
def test_even_repinned_synthetic_sources_cannot_hide_invalid_groups_or_maps(synthetic, damage):
    cfg, root, src = synthetic
    part = 'sealed_test' if damage.startswith('sealed') else 'outer_train'
    path = Path(src['parts'][part]['file'])
    frame = pd.read_parquet(path)
    if damage == 'sealed_row_overlap':
        frame.loc[0, 'row_id'] = int(src['pool'].row_id.iloc[0])
    elif damage == 'sealed_group_overlap':
        frame.loc[0, 'name'] = '  ' + src['pool'].name.iloc[0].upper().replace(' ', '\t') + ' '
    elif damage == 'dev_duplicate_row':
        frame.loc[1, 'row_id'] = frame.loc[0, 'row_id']
    elif damage == 'conflicting_hierarchy':
        frame.loc[0, 'L2'] = 'conflicting'
    frame.to_parquet(path, index=False)
    manifest = json.loads(src['manifest'].read_text())
    manifest['parts'][part].update(sha256=file_sha(path), bytes=path.stat().st_size)
    if damage == 'label_maps':
        manifest['label_maps']['L3'] = ['beta', 'alpha']
    write_json(src['manifest'], manifest)
    protocol = json.loads(src['source_protocol'].read_text())
    protocol['splits'] = manifest['parts']
    write_json(src['source_protocol'], protocol)
    cfg.fixture_pins['manifest'] = file_sha(src['manifest'])
    cfg.fixture_pins['protocol'] = file_sha(src['source_protocol'])
    with pytest.raises((RuntimeError, ValueError)):
        s08.write_plan(cfg, root)


@pytest.mark.parametrize('approval', [None, True, {'approved': True},
    {'approved': True, 'delta_NI': .01, 'primary_candidate': 'class_aware', 'one_time_use': True},
    {'approved': True, 'plan_sha256': '0' * 64, 'protocol_sha256': '0' * 64,
     'training_sha256': '0' * 64, 'delta_NI': .01, 'owner': 'fabricated',
     'rationale': 'self assertion is not user consent', 'primary_candidate': 'global'}])
def test_no_unauthenticated_approval_can_reach_sealed_labels_or_prediction(synthetic, monkeypatch, approval):
    cfg, root, src = synthetic
    s08.write_plan(cfg, root)
    guard_closed_reads(monkeypatch, src['parts']['sealed_test']['file'])
    if hasattr(s08, 'predict'):
        monkeypatch.setattr(s08, 'predict', forbid)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        s08.evaluate(root, approval=approval)
    for name in ('predictions_test.parquet', 'paired_test.csv', 'per_class.parquet',
                 'support_bins.csv', 'confusion_pairs.csv'):
        assert not list(root.rglob(name))


@pytest.mark.parametrize('existing', ['execution_started.json', 'runs/stale/result.json',
                                      'shared/stale.json', 'stages/stale.json'])
def test_execution_refuses_reuse_without_touching_existing_bytes(synthetic, existing, monkeypatch):
    cfg, root, _ = synthetic
    s08.write_plan(cfg, root)
    target = root / existing
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('stale or interrupted execution')
    before = {str(p): file_sha(p) for p in root.rglob('*') if p.is_file()}
    if hasattr(s08, 'train_model'):
        monkeypatch.setattr(s08, 'train_model', forbid)
    with pytest.raises((RuntimeError, ValueError)):
        s08.execute(root)
    assert before == {str(p): file_sha(p) for p in root.rglob('*') if p.is_file()}


def test_production_profile_cannot_override_source_pins(synthetic):
    cfg, _, _ = synthetic
    cfg = dataclasses.replace(cfg, fixture=False)
    with pytest.raises((RuntimeError, ValueError)):
        cfg.validate()


def test_production_budgets_and_design_are_not_fixture_defaults():
    cfg = s08.S08Config()
    assert cfg.full_counts == (300000, 100000)
    assert cfg.selected_counts == (75000, 25000)
    matrix = s08.run_matrix(cfg)
    assert len(matrix) == 9
    assert {(run['n_char'], run['n_word']) for run in matrix if run['method'] != 'none'} == {(75000, 25000)}
    assert set(run['method'] for run in matrix) == {'none', 'class_aware', 'global'}
