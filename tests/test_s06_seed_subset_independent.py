"""S06 comparison seed subsets must retain the complete Gate 1 contract."""
import dataclasses
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from kkt_clf import s06_baseline, s06_comparison as comp
from tests.test_s06_gate_recovery_independent import gate


@pytest.fixture
def subset_gate(gate, monkeypatch, tmp_path):
    cfg, root, manifest, verification, protocol = gate
    cfg.seeds = (42,)
    g1 = cfg.gate1_config()
    candidate = {'name': 'fixed', 'model': {}, 'train': {}}
    model, loss, train = g1.to_s05_config().build_sections(candidate['model'], candidate['train'])
    chosen = {'name': 'fixed', 'fixed_epochs': 3}
    report = {'chosen': chosen, 'candidates': [candidate]}
    report_path = tmp_path / 'selection.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    protocol.update({
        'config': {'seeds': [42, 43, 44], 'reference_seed': 42},
        'chosen': chosen, 'selection_report': report,
        'selection_report_file': str(report_path),
        'selection_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'splits': manifest['parts'], 'full_tfidf_id': verification['full_tfidf_id'],
        'corpus_fingerprint': dict(verification['corpus_fingerprint']),
        'min_count_policy': manifest['policy']['min_count'],
        'resolved_sections': {'model': dataclasses.asdict(model), 'loss': dataclasses.asdict(loss),
                              'train': dataclasses.asdict(train)},
    })
    monkeypatch.setattr(comp, 'validate_protocol_against_inputs', s06_baseline.validate_protocol_against_inputs)
    return cfg, root, manifest, verification, protocol


def test_subset_accepts_complete_gate(subset_gate):
    cfg, _, _, _, _ = subset_gate
    result = comp.verify_preconditions(cfg)
    assert set(result['gate1']['seeds']) == {42, 43, 44}


@pytest.mark.parametrize('seeds', [(42, 45), ()])
def test_empty_or_outside_protocol_seed_rejected(subset_gate, seeds):
    cfg = subset_gate[0]
    cfg.seeds = seeds
    with pytest.raises((comp.S06ComparisonError, s06_baseline.S06Gate1Error, ValueError)):
        comp.verify_preconditions(cfg)


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_subset_still_requires_every_baseline(subset_gate, seed):
    cfg, root, *_ = subset_gate
    path = root / 'runs.csv'
    rows = pd.read_csv(path)
    rows[rows.run_id != f'seed_{seed}'].to_csv(path, index=False)
    with pytest.raises(comp.S06ComparisonError):
        comp.verify_preconditions(cfg)


@pytest.mark.parametrize('field', ['model', 'loss', 'train', 'split', 'corpus', 'full'])
def test_subset_does_not_relax_protocol_matching(subset_gate, field):
    cfg, _, _, _, protocol = subset_gate
    if field in ('model', 'loss', 'train'):
        protocol['resolved_sections'][field]['unexpected_field'] = 'changed'
    elif field == 'split':
        protocol['splits'] = {**protocol['splits'], 'comparison_val': {'sha256': 'different', 'n_rows': 2}}
    elif field == 'corpus':
        protocol['corpus_fingerprint']['sha256'] = 'different'
    else:
        protocol['full_tfidf_id'] = 'different'
    with pytest.raises((comp.S06ComparisonError, s06_baseline.S06Gate1Error, ValueError)):
        comp.verify_preconditions(cfg)


def test_subset_plan_rejects_changed_protocol_hash(subset_gate):
    cfg = subset_gate[0]
    path = Path(cfg.protocol_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset_gate[4]), encoding='utf-8')
    comp.write_plan(cfg)
    plan_path = Path(cfg.planned_path)
    before = plan_path.read_bytes()
    path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises((comp.S06ComparisonError, ValueError)):
        comp.write_plan(cfg)
    assert plan_path.read_bytes() == before
