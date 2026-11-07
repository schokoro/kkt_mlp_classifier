"""S06 review regressions: fail closed on protocol and provenance violations."""
import copy
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from kkt_clf import s06_baseline as baseline
from kkt_clf.s06_split import S06SplitError, s06_split, save_s06_split
from tests.test_s06_baseline_independent import setup, freeze


@pytest.fixture
def saved_split(tmp_path):
    rows = [dict(name=f'{label} product {i}', L1='root', L2='group', L3=label)
            for label in ('alpha', 'beta') for i in range(20)]
    df = pd.DataFrame(rows)
    source = tmp_path / 'input.parquet'
    df.to_parquet(source, index=False)
    frozen = s06_split(df, seed=42)
    save_s06_split(frozen, str(source), str(tmp_path / 'splits'), str(tmp_path / 'audit'),
                   12, (0.8, 0.1, 0.1), 0.1, 42, 2, 2)
    path = tmp_path / 'splits' / 'manifest.json'
    return baseline.S06Gate1Config(splits_manifest=str(path)), path


def test_gate0_rejects_policy_one(saved_split):
    cfg, path = saved_split
    manifest = json.loads(path.read_text())
    manifest['policy']['min_rows_per_class'] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises((baseline.S06Gate1Error, ValueError)):
        baseline.verify_gate0_split(cfg)


@pytest.mark.parametrize('part', ['train_monitor', 'comparison_val', 'sealed_test'])
def test_gate0_rejects_actual_support_one_even_if_manifest_claims_ok(saved_split, part):
    from kkt_clf.data import file_fingerprint
    cfg, path = saved_split
    manifest = json.loads(path.read_text())
    part_path = manifest['parts'][part]['file']
    df = pd.read_parquet(part_path)
    label = df.L3.iloc[0]
    keep = df.L3.ne(label) | (df.L3.eq(label) & ~df.L3.duplicated())
    df = df[keep]
    df.to_parquet(part_path, index=False)
    fp = file_fingerprint(part_path)
    manifest['parts'][part].update(sha256=fp['sha256'], bytes=fp['bytes'], n_rows=len(df))
    path.write_text(json.dumps(manifest))
    with pytest.raises((baseline.S06Gate1Error, ValueError)):
        baseline.verify_gate0_split(cfg)


@pytest.mark.parametrize('scheduler', ['cosine', 'onecycle'])
def test_unverified_varying_schedule_rejected_before_selection(tmp_path, scheduler, monkeypatch):
    cfg = baseline.S06Gate1Config(output_root=str(tmp_path), train={'scheduler': scheduler})
    monkeypatch.setattr(baseline, 'run_selection', lambda *a, **k: pytest.fail('selection started'))
    with pytest.raises((baseline.S06Gate1Error, ValueError)):
        cfg.validate()
        baseline.freeze_training_protocol(cfg, {}, {}, None, None)


@pytest.mark.parametrize('tamper', ['file', 'hash', 'nested_epochs', 'outer_epochs'])
def test_frozen_report_and_epoch_consistency_checked_before_refit(setup, monkeypatch, tamper):
    cfg, manifest, verification, loaded, _ = setup
    protocol = freeze(setup)
    if tamper == 'file':
        with open(protocol['selection_report_file'], 'a') as f:
            f.write(' ')
    elif tamper == 'hash':
        protocol['selection_report_sha256'] = '0' * 64
    elif tamper == 'nested_epochs':
        protocol['selection_report']['chosen']['fixed_epochs'] += 1
    else:
        protocol['fixed_epochs'] += 1
    monkeypatch.setattr(baseline, 'load_training_protocol', lambda _: protocol)
    monkeypatch.setattr(baseline, 'run_refit', lambda *a, **k: pytest.fail('refit started'))
    with pytest.raises((baseline.S06Gate1Error, ValueError)):
        baseline.run_gate1_refit(cfg, manifest, verification, loaded, None)


def test_feasible_groups_do_not_claim_proven_impossible():
    sizes = [1, 1, 4, 4, 4]
    df = pd.DataFrame([dict(name=f'group {group}', L1='root', L2='branch', L3='class')
                       for group, size in enumerate(sizes) for _ in range(size)])
    try:
        split = s06_split(df, seed=42)
    except S06SplitError as exc:
        message = str(exc).lower()
        assert 'несовместим' not in message and 'невозмож' not in message
    else:
        for part in ('train_fit', 'train_monitor', 'comparison_val', 'sealed_test'):
            assert len(split.parts[part]) >= 2
        assert sum(len(split.parts[p]) for p in ('train_fit', 'train_monitor', 'comparison_val', 'sealed_test')) == 14


@pytest.mark.parametrize('vocab', [
    {'n_char': 1, 'n_word': 1, 'n_total': 2},
    {'n_char': 100000, 'n_word': 300000, 'n_total': 400000},
])
def test_gate1_rejects_other_tfidf_even_with_same_corpus(tmp_path, monkeypatch, vocab):
    cfg = baseline.S06Gate1Config(output_root=str(tmp_path))
    verification = {'artifact_dir': str(tmp_path / 'full'), 'decision': 'ok', 'full_tfidf_id': 'id', 'artifact_version': 'v1',
                    'seed': 42, 'vocab': vocab, 'fit': {}, 'source': {},
                    'corpus_fingerprint': {'sha256': 'same', 'bytes': 1},
                    'corpus_matches_source': True, 'files': {}, 'mode': 'pretrained'}
    monkeypatch.setattr(baseline, 'verify_full_tfidf', lambda *a: copy.deepcopy(verification))
    with pytest.raises((baseline.S06Gate1Error, ValueError)):
        baseline.verify_s06_tfidf(cfg)


def test_tfidf_reference_does_not_claim_unverified_overlap(tmp_path, monkeypatch):
    from scripts import s06_tfidf_ref as ref
    art = tmp_path / 's02' / 'full'
    art.mkdir(parents=True)
    corpus = tmp_path / 'corpus.parquet'
    pd.DataFrame({'name': ['one', 'two']}).to_parquet(corpus)
    feature = art / 'features.joblib'
    feature.write_bytes(b'fixture')
    from kkt_clf.data import file_fingerprint
    fp, cfp = file_fingerprint(str(feature)), file_fingerprint(str(corpus))
    vocab = {'n_char': 3, 'n_word': 1, 'n_total': 4}
    meta = {'source': {'sha256': cfp['sha256'], 'bytes': cfp['bytes'], 'rows_in_file': 2},
            'files': {'features.joblib': {'sha256': fp['sha256'], 'bytes': fp['bytes']}}, 'vocab': vocab}
    (art / 'vectorizer_meta.json').write_text(json.dumps(meta))
    import scipy.sparse as sp
    builder = SimpleNamespace(n_char_features=3, n_word_features=1, n_features=4, fit_report_={},
                              transform=lambda texts: sp.csr_matrix((len(texts), 4)))
    monkeypatch.setattr(ref, 'load_vectorizer_artifact', lambda _: SimpleNamespace(builder=builder))
    out = tmp_path / 'ref.json'
    monkeypatch.setattr(sys, 'argv', ['s06_tfidf_ref.py', '--s02-dir', str(tmp_path / 's02'),
                                    '--full-name', 'full', '--corpus', str(corpus), '--out', str(out),
                                    '--expect-vocab-total', '4'])
    ref.main()
    if out.exists():
        result = json.loads(out.read_text())
        provenance = result.get('corpus_provenance', {})
        if result.get('status') == 'ok':
            assert provenance.get('fit_on_labeled_train') is not False
            assert '2 совпадения' not in json.dumps(provenance, ensure_ascii=False)
