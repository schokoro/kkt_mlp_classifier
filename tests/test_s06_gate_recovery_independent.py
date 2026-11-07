"""Gate 1 must reject incomplete or internally inconsistent evidence, without training."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from kkt_clf import s06_comparison as comp


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def gate(tmp_path, monkeypatch):
    cfg = comp.S06ComparisonConfig(output_root=str(tmp_path))
    root = Path(cfg.baseline_dir)
    root.mkdir(parents=True)
    parts = {name: {'sha256': 'sha-' + name, 'bytes': 10, 'n_rows': 2}
             for name in ('outer_train', 'train_fit', 'train_monitor', 'comparison_val', 'sealed_test')}
    maps = {'L1': ['food'], 'L2': ['items'], 'L3': ['bread', 'milk']}
    manifest = {'parts': parts, 'label_maps': maps, 'policy': {'min_count': 12}}
    verification = {'decision': 'ok', 'full_tfidf_id': 'tfidf-sha',
                    'corpus_fingerprint': {'sha256': 'corpus-sha'}}
    protocol = {'config': {'seeds': [42, 43, 44]}, 'fixed_epochs': 3}
    bman = {'fixed_epochs': 3, 'config_id': 'config-a', 'label_maps': maps,
            'inputs': {'splits': parts, 'full_tfidf': {'full_tfidf_id': 'tfidf-sha'},
                       'corpus': {'sha256': 'corpus-sha'}}}
    rows = []
    hashes = {}
    for seed in (42, 43, 44):
        ckpt = root / 'checkpoints' / f'seed_{seed}'
        ckpt.mkdir(parents=True)
        (ckpt / 'model.pt').write_bytes(f'independent weights {seed}'.encode())
        meta = {'seed': seed, 'train': {'epochs_completed': 3, 'restored_best': False},
                'train_fingerprints': {'outer_train_sha256': parts['outer_train']['sha256'],
                                       'train_monitor_sha256': parts['train_monitor']['sha256']},
                'full_tfidf': {'full_tfidf_id': 'tfidf-sha'}}
        write_json(ckpt / 'meta.json', meta)
        write_json(ckpt / 'label_maps.json', maps)
        rows.append({'run_id': f'seed_{seed}', 'train_seed': seed, 'status': 'success',
                     'config_id': 'config-a', 'epochs_completed': 3,
                     'full_tfidf_id': 'tfidf-sha', 'l3_macro_f1': 0.8,
                     'l3_accuracy': 0.85, 'checkpoint_path': str(ckpt)})
        if seed == 42:
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ckpt.iterdir()}
    pd.DataFrame(rows).to_csv(root / 'runs.csv', index=False)
    write_json(root / 'majority_baseline.json', {'l3_macro_f1': 0.3})
    write_json(root / 'manifest.json', bman)
    selected = {'run_id': 'seed_42', 'checkpoint_dir': rows[0]['checkpoint_path'],
                'checkpoint_files_sha256': hashes,
                'roundtrip': {'status': 'success', 'metrics_match': True,
                              'l3_macro_f1_saved': 0.8, 'l3_macro_f1_recomputed': 0.8,
                              'l3_accuracy_saved': 0.85, 'l3_accuracy_recomputed': 0.85}}
    write_json(root / 'selected_checkpoint.json', selected)
    monkeypatch.setattr(comp, 'verify_gate0_split', lambda _: manifest)
    monkeypatch.setattr(comp, 'verify_full_tfidf', lambda *args: verification)
    monkeypatch.setattr(comp, 'load_training_protocol', lambda _: protocol)
    monkeypatch.setattr(comp, 'validate_protocol_against_inputs', lambda *args: None)
    # Verify both public orchestration and the actual gate before any mutation.
    assert set(comp._verify_gate1_baseline(cfg, manifest, verification, protocol)['seeds']) == {42, 43, 44}
    assert set(comp.verify_preconditions(cfg)['gate1']['seeds']) == {42, 43, 44}
    return cfg, root, manifest, verification, protocol


def reject(gate, entry):
    cfg, _, manifest, verification, protocol = gate
    with pytest.raises(comp.S06ComparisonError):
        if entry == 'gate':
            comp._verify_gate1_baseline(cfg, manifest, verification, protocol)
        else:
            comp.verify_preconditions(cfg)


@pytest.mark.parametrize('entry', ['gate', 'preconditions'])
@pytest.mark.parametrize('seed', [42, 43, 44])
def test_missing_baseline_seed_is_rejected(gate, entry, seed):
    path = gate[1] / 'runs.csv'
    rows = pd.read_csv(path)
    rows[rows.run_id != f'seed_{seed}'].to_csv(path, index=False)
    reject(gate, entry)


@pytest.mark.parametrize('entry', ['gate', 'preconditions'])
def test_single_seed_protocol_does_not_bypass_gate(gate, entry):
    gate[0].seeds = (42,)
    gate[4]['config']['seeds'] = [42]
    path = gate[1] / 'runs.csv'
    rows = pd.read_csv(path)
    rows[rows.run_id == 'seed_42'].to_csv(path, index=False)
    reject(gate, entry)


@pytest.mark.parametrize('entry', ['gate', 'preconditions'])
@pytest.mark.parametrize('field', ['corpus', 'tfidf', 'split', 'label_maps'])
def test_missing_on_both_sides_is_not_matching_provenance(gate, entry, field):
    _, root, manifest, verification, _ = gate
    path = root / 'manifest.json'
    bman = json.loads(path.read_text())
    if field == 'corpus':
        del bman['inputs']['corpus']['sha256']
        del verification['corpus_fingerprint']['sha256']
    elif field == 'tfidf':
        del bman['inputs']['full_tfidf']['full_tfidf_id']
        del verification['full_tfidf_id']
        for seed in (42, 43, 44):
            meta_path = root / 'checkpoints' / f'seed_{seed}' / 'meta.json'
            meta = json.loads(meta_path.read_text())
            del meta['full_tfidf']['full_tfidf_id']
            write_json(meta_path, meta)
    elif field == 'split':
        del bman['inputs']['splits']['comparison_val']['sha256']
        del manifest['parts']['comparison_val']['sha256']
    else:
        del bman['label_maps']
        del manifest['label_maps']
        for seed in (42, 43, 44):
            write_json(root / 'checkpoints' / f'seed_{seed}' / 'label_maps.json', {})
    write_json(path, bman)
    reject(gate, entry)


@pytest.mark.parametrize('entry', ['gate', 'preconditions'])
@pytest.mark.parametrize('corruption', ['roundtrip_metrics', 'run_metrics', 'run_config', 'checkpoint_hash'])
def test_success_status_does_not_override_inconsistent_evidence(gate, entry, corruption):
    root = gate[1]
    selected_path = root / 'selected_checkpoint.json'
    selected = json.loads(selected_path.read_text())
    if corruption == 'roundtrip_metrics':
        selected['roundtrip']['l3_macro_f1_recomputed'] = 0.1
        selected['roundtrip']['metrics_match'] = False
        write_json(selected_path, selected)
    elif corruption == 'checkpoint_hash':
        (root / 'checkpoints' / 'seed_42' / 'model.pt').write_bytes(b'replaced weights')
    else:
        path = root / 'runs.csv'
        rows = pd.read_csv(path)
        if corruption == 'run_metrics':
            rows.loc[rows.run_id == 'seed_42', 'l3_macro_f1'] = 0.6
        else:
            rows.loc[rows.run_id == 'seed_43', 'config_id'] = 'different-config'
        rows.to_csv(path, index=False)
    reject(gate, entry)
