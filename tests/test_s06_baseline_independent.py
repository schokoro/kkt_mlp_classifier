"""Independent S06 Gate 1 checks with a tiny in-memory corpus."""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from kkt_clf import s05_baseline as s05, s06_baseline as s06
from kkt_clf.config import BranchConfig, FeatureConfig
from kkt_clf.features import FeatureBuilder
from kkt_clf.labels import LabelMaps


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    df = pd.DataFrame({'name': ['alpha milk', 'beta bread', 'alpha food', 'beta food'],
                       'L1': ['root'] * 4, 'L2': ['group'] * 4,
                       'L3': ['a', 'b', 'a', 'b']})
    parts = {'outer_train': df, 'train_fit': df.iloc[:2],
             'train_monitor': df.iloc[2:], 'comparison_val': df.iloc[:2]}
    manifest = {'label_maps': LabelMaps.from_df(df).to_dict(),
                'policy': {'min_count': 12, 'split_seed': 7},
                'parts': {k: {'sha256': k, 'bytes': 1, 'n_rows': len(v)} for k, v in parts.items()},
                'inputs': {'labeled_parquet': {'sha256': 'source'}}}
    manifest['parts']['sealed_test'] = {'sha256': 'sealed', 'bytes': 1, 'n_rows': 2}
    builder = FeatureBuilder(FeatureConfig(BranchConfig((2, 3), 1), BranchConfig((1, 1), 1), None, 10))
    builder.fit(df.name.tolist(), seed=3)
    loaded = SimpleNamespace(builder=builder, meta={'files': {'features.joblib': {'sha256': 'full-id'}}})
    cfg = s06.S06Gate1Config(output_root=str(tmp_path / 'result'),
                             splits_manifest=str(tmp_path / 'split.json'),
                             model={'hidden': [4], 'dropout': [0.0]},
                             train={'scheduler': 'none', 'batch_size': 2, 'epochs': 2, 'patience': 2},
                             device='cpu')
    (tmp_path / 'result' / 'shared').mkdir(parents=True)
    (tmp_path / 'split.json').write_text('{}')
    verification = {'decision': 'ok', 'full_tfidf_id': 'full-id', 'artifact_version': 'v1',
                    'corpus_fingerprint': {'sha256': 'corpus', 'bytes': 10}}
    events = []
    def read_part(_manifest, name):
        events.append(('read', name))
        assert name != 'sealed_test', 'sealed_test rows accessed'
        return parts[name].copy()
    monkeypatch.setattr(s05, 'load_part', read_part)
    train = s05.train_model
    def record_train(*args, **kwargs):
        events.append(('train', args[9], args[1].shape[0]))
        return train(*args, **kwargs)
    monkeypatch.setattr(s05, 'train_model', record_train)
    return cfg, manifest, verification, loaded, events


def freeze(setup):
    cfg, manifest, verification, loaded, _ = setup
    return s06.freeze_training_protocol(cfg, manifest, verification, loaded, torch.device('cpu'))


def test_selection_only_uses_fit_and_monitor_and_freezes_once(setup):
    cfg, _, _, _, events = setup
    protocol = freeze(setup)
    assert [e[1] for e in events if e[0] == 'read'] == ['train_fit', 'train_monitor']
    assert [e[2] for e in events if e[0] == 'train'] == [2]
    assert protocol['fixed_epochs'] == protocol['selection_report']['candidates'][0]['best_epoch']
    before = open(cfg.protocol_path, 'rb').read()
    with pytest.raises(s06.S06Gate1Error):
        freeze(setup)
    assert open(cfg.protocol_path, 'rb').read() == before


@pytest.mark.parametrize('change', ['split', 'tfidf', 'corpus', 'seeds'])
def test_mismatched_provenance_or_config_blocks_refit_before_training(setup, monkeypatch, change):
    cfg, manifest, verification, loaded, events = setup
    freeze(setup)
    events.clear()
    if change == 'split':
        manifest['parts']['train_fit']['sha256'] = 'different'
    elif change == 'tfidf':
        verification['full_tfidf_id'] = 'different'
    elif change == 'corpus':
        verification['corpus_fingerprint']['sha256'] = 'different'
    elif change == 'seeds':
        cfg.seeds = [42, 43]
    monkeypatch.setattr(s06, 'run_refit', lambda *a, **k: pytest.fail('refit started'))
    with pytest.raises(s06.S06Gate1Error):
        s06.run_gate1_refit(cfg, manifest, verification, loaded, torch.device('cpu'))
    assert not events


def test_refit_trains_all_seeds_before_validation_and_never_reads_test(setup):
    cfg, manifest, verification, loaded, events = setup
    protocol = freeze(setup)
    events.clear()
    s06.run_gate1_refit(cfg, manifest, verification, loaded, torch.device('cpu'))
    val_at = events.index(('read', 'comparison_val'))
    assert [e[1] for e in events[:val_at] if e[0] == 'train'] == [42, 43, 44]
    assert all(e[2] == 4 for e in events if e[0] == 'train')
    assert ('read', 'sealed_test') not in events
    selected = json.loads((__import__('pathlib').Path(cfg.baseline_dir) / 'selected_checkpoint.json').read_text())
    assert selected['run_id'] == 'seed_42'
    assert protocol['fixed_epochs'] >= 1


@pytest.mark.parametrize('scores', [{42: 0.1, 43: 0.9, 44: 0.8}, {42: 0.9, 43: 0.1, 44: 0.8}])
def test_checkpoint_reference_seed_independent_of_validation_scores(setup, monkeypatch, scores):
    cfg, manifest, _, loaded, _ = setup
    selected_paths = []
    class StopAfterSelection(Exception):
        pass
    def capture(path, device):
        selected_paths.append(path)
        raise StopAfterSelection
    monkeypatch.setattr(s05, 'load_artifact', capture)
    results = {seed: {'status': 'trained'} for seed in scores}
    evaluations = {seed: {'l3_macro_f1': score} for seed, score in scores.items()}
    with pytest.raises(StopAfterSelection):
        s05._select_and_verify_checkpoint(cfg.to_s05_config(), manifest, loaded, None,
                                          '/checkpoints', results, evaluations, [], (), [],
                                          torch.device('cpu'), (1, 1, 2), 2)
    assert selected_paths == ['/checkpoints/seed_42']


def test_gate0_rejects_old_policy_and_missing_integrity(setup, tmp_path, monkeypatch):
    cfg, manifest, _, _, _ = setup
    (tmp_path / 'split.json').write_text('{}')
    manifest['checks'] = {'ok': True}
    monkeypatch.setattr(s06, 'load_split_manifest', lambda _: manifest)
    assert s06.verify_gate0_split(cfg) is manifest
    for mutation in ('policy', 'checks', 'parts', 'label_maps'):
        bad = copy.deepcopy(manifest)
        if mutation == 'policy': bad['policy']['min_count'] = 10
        elif mutation == 'checks': bad['checks']['ok'] = False
        elif mutation == 'parts': del bad['parts']['sealed_test']
        else: bad['label_maps'] = None
        monkeypatch.setattr(s06, 'load_split_manifest', lambda _, bad=bad: bad)
        with pytest.raises(s06.S06Gate1Error):
            s06.verify_gate0_split(cfg)
