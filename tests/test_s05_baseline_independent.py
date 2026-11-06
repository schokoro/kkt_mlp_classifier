"""S05 protocol checks on synthetic data; no ODS or external model artifacts."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from kkt_clf import s05_baseline as s05
from kkt_clf.config import BranchConfig, FeatureConfig, LossConfig, ModelConfig, TrainConfig
from kkt_clf.features import FeatureBuilder
from kkt_clf.labels import LabelMaps
from kkt_clf.model import SparseMLP
from kkt_clf.train import set_seed, train_model


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    df = pd.DataFrame({'name': ['alpha milk', 'beta bread', 'alpha food', 'beta food', 'gamma'],
                       'L1': ['root'] * 5, 'L2': ['group'] * 5,
                       'L3': ['a', 'b', 'a', 'b', 'c']})
    maps = LabelMaps.from_df(df)
    parts = {'outer_train': df.iloc[:4], 'train_fit': df.iloc[:2],
             'train_monitor': df.iloc[2:4], 'comparison_val': df.iloc[[0, 1]].copy()}
    parts['comparison_val'].loc[parts['comparison_val'].index[1], 'name'] = ''
    manifest = {'label_maps': maps.to_dict(), 'policy': {'split_seed': 7},
                'parts': {k: {'sha256': k, 'bytes': 1, 'n_rows': len(v)} for k, v in parts.items()},
                'inputs': {'labeled_parquet': {'sha256': 'source'}}}
    manifest['parts']['sealed_test'] = {'sha256': 'sealed', 'bytes': 1, 'n_rows': 1}
    builder = FeatureBuilder(FeatureConfig(BranchConfig((2, 3), 1), BranchConfig((1, 1), 1), None, 10))
    builder.fit(['alpha milk', 'beta bread', 'alpha food', 'beta food', 'gamma'], seed=3)
    loaded = SimpleNamespace(builder=builder, meta={'files': {'features.joblib': {'sha256': 'full-id'}}})
    cfg = s05.S05Config(output_dir=str(tmp_path / 'result'), splits_manifest=str(tmp_path / 'split.json'),
                        seeds=[42, 43], reference_seed=42, fixed_epochs=2, device='cpu',
                        model={'hidden': [4], 'dropout': [0.0]},
                        train={'scheduler': 'none', 'batch_size': 2, 'epochs': 2, 'patience': 2})
    (tmp_path / 'split.json').write_text('{}')
    events = []
    def read_part(_manifest, name):
        events.append(('read', name))
        if name == 'sealed_test':
            pytest.fail('sealed test data were read')
        return parts[name].copy()
    monkeypatch.setattr(s05, 'load_part', read_part)
    original_train = s05.train_model
    def record_train(*args, **kwargs):
        events.append(('train', args[9]))
        return original_train(*args, **kwargs)
    monkeypatch.setattr(s05, 'train_model', record_train)
    return cfg, manifest, loaded, maps, events


def test_selection_uses_only_internal_train(synthetic):
    cfg, manifest, loaded, _, events = synthetic
    report = s05.run_selection(cfg, manifest, loaded, torch.device('cpu'))
    assert [x[1] for x in events if x[0] == 'read'] == ['train_fit', 'train_monitor']
    assert report['chosen']['fixed_epochs'] == report['candidates'][0]['best_epoch']
    assert report['full_tfidf_id'] == 'full-id'


def test_refit_does_not_read_validation_until_all_seeds_finished(synthetic):
    cfg, manifest, loaded, _, events = synthetic
    s05.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                       tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    val_index = events.index(('read', 'comparison_val'))
    assert [e[1] for e in events[:val_index] if e[0] == 'train'] == cfg.seeds
    assert ('read', 'sealed_test') not in events


def test_refit_metrics_checkpoint_and_feature_identity(synthetic, monkeypatch):
    cfg, manifest, loaded, maps, events = synthetic
    def no_fit(*args, **kwargs):
        pytest.fail('pretrained TF-IDF was fitted again')
    monkeypatch.setattr(FeatureBuilder, 'fit', no_fit)
    before = loaded.builder.transform(['alpha milk', '']).copy()
    s05.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                       tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    from kkt_clf.artifact import load_artifact
    out = __import__('pathlib').Path(cfg.output_dir)
    runs = pd.read_csv(out / 'runs.csv')
    assert list(runs.status) == ['success', 'success']
    assert set(runs.full_tfidf_id) == {'full-id'}
    assert set(runs.epochs_completed) == {2}
    classes = pd.read_parquet(out / 'per_class.parquet')
    for _, frame in classes.groupby('run_id'):
        assert set(frame.L3) == set(maps.l3)
        absent = frame.set_index('L3').loc['c']
        assert absent.support == 0 and pd.isna(absent.f1)
    selected = json.loads((out / 'selected_checkpoint.json').read_text())
    assert selected['run_id'] == 'seed_42' and selected['roundtrip']['metrics_match']
    art = load_artifact(str(out / 'checkpoints' / 'seed_42'), torch.device('cpu'))
    assert art.meta['train']['n_train_rows_used'] == 4
    assert art.meta['train']['restored_best'] is False
    assert art.maps.to_dict() == maps.to_dict()
    assert (art.features.transform(['alpha milk', '']) != before).nnz == 0
    result = json.loads((out / 'refit' / 'run_result_seed_42.json').read_text())
    assert result['comparison_val_eval']['n_empty_features_val'] == 1
    assert result['comparison_val_eval']['l3_accuracy'] == runs.iloc[0].l3_accuracy


def test_fixed_training_final_epoch_and_seed_reproducibility():
    X = sp.csr_matrix(np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32))
    y = (np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), np.array([0, 1, 0, 1]))
    model_cfg = ModelConfig(hidden=[4], dropout=[0.0])
    def run(seed, epochs, restore):
        set_seed(seed)
        model = SparseMLP(2, (1, 1, 2), model_cfg)
        cfg = TrainConfig(epochs=epochs, batch_size=2, patience=10, scheduler='none', lr=0.01)
        history, best, _ = train_model(model, X, y, X, y, LossConfig(), cfg,
                                        {'l1': None, 'l2': None, 'l3': None}, torch.device('cpu'), seed,
                                        restore_best=restore)
        return model, history, best
    a, hist, best = run(42, 3, False)
    b, hist2, _ = run(42, 3, False)
    c, _, _ = run(43, 3, False)
    assert len(hist) == 3 and best['n_train_rows_used'] == 3 and not best['restored_best']
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict())
    assert [x['val_l3_accuracy'] for x in hist] == [x['val_l3_accuracy'] for x in hist2]
    if best['epoch'] < 3:
        restored, _, _ = run(42, 3, True)
        assert any(not torch.equal(a.state_dict()[k], restored.state_dict()[k]) for k in a.state_dict())


def test_failed_seed_records_null_metrics(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    original = s05.train_model
    def fail_second(*args, **kwargs):
        if args[9] == 43:
            raise RuntimeError('synthetic failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(s05, 'train_model', fail_second)
    s05.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                       tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    runs = pd.read_csv(__import__('pathlib').Path(cfg.output_dir) / 'runs.csv').set_index('train_seed')
    assert runs.loc[43, 'status'] == 'failed'
    assert pd.isna(runs.loc[43, 'l3_macro_f1']) and pd.isna(runs.loc[43, 'l3_accuracy'])
    assert pd.isna(runs.loc[43, 'epochs_completed']) and pd.isna(runs.loc[43, 'checkpoint_path'])


def test_raw_top1_metrics_keep_unsupported_class_null():
    # Class b is never predicted; c is absent in validation but remains in the universe.
    macro, accuracy, rows = s05._l3_val_metrics(np.array([0, 0, 1]), np.array([0, 1, 0]), ['a', 'b', 'c'])
    by_class = {row['L3']: row for row in rows}
    assert accuracy == pytest.approx(1 / 3)
    assert macro == pytest.approx(0.25)
    assert by_class['c']['support'] == 0
    assert by_class['c']['f1'] is None


def test_all_failed_seeds_still_have_report_rows(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    def fail(*args, **kwargs):
        raise RuntimeError('all seeds failed')
    monkeypatch.setattr(s05, 'train_model', fail)
    try:
        s05.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                       tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    except s05.S05BaselineError:
        pass
    from pathlib import Path
    path = Path(cfg.output_dir) / 'runs.csv'
    assert path.exists(), 'all planned failed runs must remain visible in runs.csv'
    runs = pd.read_csv(path)
    assert set(runs.train_seed) == set(cfg.seeds)
    assert set(runs.status) == {'failed'}
    assert runs.l3_macro_f1.isna().all() and runs.l3_accuracy.isna().all()


def test_refit_omitted_verification_rejected_before_training(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    monkeypatch.setattr(s05, 'train_model',
                        lambda *args, **kwargs: pytest.fail('training without TF-IDF verification'))
    with pytest.raises((s05.S05BaselineError, ValueError)):
        s05.run_refit(cfg, manifest, loaded, torch.device('cpu'))
