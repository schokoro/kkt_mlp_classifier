"""Independent synthetic S08 sources; never read production data or holdout labels."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from kkt_clf.config import BranchConfig, FeatureConfig, LossConfig, ModelConfig, TrainConfig
from kkt_clf.vectorizer import fit_and_save_vectorizer


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False), encoding='utf-8')


def reseal(value):
    value = {k: v for k, v in value.items() if k != 'integrity_sha256'}
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return {**value, 'integrity_sha256': hashlib.sha256(raw.encode()).hexdigest()}


@pytest.fixture
def sources(tmp_path):
    pool = pd.DataFrame([
        {'row_id': c * 100 + i, 'name': f'{label} product{chr(97+i)} fresh',
         'L1': 'root', 'L2': 'branch', 'L3': label}
        for c, label in enumerate(('alpha', 'beta')) for i in range(10)
    ])
    corpus = tmp_path / 'corpus.parquet'
    pd.DataFrame({'name': pool.name.tolist() + ['additional assorted grocery token']}).to_parquet(corpus, index=False)
    full = tmp_path / 'full'
    feature_config = FeatureConfig(char=BranchConfig((2, 3), 1, 24, True),
                                   word=BranchConfig((1, 1), 1, 8, True), vectorizer_fit_rows=None)
    fit_and_save_vectorizer(str(full), feature_config, str(corpus), max_rows=100, seed=42)
    maps = {'L1': ['root'], 'L2': ['branch'], 'L3': ['alpha', 'beta'],
            'parent_l2_of_l3': [0, 0], 'parent_l1_of_l2': [0]}
    frames = {'outer_train': pool.groupby('L3', sort=False).head(8),
              'comparison_val': pool.groupby('L3', sort=False).tail(2),
              'train_fit': pool.groupby('L3', sort=False).head(6),
              'train_monitor': pool.groupby('L3', sort=False).nth([6, 7]).reset_index(drop=True),
              # No labels exist in this fixture. Even a successful model cannot score it.
              'sealed_test': pd.DataFrame({'row_id': [900, 901, 902, 903],
                                           'name': ['closed one', 'closed two', 'closed three', 'closed four']})}
    parts = {}
    for name, frame in frames.items():
        path = tmp_path / f'{name}.parquet'
        frame.to_parquet(path, index=False)
        parts[name] = {'file': str(path), 'sha256': file_sha(path),
                       'bytes': path.stat().st_size, 'n_rows': len(frame), 'n_groups': len(frame)}
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, {'policy': {'min_count': 12}, 'parts': parts, 'label_maps': maps})
    protocol = tmp_path / 'source_protocol.json'
    write_json(protocol, {'frozen': True, 'fixed_epochs': 14, 'chosen': {'fixed_epochs': 14},
                         'full_tfidf_id': file_sha(full / 'features.joblib'),
                         'corpus_fingerprint': {'sha256': file_sha(corpus)}, 'splits': parts,
                         'resolved_sections': {
                             'model': dataclasses.asdict(ModelConfig([4], [0.])),
                             'loss': dataclasses.asdict(LossConfig()),
                             # Historical selection schedule intentionally differs.
                             'train': dataclasses.asdict(TrainConfig(epochs=15, scheduler='none',
                                                                    batch_size=8, num_threads=1))}})
    return {'pool': pool, 'corpus': corpus, 'full': full, 'manifest': manifest,
            'source_protocol': protocol, 'parts': parts, 'maps': maps, 'root': tmp_path / 'study',
            'pins': {name: file_sha(path) for name, path in {
                'manifest': manifest, 'protocol': protocol, 'corpus': corpus,
                'features': full / 'features.joblib'}.items()}}
