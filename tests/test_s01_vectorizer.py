import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from kkt_clf import Predictor, build_config, run_training, verify_roundtrip
from kkt_clf.config import build_feature_config
from kkt_clf.data import DataValidationError
from kkt_clf.features import FeatureBuilder
from kkt_clf.vectorizer import (VectorizerArtifactError, VectorizerError,
    fit_and_save_vectorizer, load_name_corpus, load_vectorizer_artifact)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / 'corpus.parquet'
    pd.DataFrame({'name': ['Молоко белое', None, '', 'Хлеб ржаной', 'Сыр твёрдый', 'Чай зелёный']}).to_parquet(path, index=False)
    return path


@pytest.fixture
def feature_cfg():
    return build_feature_config('ods_light', {'vectorizer_fit_rows': None})


def assert_features_equal(a, b, names):
    for branch in ('char', 'word'):
        x, y = getattr(a, branch), getattr(b, branch)
        np.testing.assert_array_equal(x.get_feature_names_out(), y.get_feature_names_out())
        np.testing.assert_allclose(x.idf_, y.idf_, rtol=1e-7, atol=1e-7)
    x, y = a.transform(names), b.transform(names)
    assert sp.isspmatrix_csr(x) and sp.isspmatrix_csr(y)
    assert x.shape == y.shape == (len(names), a.n_features)
    np.testing.assert_allclose((x - y).data, 0, atol=1e-7)


def test_standalone_cli_and_bounded_read(corpus, tmp_path, monkeypatch):
    # ParquetFile must request only name even if label columns exist.
    import pyarrow.parquet as pq
    labeled = tmp_path / 'extra.parquet'
    pd.DataFrame({'name': ['one', None, 'two', 'three'], 'L3': ['secret'] * 4}).to_parquet(labeled, index=False)
    original = pq.ParquetFile.iter_batches
    calls = []
    def read(self, *args, **kwargs):
        calls.append(kwargs)
        assert kwargs['columns'] == ['name']
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', read)
    a = load_name_corpus(str(labeled), 2)
    b = load_name_corpus(str(labeled), 2)
    assert a.names == b.names == ['one', 'two']
    assert calls and a.rows_read == 3
    out = tmp_path / 'vectorizer'
    cmd = [sys.executable, 'scripts/fit_vectorizer.py', '--input', str(corpus), '--output', str(out),
           '--features-preset', 'ods_light', '--max-rows', '3', '--source-id', 'sample-v1',
           '--override', json.dumps({'char': {'min_df': 1}})]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    loaded = load_vectorizer_artifact(str(out))
    assert loaded.meta['source']['rows_used'] == 3
    assert loaded.meta['source']['source_id'] == 'sample-v1'
    assert loaded.meta['source']['max_rows'] == 3


def test_large_row_group_uses_bounded_batches(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'large.parquet'
    pq.write_table(pa.table({'name': [f'name {i}' for i in range(10000)],
                             'L3': ['secret'] * 10000}), path, row_group_size=10000)
    assert pq.ParquetFile(path).num_row_groups == 1
    original = pq.ParquetFile.iter_batches
    batch_sizes = []
    def read(self, *args, **kwargs):
        assert kwargs['columns'] == ['name']
        assert kwargs['batch_size'] <= 7
        for batch in original(self, *args, **kwargs):
            batch_sizes.append(batch.num_rows)
            yield batch
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', read)
    result = load_name_corpus(str(path), 7)
    assert result.names == [f'name {i}' for i in range(7)]
    assert batch_sizes and max(batch_sizes) <= 7
    assert result.truncated is True


@pytest.mark.parametrize('max_rows', [None, 0, -1])
def test_invalid_corpus_limit(corpus, max_rows):
    with pytest.raises(ValueError, match='max_rows'):
        load_name_corpus(str(corpus), max_rows)


def test_truncated_when_limit_reached_on_last_row(tmp_path):
    path = tmp_path / 'names.parquet'
    pd.DataFrame({'name': ['one', None, 'two']}).to_parquet(path, index=False)
    exact = load_name_corpus(str(path), 2)
    assert exact.rows_read == 3
    assert exact.truncated is False
    assert load_name_corpus(str(path), 1).truncated is True


def test_artifact_metadata_and_checksum(corpus, feature_cfg, tmp_path, monkeypatch):
    import kkt_clf.vectorizer as vectorizer

    path = tmp_path / 'artifact'
    builder, _ = fit_and_save_vectorizer(str(path), feature_cfg, str(corpus),
                                         max_rows=4, seed=123, source_id='sample-v1')
    meta_path = path / 'vectorizer_meta.json'
    meta = json.loads(meta_path.read_text())
    data = (path / 'features.joblib').read_bytes()
    assert meta['files']['features.joblib'] == {
        'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    assert meta['seed'] == 123
    assert meta['source']['source_id'] == 'sample-v1'
    assert meta['source']['rows_used'] == 4
    assert meta['source']['max_rows'] == 4
    assert meta['source']['strategy'] == 'head'
    assert meta['tfidf_config']['char'] and meta['tfidf_config']['word']
    assert meta['vocab'] == {'n_char': builder.n_char_features,
                             'n_word': builder.n_word_features, 'n_total': builder.n_features}
    assert meta['notes'] and meta['versions'] and meta['fit']
    for key in ('artifact_version', 'seed', 'source', 'tfidf_config', 'vocab'):
        modified = dict(meta)
        del modified[key]
        meta_path.write_text(json.dumps(modified))
        with pytest.raises(VectorizerArtifactError, match='обязательн'):
            load_vectorizer_artifact(str(path))
    meta_path.write_text(json.dumps(meta))
    (path / 'features.joblib').write_bytes(data + b'corruption')
    def forbidden(*args, **kwargs):
        pytest.fail('checksum must be checked before deserialization')
    monkeypatch.setattr(vectorizer.joblib, 'load', forbidden)
    with pytest.raises(VectorizerArtifactError, match='Контрольная сумма'):
        load_vectorizer_artifact(str(path))


def test_output_directory_checked_before_fit(corpus, feature_cfg, tmp_path, monkeypatch):
    path = tmp_path / 'existing'
    path.mkdir()
    sentinel = path / 'user.txt'
    sentinel.write_text('keep')
    def forbidden(*args, **kwargs):
        pytest.fail('fit must not run for a nonempty output directory')
    monkeypatch.setattr(FeatureBuilder, 'fit', forbidden)
    with pytest.raises(VectorizerArtifactError):
        fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=3)
    assert sentinel.read_text() == 'keep'


def test_cli_reports_actual_fit_size(corpus, tmp_path):
    out = tmp_path / 'limited'
    result = subprocess.run([
        sys.executable, 'scripts/fit_vectorizer.py', '--input', str(corpus),
        '--output', str(out), '--features-preset', 'ods_light', '--max-rows', '4',
        '--override', json.dumps({'vectorizer_fit_rows': 2}),
    ], check=True, capture_output=True, text=True)
    # Logging precedes the JSON object in stdout.
    report = json.loads(result.stdout[result.stdout.index('{'):])
    meta = json.loads((out / 'vectorizer_meta.json').read_text())
    assert meta['source']['rows_read'] == 6
    assert meta['source']['rows_used'] == 4
    assert meta['fit']['rows_in'] == 4
    assert meta['fit']['rows_fit'] == 2
    assert report['rows_read'] == 6
    assert report['rows_used'] == 4
    assert report['rows_fit'] == 2


@pytest.mark.parametrize('section,key,value', [
    ('files', 'features.joblib', 'invalid'),
    ('files', 'features.joblib', {'sha256': 123, 'bytes': 1}),
    ('versions', 'sklearn', ['invalid']),
    ('source', 'rows_read', 'invalid'),
    ('vocab', 'n_char', 'invalid'),
    ('fit', 'rows_fit', 'invalid'),
    ('tfidf_config', 'char', 'invalid'),
])
def test_corrupt_nested_metadata_types(corpus, feature_cfg, tmp_path, section, key, value):
    path = tmp_path / 'artifact'
    fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    meta_path = path / 'vectorizer_meta.json'
    meta = json.loads(meta_path.read_text())
    meta[section][key] = value
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))


@pytest.mark.parametrize('section,key', [
    ('source', 'rows_read'), ('source', 'rows_used'),
    ('vocab', 'n_char'), ('vocab', 'n_word'), ('vocab', 'n_total'),
    ('fit', 'rows_fit'), ('tfidf_config', 'char'), ('tfidf_config', 'word'),
    ('files', 'features.joblib'),
])
def test_missing_nested_metadata(corpus, feature_cfg, tmp_path, section, key):
    path = tmp_path / 'artifact'
    fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    meta_path = path / 'vectorizer_meta.json'
    meta = json.loads(meta_path.read_text())
    del meta[section][key]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))


def make_cfg(tmp_path, mode='train', vectorizer=None):
    path = tmp_path / 'labeled.parquet'
    if not path.exists():
        pd.DataFrame([{'name': f'item{c} token{i}', 'L1': 'root', 'L2': f'group{c}', 'L3': f'class{c}'}
                      for c in range(3) for i in range(20)]).to_parquet(path, index=False)
    return build_config('ods_light', {'output_dir': str(tmp_path / ('out_' + mode)),
        'data': {'path': str(path), 'min_count': 5},
        'features': {'mode': mode, 'pretrained_path': vectorizer, 'vectorizer_fit_rows': 20,
                     'char': {'min_df': 1, 'max_features': 100}, 'word': {'min_df': 1, 'max_features': 50}},
        'model': {'hidden': [8], 'dropout': [0.0]},
        'train': {'epochs': 1, 'batch_size': 16, 'device': 'cpu', 'num_threads': 1}})


@pytest.mark.parametrize('same_as_pretrained', [False, True])
def test_training_rejects_nonempty_output_before_writing(corpus, feature_cfg, tmp_path, same_as_pretrained):
    path = tmp_path / 'existing'
    if same_as_pretrained:
        fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
        cfg = make_cfg(tmp_path, 'pretrained', str(path))
    else:
        path.mkdir()
        (path / 'reports').mkdir()
        (path / 'reports' / 'rare_l3_classes.csv').write_bytes(b'user report')
        cfg = make_cfg(tmp_path)
    (path / 'user.txt').write_bytes(b'user data')
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    directories_before = {p.relative_to(path) for p in path.rglob('*') if p.is_dir()}
    cfg.output_dir = str(path)
    with pytest.raises((ValueError, VectorizerError), match='(?i)(не пуст|non.?empty|exist)'):
        run_training(cfg)
    after = {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    assert after == before
    assert {p.relative_to(path) for p in path.rglob('*') if p.is_dir()} == directories_before


def test_baseline_fit_only_train_and_parameters(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    original = FeatureBuilder.fit
    calls = []
    def capture(self, texts, seed=0):
        calls.append((list(texts), seed, self.cfg))
        return original(self, texts, seed)
    monkeypatch.setattr(FeatureBuilder, 'fit', capture)
    result = run_training(cfg)
    assert len(calls) == 1
    texts, seed, params = calls[0]
    assert texts == result.data.splits['train']['name'].tolist()
    assert set(texts).isdisjoint(result.data.splits['val']['name'])
    assert set(texts).isdisjoint(result.data.splits['test']['name'])
    assert seed == cfg.seed and params.vectorizer_fit_rows == 20
    p = Predictor.load(result.artifact_dir, 'cpu')
    assert p.features.fit_report_['rows_fit'] == 20
    for branch in ('char', 'word'):
        vec, expected = getattr(p.features, branch), getattr(cfg.features, branch)
        assert vec.min_df == expected.min_df and vec.max_features == expected.max_features
        assert vec.ngram_range == expected.ngram_range
    assert verify_roundtrip(result)


def test_pretrained_reuse_and_model_roundtrip(corpus, feature_cfg, tmp_path, monkeypatch):
    path = tmp_path / 'standalone'
    before, _ = fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    loaded = load_vectorizer_artifact(str(path)).builder
    names = ['МОЛОКО белое', 'неизвестное', '', None]
    assert_features_equal(before, loaded, names)
    def forbidden(*args, **kwargs):
        pytest.fail('pretrained training must not fit')
    monkeypatch.setattr(FeatureBuilder, 'fit', forbidden)
    cfg = make_cfg(tmp_path, 'pretrained', str(path))
    for i in range(2):
        cfg.output_dir = str(tmp_path / f'run{i}')
        result = run_training(cfg)
        predictor = Predictor.load(result.artifact_dir, 'cpu')
        assert_features_equal(before, predictor.features, names)
        assert predictor.model.input.in_features == before.n_features
        assert verify_roundtrip(result)


@pytest.mark.parametrize('kind', ['missing_name', 'empty', 'invalid'])
def test_bad_corpus(tmp_path, feature_cfg, kind):
    path = tmp_path / 'bad.parquet'
    if kind == 'missing_name':
        pd.DataFrame({'L3': ['x']}).to_parquet(path)
    elif kind == 'empty':
        pd.DataFrame({'name': ['', None]}).to_parquet(path)
    else:
        path.write_text('not parquet')
    with pytest.raises((DataValidationError, VectorizerError)):
        fit_and_save_vectorizer(str(tmp_path / 'out'), feature_cfg, str(path), max_rows=2)


def test_artifact_errors_and_no_overwrite(corpus, feature_cfg, tmp_path):
    path = tmp_path / 'artifact'
    builder, meta = fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    sentinel = (path / 'features.joblib').read_bytes()
    with pytest.raises(VectorizerArtifactError):
        fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    assert (path / 'features.joblib').read_bytes() == sentinel
    other_cfg = build_feature_config('ods_light', {'char': {'min_df': 1}})
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path), expected_cfg=other_cfg)
    meta_path = path / 'vectorizer_meta.json'
    saved = json.loads(meta_path.read_text())
    saved['artifact_version'] = 'incompatible'
    meta_path.write_text(json.dumps(saved))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))
    saved['artifact_version'] = meta['artifact_version']
    meta_path.write_text(json.dumps(saved))
    (path / 'features.joblib').write_bytes(b'corrupted')
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(tmp_path / 'missing'))


@pytest.mark.parametrize('overrides', [
    {'mode': 'unknown'}, {'mode': 'pretrained'}, {'vectorizer_fit_rows': 0},
    {'transform_chunk_rows': 0}, {'char': {'ngram_range': [3, 1]}},
    {'word': {'max_features': 0}},
])
def test_invalid_feature_config(overrides):
    with pytest.raises(ValueError):
        build_feature_config('ods_light', overrides)


@pytest.mark.parametrize('field,value', [
    ('seed', None), ('seed', True), ('seed', '42'),
    ('fit', None), ('fit', {}),
    ('fit', {'rows_in': 4, 'rows_fit': 4}),
    ('fit', {'rows_in': 4, 'rows_fit': 0, 'subsampled': True}),
    ('fit', {'rows_in': 4, 'rows_fit': 3, 'subsampled': False}),
])
def test_rejects_incomplete_or_corrupt_fit_and_seed(corpus, feature_cfg, tmp_path, field, value):
    path = tmp_path / 'artifact'
    fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    meta_path = path / 'vectorizer_meta.json'
    meta = json.loads(meta_path.read_text())
    meta[field] = value
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))


@pytest.mark.parametrize('report', [None, {},
    {'rows_in': 4, 'rows_fit': 3, 'subsampled': True},
])
def test_rejects_builder_fit_report_inconsistent_with_metadata(corpus, feature_cfg, tmp_path, report):
    import joblib

    path = tmp_path / 'artifact'
    fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    features_path = path / 'features.joblib'
    builder = joblib.load(features_path)
    builder.fit_report_ = report
    joblib.dump(builder, features_path)
    data = features_path.read_bytes()
    meta_path = path / 'vectorizer_meta.json'
    meta = json.loads(meta_path.read_text())
    meta['files']['features.joblib'] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))


def test_training_rejects_output_inside_pretrained_via_symlink(corpus, feature_cfg, tmp_path):
    path = tmp_path / 'pretrained'
    fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    alias = tmp_path / 'alias'
    alias.symlink_to(path, target_is_directory=True)
    cfg = make_cfg(tmp_path, 'pretrained', str(path))
    cfg.output_dir = str(alias / 'new-output')
    with pytest.raises((ValueError, VectorizerError)):
        run_training(cfg)
    assert not (path / 'new-output').exists()
    assert {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()} == before
