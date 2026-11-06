"""Integrity and reuse of independently generated G×I rating files."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

from kkt_clf.attribution import MarginAttributions, build_gxi_ratings, select_features_gradient_x_input
from kkt_clf.selection import (SelectionArtifactError, SelectionConfig, SelectionError,
                               load_selection_artifact, save_selection_artifact)


@pytest.fixture
def ratings(tmp_path):
    values = sp.csr_matrix([[3., 1., 0., 2., 1., 0.],
                            [1., 0., 2., 0., 1., 3.],
                            [0., 2., 1., 1., 0., 2.],
                            [1., 0., 0., 0., 0., 0.]])
    att = MarginAttributions(values, np.array([1., -1., 2., 0.]),
                             np.array([1, 0, 0, 1]),
                             np.array([True, False, True, False]),
                             np.zeros(4, dtype=bool), np.array([0, 0, 1, 1]),
                             np.arange(4), 2)
    path = tmp_path / 'ratings'
    manifest, _ = build_gxi_ratings(att, 3, str(path))
    return path, manifest


@pytest.fixture
def config():
    return SelectionConfig(method='gradient_x_input', k=4, char_share=.5, seed=19)


def select(path, config):
    return select_features_gradient_x_input(str(path), config, 3, 6)


def test_manifest_files_have_consistent_sizes_and_class_counts(ratings):
    path, manifest = ratings
    assert json.loads((path / 'meta.json').read_text()) == manifest
    for branch in ('char', 'word'):
        entry = manifest['branches'][branch]
        for key in ('classes', 'class_counts', 'rank_counts', 'rank_indices',
                    'rank_scores', 'global_scores'):
            assert (path / entry[key]).is_file()
            assert (path / entry[key]).stat().st_size % 8 == 0
        classes = np.fromfile(path / entry['classes'], dtype=np.int64)
        n_rows = np.fromfile(path / entry['class_counts'], dtype=np.int64)
        counts = np.fromfile(path / entry['rank_counts'], dtype=np.int64)
        indices = np.fromfile(path / entry['rank_indices'], dtype=np.int64)
        scores = np.fromfile(path / entry['rank_scores'], dtype=np.float64)
        global_scores = np.fromfile(path / entry['global_scores'], dtype=np.float64)
        np.testing.assert_array_equal(classes, [0, 1])
        np.testing.assert_array_equal(n_rows, [2, 2])
        assert len(counts) == len(classes) == len(n_rows)
        assert len(indices) == len(scores) == counts.sum()
        assert len(global_scores) == entry['n_features'] == 3
        assert np.all((0 <= indices) & (indices < entry['n_features']))


def test_saved_selection_can_reselect_identical_indices_and_scores(ratings, config, tmp_path):
    path, _ = ratings
    original = select(path, config)
    out = tmp_path / 'selection'
    save_selection_artifact(str(out), original, {'vectorizer': 'fixture'}, ratings_copy=True)
    restored, _ = load_selection_artifact(str(out))
    reselected = select(restored.ratings_dir, config)
    for branch in ('char', 'word'):
        expected = getattr(original, branch)
        for result in (restored, reselected):
            actual = getattr(result, branch)
            np.testing.assert_array_equal(actual.indices, expected.indices)
            np.testing.assert_array_equal(actual.scores, expected.scores)
            assert actual.origin == expected.origin


@pytest.mark.parametrize('mode', ['direct', 'artifact'])
@pytest.mark.parametrize('branch', ['char', 'word'])
@pytest.mark.parametrize('key', ['classes', 'class_counts', 'rank_counts', 'rank_indices',
                                 'rank_scores', 'global_scores'])
def test_missing_declared_file_rejected_by_direct_selection_and_artifact_load(
        ratings, config, tmp_path, branch, key, mode):
    path, manifest = ratings
    result = select(path, config)
    out = tmp_path / 'selection'
    save_selection_artifact(str(out), result, {'vectorizer': 'fixture'}, ratings_copy=True)
    filename = manifest['branches'][branch][key]
    (path / filename).unlink()
    (out / 'ratings' / filename).unlink()
    if mode == 'direct':
        with pytest.raises(SelectionError):
            select(path, config)
    else:
        with pytest.raises(SelectionArtifactError):
            load_selection_artifact(str(out))


@pytest.mark.parametrize('branch', ['char', 'word'])
@pytest.mark.parametrize('damage', ['short_class_counts', 'negative_class_count',
                                    'short_rank_counts', 'negative_rank_count',
                                    'oversized_rank_count', 'short_indices', 'short_scores',
                                    'partial_indices', 'short_global', 'short_classes'])
def test_malformed_rating_arrays_rejected(ratings, config, branch, damage):
    path, manifest = ratings
    entry = manifest['branches'][branch]
    key, action = {
        'short_class_counts': ('class_counts', 'short'),
        'negative_class_count': ('class_counts', 'negative'),
        'short_rank_counts': ('rank_counts', 'short'),
        'negative_rank_count': ('rank_counts', 'negative'),
        'oversized_rank_count': ('rank_counts', 'oversized'),
        'short_indices': ('rank_indices', 'short'),
        'short_scores': ('rank_scores', 'short'),
        'partial_indices': ('rank_indices', 'partial'),
        'short_global': ('global_scores', 'short'),
        'short_classes': ('classes', 'short'),
    }[damage]
    file = path / entry[key]
    data = file.read_bytes()
    if action == 'short':
        file.write_bytes(data[:-8])
    elif action == 'partial':
        file.write_bytes(data[:-1])
    else:
        values = np.fromfile(file, dtype=np.int64)
        values[0] = -1 if action == 'negative' else entry['n_features'] + 1
        values.tofile(file)
    with pytest.raises(SelectionError):
        select(path, config)
