"""Independent checks for completeness of reusable rating manifests."""
import json

import numpy as np
import pytest

from kkt_clf.selection import SelectionError
from tests.test_s04_ratings_integrity import config, ratings, select


_KEYS = ('classes', 'class_counts', 'rank_counts', 'rank_indices',
         'rank_scores', 'global_scores')


def _write_manifest(path, manifest):
    (path / 'meta.json').write_text(json.dumps(manifest), encoding='utf8')


@pytest.mark.parametrize('branch', ['char', 'word'])
@pytest.mark.parametrize('key', _KEYS)
def test_present_checksums_must_cover_every_declared_file(ratings, config, branch, key):
    path, manifest = ratings
    select(path, config)  # The intact fixture is usable.
    del manifest['files'][manifest['branches'][branch][key]]
    _write_manifest(path, manifest)
    with pytest.raises(SelectionError):
        select(path, config)


def test_empty_checksum_section_is_not_a_legacy_manifest(ratings, config):
    path, manifest = ratings
    manifest['files'] = {}
    _write_manifest(path, manifest)
    with pytest.raises(SelectionError):
        select(path, config)


def test_legacy_manifest_without_checksum_section_is_supported(ratings, config):
    path, manifest = ratings
    del manifest['files']
    _write_manifest(path, manifest)
    select(path, config)


@pytest.mark.parametrize('branch', ['char', 'word'])
@pytest.mark.parametrize('missing', ['absent', 'null'])
def test_gxi_requires_class_counts_declaration(ratings, config, branch, missing):
    path, manifest = ratings
    if missing == 'absent':
        del manifest['branches'][branch]['class_counts']
    else:
        manifest['branches'][branch]['class_counts'] = None
    _write_manifest(path, manifest)
    with pytest.raises(SelectionError):
        select(path, config)


def test_gxi_rejects_different_per_class_counts_even_with_same_total(ratings, config):
    path, manifest = ratings
    # Omit checksums to isolate the cross-branch structural invariant.
    del manifest['files']
    np.array([1, 3], dtype=np.int64).tofile(path / manifest['branches']['word']['class_counts'])
    _write_manifest(path, manifest)
    with pytest.raises(SelectionError):
        select(path, config)


def test_gxi_rejects_total_inconsistent_with_attribution_rows(ratings, config):
    path, manifest = ratings
    manifest['attribution']['n_attributed_rows'] += 1
    _write_manifest(path, manifest)
    with pytest.raises(SelectionError):
        select(path, config)
