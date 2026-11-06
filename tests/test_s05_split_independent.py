import hashlib
import json
from itertools import combinations

import pandas as pd
import pytest

from kkt_clf.freeze_split import FreezeSplitError, freeze_split, save_frozen_split


@pytest.fixture
def rows():
    data = []
    for label in ('a', 'b'):
        for i in range(40):
            name = f'{label} item {i}'
            data.append((name, 'root', f'parent-{label}', label))
            if i == 0:
                data.extend([(name, 'root', f'parent-{label}', label),
                             (f' {label.upper()}  item\t{i} ', 'root', f'parent-{label}', label),
                             (f'{label}i tem {i}', 'root', f'parent-{label}', label)])
    data.extend((f'rare {i}', 'root', 'parent-r', 'rare') for i in range(3))
    return pd.DataFrame(data, columns=['name', 'L1', 'L2', 'L3'])


def split(rows):
    return freeze_split(rows, min_count=10, seed=42, share_tolerance=0.1)


def test_determinism_identity_isolation_and_support(rows):
    first, second = split(rows), split(rows)
    for part in first.parts:
        pd.testing.assert_frame_equal(first.parts[part], second.parts[part])
    outer = [first.parts[p] for p in ('outer_train', 'comparison_val', 'sealed_test')]
    ids = [set(p.row_id) for p in outer]
    assert all(a.isdisjoint(b) for a, b in combinations(ids, 2))
    assert set.union(*ids) | set(first.excluded.row_id) == set(range(len(rows)))
    assert set.union(*ids).isdisjoint(first.excluded.row_id)
    for part in outer:
        for row in part.itertuples():
            assert (row.name, row.L1, row.L2, row.L3) == tuple(rows.iloc[row.row_id])
        assert set(part.L3) == {'a', 'b'}
    fit, monitor = first.parts['train_fit'], first.parts['train_monitor']
    assert set(fit.row_id).isdisjoint(monitor.row_id)
    assert set(fit.row_id) | set(monitor.row_id) == ids[0]
    assert set(fit.L3) == set(monitor.L3) == {'a', 'b'}
    for a, b in [*combinations(outer, 2), (fit, monitor)]:
        for normalize in (lambda s: ''.join(s.lower().split()),
                          lambda s: ' '.join(s.lower().split())):
            assert {normalize(s) for s in a.name}.isdisjoint(normalize(s) for s in b.name)
    assert set(first.excluded.L3) == {'rare'}
    assert set(first.excluded.row_id) == set(range(len(rows) - 3, len(rows)))


def test_conflicting_normalized_names_rejected(rows):
    rows.loc[1, 'name'] = ' A  ITEM\t0 '
    rows.loc[1, ['L2', 'L3']] = ['parent-b', 'b']
    with pytest.raises(FreezeSplitError, match='конфликт'):
        split(rows)


def test_impossible_group_stratification_rejected():
    rows = pd.DataFrame([('same', 'root', 'parent', 'class')] * 20,
                        columns=['name', 'L1', 'L2', 'L3'])
    with pytest.raises(FreezeSplitError, match='стратификац'):
        split(rows)


def test_manifest_hashes_support_and_no_overwrite(rows, tmp_path):
    source = tmp_path / 'input.parquet'
    rows.to_parquet(source, index=False)
    out, audit = tmp_path / 'splits', tmp_path / 'audit'
    frozen = split(rows)
    args = (frozen, str(source), str(out), str(audit), 10, (0.8, 0.1, 0.1), 0.1, 42, 0.1)
    save_frozen_split(*args)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['policy']['split_seed'] == 42
    assert manifest['policy']['min_count'] == 10
    assert manifest['excluded_min_count']['row_ids'] == list(range(len(rows) - 3, len(rows)))
    assert manifest['inputs']['labeled_parquet']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    for part, meta in manifest['parts'].items():
        path = out / f'{part}.parquet'
        assert meta['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert meta['bytes'] == path.stat().st_size
        saved = pd.read_parquet(path)
        assert meta['n_rows'] == len(saved) == len(frozen.parts[part])
        assert set(saved.row_id) == set(frozen.parts[part].row_id)
    support = pd.read_csv(audit / 'class_support_split.csv').set_index('L3')
    for label in ('a', 'b'):
        for part in frozen.parts:
            assert support.loc[label, f'support_{part}'] == (frozen.parts[part].L3 == label).sum()
    assert not support.loc['rare', 'kept_after_min_count']
    before = {p: p.read_bytes() for directory in (out, audit) for p in directory.iterdir()}
    with pytest.raises(FreezeSplitError, match='перезапис'):
        save_frozen_split(*args)
    assert before == {p: p.read_bytes() for directory in (out, audit) for p in directory.iterdir()}


def test_split_audit_reports_number_of_unique_row_ids(rows, tmp_path):
    source = tmp_path / 'input.parquet'
    rows.to_parquet(source, index=False)
    frozen = split(rows)
    save_frozen_split(frozen, str(source), str(tmp_path / 'splits'), str(tmp_path / 'audit'),
                      10, (0.8, 0.1, 0.1), 0.1, 42, 0.1)
    audit = pd.read_csv(tmp_path / 'audit' / 'split_audit.csv').set_index('part')
    for part, frame in frozen.parts.items():
        assert audit.loc[part, 'n_row_ids_unique'] == frame.row_id.nunique()
    assert audit.loc['excluded_by_min_count', 'n_row_ids_unique'] == frozen.excluded.row_id.nunique()
