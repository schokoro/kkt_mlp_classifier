"""Independent Gate 0 checks on small, in-memory labelled data."""
import itertools

import pandas as pd
import pytest

from kkt_clf.freeze_split import FreezeSplitError
from kkt_clf.s06_split import s06_split


LEAVES = ("train_fit", "train_monitor", "comparison_val", "sealed_test")


def rows(label, count):
    return [dict(name=f"item {label} number {i}", L1="root", L2="branch", L3=label)
            for i in range(count)]


def test_threshold_precedes_split_and_every_retained_class_is_complete():
    df = pd.DataFrame(rows("edge", 12) + rows("large", 25) + rows("rare", 11))
    result = s06_split(df, seed=42)
    assert set(result.excluded.row_id) == set(range(37, 48))
    assert set(result.excluded.L3) == {"rare"}
    leaves = result.parts
    assert set(leaves["outer_train"].row_id) == (
        set(leaves["train_fit"].row_id) | set(leaves["train_monitor"].row_id)
    )
    for part in LEAVES:
        counts = leaves[part].groupby("L3").size().to_dict()
        assert set(counts) == {"edge", "large"}
        assert all(n >= (1 if part == "train_fit" else 2) for n in counts.values())
    for part in LEAVES:
        for row in leaves[part].itertuples():
            assert row.L3 == df.iloc[row.row_id].L3
            assert row.name == df.iloc[row.row_id]["name"]
    all_ids = [set(leaves[p].row_id) for p in LEAVES]
    assert all(not a & b for a, b in itertools.combinations(all_ids, 2))
    assert set.union(*all_ids) == set(range(37))
    assert len(leaves["train_fit"].query("L3 == 'edge'")) >= 6


def test_seed_reproducible_and_equivalent_names_do_not_cross_parts():
    data = rows("alpha", 20) + rows("beta", 20)
    data[0]["name"], data[1]["name"] = "Alpha Product 0", "ALPHA  PRODUCT 0"
    data[2]["name"], data[3]["name"] = "Alpha Product 1", "alpha product1"
    df = pd.DataFrame(data)
    first, second = s06_split(df, seed=42), s06_split(df.copy(), seed=42)
    for part in LEAVES:
        assert set(first.parts[part].row_id) == set(second.parts[part].row_id)
    char_keys, word_keys = [], []
    for part in LEAVES:
        names = first.parts[part]["name"]
        char_keys.append({"".join(name.lower().split()) for name in names})
        word_keys.append({tuple(name.lower().split()) for name in names})
    for keys in (char_keys, word_keys):
        assert all(not a & b for a, b in itertools.combinations(keys, 2))
    for a, b in ((0, 1), (2, 3)):
        assert any({a, b} <= set(first.parts[p].row_id) for p in LEAVES)


def test_impossible_group_constraints_block_instead_of_dropping_class():
    data = rows("good", 12) + rows("blocked", 12)
    for row in data[12:]:
        row["name"] = "same blocked name"
    with pytest.raises(FreezeSplitError) as exc:
        s06_split(pd.DataFrame(data), seed=42)
    assert "blocked" in str(exc.value)
