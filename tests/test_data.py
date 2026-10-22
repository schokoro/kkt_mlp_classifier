import numpy as np
import pandas as pd
import pytest

from kkt_clf.config import DataConfig
from kkt_clf.data import (
    DataValidationError,
    DuplicateWarning,
    TaxonomyError,
    check_columns,
    check_duplicates,
    check_taxonomy,
    drop_empty,
    filter_min_count,
    prepare_data,
    split,
    stratified_sample,
)


def test_taxonomy_ok(fx):
    check_taxonomy(pd.read_parquet(fx["ods"]))


def test_taxonomy_violation_stops_with_list(fx):
    df = pd.read_parquet(fx["defect_taxonomy"])
    with pytest.raises(TaxonomyError) as e:
        check_taxonomy(df)
    v = e.value.violations
    assert set(v["level"]) == {"L3", "L2"}
    assert all(len(p) == 2 for p in v["parents"])
    assert "Нарушена уникальность" in str(e.value)


def test_prepare_data_stops_on_taxonomy(fx):
    with pytest.raises(TaxonomyError):
        prepare_data(DataConfig(path=fx["defect_taxonomy"]), seed=0)


def test_missing_columns():
    with pytest.raises(DataValidationError):
        check_columns(pd.DataFrame({"name": ["a"], "L1": ["x"]}))


def test_drop_empty_counts(fx):
    df = pd.read_parquet(fx["defect_dups_empty"])
    clean, counts = drop_empty(df)
    assert counts["total"] == 5
    assert counts["name"] == 3 and counts["L2"] == 1 and counts["L3"] == 1
    assert len(clean) == len(df) - 5


def test_duplicates_warn_without_changes(fx):
    df, _ = drop_empty(pd.read_parquet(fx["defect_dups_empty"]))
    before = df.copy()
    with pytest.warns(DuplicateWarning):
        rep = check_duplicates(df)
    pd.testing.assert_frame_equal(df, before)
    assert rep.n_duplicate_rows == 8
    assert rep.n_conflicting_names == 3
    assert len(rep.examples) > 0 and {"name", "n_rows", "n_label_paths"} <= set(rep.examples.columns)


def test_no_duplicates_no_warning(fx):
    import warnings

    df = pd.read_parquet(fx["ods_small"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DuplicateWarning)
        assert check_duplicates(df).n_duplicate_rows == 0


def test_stratified_sample(fx):
    df = pd.read_parquet(fx["ods"])
    s = stratified_sample(df, 5000, seed=1)
    assert len(s) == 5000
    full = df["L3"].value_counts(normalize=True)
    part = s["L3"].value_counts(normalize=True).reindex(full.index, fill_value=0)
    assert np.abs(full - part).max() < 1.0 / 5000 + 1e-9
    pd.testing.assert_frame_equal(s, stratified_sample(df, 5000, seed=1))
    assert stratified_sample(df, None, 1) is df


def test_min_count_and_split(fx):
    df = pd.read_parquet(fx["ods"])
    kept, rare = filter_min_count(df, 10)
    assert kept["L3"].value_counts().min() >= 10
    assert (rare["count"] < 10).all()
    assert len(kept) + rare["count"].sum() == len(df)
    sp = split(kept, (0.8, 0.1, 0.1), seed=3)
    n = len(kept)
    assert abs(len(sp["train"]) / n - 0.8) < 0.01
    assert abs(len(sp["val"]) / n - 0.1) < 0.01
    names = [set(sp[k]["name"]) for k in ("train", "val", "test")]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
    classes = set(kept["L3"])
    for k in ("train", "val", "test"):
        assert set(sp[k]["L3"]) == classes, k
    sp2 = split(kept, (0.8, 0.1, 0.1), seed=3)
    pd.testing.assert_frame_equal(sp["test"], sp2["test"])


def test_prepare_data_order_min_count_after_sample(fx):
    # сэмпл до min_count: классы, опустившиеся ниже порога из-за сэмпла, отсекаются
    pdata = prepare_data(DataConfig(path=fx["ods"], max_rows=4000, min_count=10), seed=0)
    full = pd.concat(pdata.splits.values())
    assert full["L3"].value_counts().min() >= 10
    assert pdata.report["rows_after_sample"] == 4000
    assert pdata.report["rows_after_min_count"] == len(full)
    assert pdata.report["rare_l3_classes"] == len(pdata.rare_classes)
