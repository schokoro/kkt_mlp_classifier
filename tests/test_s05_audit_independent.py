"""Contract checks for the S05 audit using only disposable synthetic inputs."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_ods.py"


@pytest.fixture
def audit_case(tmp_path):
    rows = [
        ("Молоко  цельное", "Food", "Dairy", "Milk"),
        ("молоко\tцельное", "Food", "Dairy", "Milk"),
        ("МОЛОКОЦЕЛЬНОЕ", "Food", "Dairy", "Cheese"),
        ("same", "Food", "Dairy", "Cheese"),
        ("same", "Food", "Dairy", "Cheese"),
        ("other", "Other", "Dairy", "Rare"),
        ("third", "Food", "Bakery", "Milk"),
        (None, "Food", "Dairy", "Milk"),
        (" ", None, "Dairy", "Milk"),
        ("invalid", "Food", "  ", "Milk"),
        ("invalid2", "Food", "Dairy", ""),
    ]
    source = tmp_path / "input.parquet"
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame(rows, columns=["name", "L1", "L2", "L3"]).to_parquet(source)
    pd.DataFrame({"name": ["молоко цельное", "unrelated"]}).to_parquet(corpus)
    out = tmp_path / "audit"
    command = [sys.executable, str(SCRIPT), "--input", str(source), "--corpus", str(corpus),
               "--s02-dir", str(tmp_path / "missing_s02"), "--legacy-csv", str(tmp_path / "missing.csv"),
               "--out-dir", str(out), "--min-count", "3"]
    return source, corpus, out, command


def run(command):
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_audit_reports_actual_data_and_preserves_inputs(audit_case):
    source, corpus, out, command = audit_case
    before = (source.read_bytes(), corpus.read_bytes())
    result = run(command)
    assert result.returncode == 0, result.stderr
    assert (source.read_bytes(), corpus.read_bytes()) == before
    audit = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))["audit"]
    assert audit["rows_read"] == 11
    assert audit["rows_after_drop_empty"] == 7
    assert audit["empty_values"] == {"name": 2, "L1": 1, "L2": 1, "L3": 1, "total": 4}

    violations = {(v["level"], v["child"], tuple(v["parents"]))
                  for v in audit["taxonomy_violations"]["examples"]}
    assert violations == {("L3", "Milk", ("Bakery", "Dairy")),
                          ("L2", "Dairy", ("Food", "Other"))}
    assert audit["taxonomy_violations"]["n"] == 2
    duplicates = audit["duplicates"]
    assert duplicates["raw_name"]["n_duplicate_extra_rows"] == 1
    assert duplicates["raw_name"]["n_names_appearing_more_than_once"] == 1
    keys = duplicates["normalized_keys"]
    assert keys["char_branch_strip_whitespace"]["n_groups_multi_l3"] == 1
    assert keys["char_branch_strip_whitespace"]["n_rows_in_multi_l3_groups"] == 3
    assert keys["word_branch_collapse_ws"]["n_groups_multi_l3"] == 0
    assert keys["word_branch_collapse_ws"]["n_collided_groups"] == 2

    rare = audit["min_count_filter"]
    assert (rare["n_excluded_l3"], rare["n_excluded_rows"], rare["rows_after"], rare["n_l3_after"]) == (1, 1, 6, 2)
    assert rare["excluded_l3_full_list"] == [{"L1": "Other", "L2": "Dairy", "L3": "Rare", "count": 1}]
    support = pd.read_csv(out / "class_support.csv").set_index(["L1", "L2", "L3"])
    expected = {("Food", "Dairy", "Milk"): (2, 2), ("Food", "Bakery", "Milk"): (1, 1),
                ("Food", "Dairy", "Cheese"): (3, 3), ("Other", "Dairy", "Rare"): (1, 0)}
    assert set(support.index) == set(expected)
    for key, (raw, after) in expected.items():
        row = support.loc[key]
        assert row["count_raw"] == raw
        assert row["count_after_min_count"] == after
        assert bool(row["kept_after_min_count"]) == (after > 0)
        assert row["share_raw"] == pytest.approx(raw / 7, abs=1e-6)


@pytest.mark.parametrize("existing", ["data_audit.json", "class_support.csv"])
def test_existing_output_refuses_overwrite_without_writing_other_file(audit_case, existing):
    source, corpus, out, command = audit_case
    out.mkdir()
    sentinel = b"existing user data\n"
    (out / existing).write_bytes(sentinel)
    before = (source.read_bytes(), corpus.read_bytes())
    result = run(command)
    assert result.returncode != 0
    assert (out / existing).read_bytes() == sentinel
    assert sorted(p.name for p in out.iterdir()) == [existing]
    assert (source.read_bytes(), corpus.read_bytes()) == before
