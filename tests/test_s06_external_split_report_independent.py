"""Regression coverage for reporting against a frozen external S06 split."""
import hashlib
import json
import subprocess

import pandas as pd
import pytest

from kkt_clf import s06_report as report


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _json_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


@pytest.fixture
def external(tmp_path, monkeypatch):
    def no_subprocess(*args, **kwargs):
        raise OSError("subprocess disabled")
    monkeypatch.setattr(subprocess, "run", no_subprocess)
    root = tmp_path / "experiment"
    split = tmp_path / "frozen" / "manifest.json"
    val = split.parent / "comparison_val.parquet"
    val.parent.mkdir(parents=True)
    pd.DataFrame({"row_id": [1, 2], "L3": ["a", "b"]}).to_parquet(val, index=False)
    maps = {"L1": ["top"], "L2": ["middle"], "L3": ["a", "b"]}
    part = {"file": "comparison_val.parquet", "sha256": _sha(val),
            "bytes": val.stat().st_size, "n_rows": 2}
    manifest = {"parts": {"comparison_val": part}, "label_maps": maps,
                "inputs": {"labeled_parquet": {"sha256": "source-hash"}}}
    _json(split, manifest)
    protocol = root / "shared" / "training_protocol.json"
    _json(protocol, {"splits": {"comparison_val": {k: part[k] for k in ("sha256", "bytes", "n_rows")}},
                     "labeled_parquet": {"sha256": "source-hash"}})
    run = {"run_id": "none_seed42", "config_slug": "none", "method": "none",
           "budget": None, "seed": 42, "status": "pending", "reason": None}
    plan = {"runs": [run], "n_planned": 1, "protocol_sha256": _sha(protocol),
            "config": {"splits_manifest": str(split), "seeds": [42], "reference_seed": 42,
                       "methods": [], "budgets": [], "char_share": .5}}
    _json(root / "runs" / "planned.json", plan)
    _json(root / "runs" / "baseline_full" / "manifest.json", {
        "label_maps": {"l3_universe_sha256": _json_sha(maps),
                       "n_classes": {"L1": 1, "L2": 1, "L3": 2}},
        "inputs": {"splits": {"comparison_val": {k: part[k] for k in ("sha256", "bytes", "n_rows")}}}})
    pred = root / "runs" / "predictions" / "none_seed42.parquet"
    pred.parent.mkdir(parents=True)
    pd.DataFrame({"row_id": [1, 2], "run_id": ["none_seed42"] * 2,
                  "true_L3": ["a", "b"], "predicted_L3": ["a", "a"],
                  "correct": [True, False]}).to_parquet(pred, index=False)
    _json(root / "runs" / "none" / "run_result_seed_42.json", {
        **run, "schema_version": report.RUN_RESULT_SCHEMA, "status": "success",
        "hashes": {"model_pt_sha256": "model", "tfidf_features_sha256": "features",
                   "predictions_sha256": _sha(pred)},
        "artifacts": {"checkpoint_dir": "checkpoint", "tfidf_artifact": "tfidf"},
        "comparison_val_eval": {"predictions_path": str(pred), "l3_macro_f1": 1 / 3,
                                "l3_accuracy": .5}})
    return root, split, val, protocol


def test_report_uses_external_split_and_records_its_provenance(external):
    root, split, val, _ = external
    info = report.build_s06_report(str(root))
    assert info["statuses"]["success"] == 1
    recorded = json.loads((root / "report" / "manifest.json").read_text())
    assert recorded["inputs"]["splits_manifest"]["sha256"] == _sha(split)
    assert pd.read_parquet(root / "report" / "predictions_val.parquet")["row_id"].tolist() == [1, 2]


@pytest.mark.parametrize("damage", ["manifest_missing", "split_fingerprint", "label_maps",
                                    "protocol_missing", "protocol_changed", "protocol_split_missing",
                                    "split_rows", "split_bytes", "val_missing", "val_changed"])
def test_external_split_tampering_fails_closed(external, damage):
    root, split, val, protocol = external
    # A tempting alternate manifest must never substitute for the plan's path.
    alternate = root / "splits" / "manifest.json"
    _json(alternate, json.loads(split.read_text()))
    if damage == "manifest_missing":
        split.rename(split.with_suffix(".unavailable"))
    elif damage in ("split_fingerprint", "split_rows", "split_bytes", "label_maps"):
        data = json.loads(split.read_text())
        if damage == "split_fingerprint":
            data["parts"]["comparison_val"]["sha256"] = "0" * 64
        elif damage == "split_rows":
            data["parts"]["comparison_val"]["n_rows"] = 3
        elif damage == "split_bytes":
            data["parts"]["comparison_val"]["bytes"] += 1
        else:
            data["label_maps"]["L3"].append("extra")
        _json(split, data)
    elif damage == "protocol_missing":
        protocol.rename(protocol.with_suffix(".unavailable"))
    elif damage == "protocol_changed":
        _json(protocol, {"splits": {}})
    elif damage == "protocol_split_missing":
        data = json.loads(protocol.read_text())
        data["splits"] = {}
        _json(protocol, data)
        plan_path = root / "runs" / "planned.json"
        plan = json.loads(plan_path.read_text())
        plan["protocol_sha256"] = _sha(protocol)
        _json(plan_path, plan)
    elif damage == "val_missing":
        val.rename(val.with_suffix(".unavailable"))
    else:
        pd.DataFrame({"row_id": [1, 3], "L3": ["a", "b"]}).to_parquet(val, index=False)
    with pytest.raises(report.S06ReportError):
        report.build_s06_report(str(root))
    assert not (root / "report" / "report.html").exists()


def test_saved_tables_rerender_without_sources_or_inference(external, monkeypatch):
    root, split, _, _ = external
    report.build_s06_report(str(root))
    # Rename rather than delete the sources, leaving only the saved report available.
    (root / "runs").rename(root / "unavailable_runs")
    (root / "shared").rename(root / "unavailable_shared")
    split.parent.rename(split.parent.with_name("unavailable_frozen"))
    from kkt_clf import inference, s06_comparison
    def forbidden(*args, **kwargs):
        raise AssertionError("model inference or training called during table-only render")
    monkeypatch.setattr(inference, "run_inference", forbidden)
    monkeypatch.setattr(s06_comparison, "run_train", forbidden)
    monkeypatch.setattr(s06_comparison, "run_refit_selected", forbidden)
    result = report.build_s06_report(str(root), str(root / "rerender"))
    assert result["mode"] == "html_rebuild_from_saved_tables"
    assert (root / "rerender" / "report.html").is_file()
