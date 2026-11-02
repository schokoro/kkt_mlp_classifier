import os

import numpy as np
import pandas as pd
import pytest

from kkt_clf import InferenceConfig, Predictor, run_inference, run_training, verify_roundtrip
from kkt_clf.artifact import FILES
from kkt_clf.inference import output_columns
from kkt_clf.metrics import UNKNOWN_ROW

from .conftest import small_config


def test_training_outputs(trained):
    r = trained
    assert all(os.path.exists(os.path.join(r.artifact_dir, f)) for f in FILES)
    for f in ("history.csv", "tau_table_val.csv", "param_counts.csv", "test__summary.csv",
              "test__top_errors.csv", "test__mismatch_rates.csv", "test__f1_vs_freq.csv",
              "test__report__no_reject__L3__head.csv", "test__report__with_reject__L3__head.csv",
              "test__report__no_reject__L2__from_l3.csv", "rare_l3_classes.csv", "run_report.json"):
        assert os.path.exists(os.path.join(r.report_dir, f)), f
    assert os.path.exists(os.path.join(r.output_dir, "splits", "test.parquet"))
    rep = pd.read_csv(os.path.join(r.report_dir, "test__report__no_reject__L3__head.csv"), index_col=0)
    assert "support" in rep.columns
    assert set(r.summary["variant"]) == {"no_reject", "with_reject"}
    assert 0 < r.evaluation["coverage"] <= 1
    assert len(r.history) >= 1
    # scheduler по умолчанию — plateau с проверкой каждые N шагов
    assert os.path.exists(os.path.join(r.report_dir, "lr_trace.csv"))
    tr = r.lr_trace
    assert len(tr) > len(r.history) and tr["val_loss_subset"].notna().all()
    assert (tr["lr"].diff().dropna() <= 0).all()
    assert "lr" in r.history.columns
    # модель учится: заметно лучше случайного угадывания на 40 классах
    acc = r.summary.query("variant=='no_reject' and level=='L3'")["accuracy"].iloc[0]
    assert acc > 0.3


def test_roundtrip_predictions_equal(trained):
    assert verify_roundtrip(trained, n_rows=10_000)


def test_reproducible(fx, tmp_path, trained):
    r2 = run_training(small_config(fx, tmp_path))
    pd.testing.assert_frame_equal(trained.data.splits["test"], r2.data.splits["test"])
    pd.testing.assert_frame_equal(trained.summary, r2.summary)


@pytest.mark.parametrize("hidden,dropout,norm,scheduler", [
    ([96, 48], [0.2, 0.1], "layernorm", "onecycle"),
    ([32, 32, 16], [0.1, 0.1, 0.0], "batchnorm", "cosine"),
    ([64], [0.1], "batchnorm", "plateau"),
])
def test_architecture_change_via_config(fx, tmp_path, hidden, dropout, norm, scheduler):
    cfg = small_config(fx, tmp_path, hidden=hidden, dropout=dropout, norm=norm)
    cfg.train.scheduler = scheduler
    r = run_training(cfg)
    assert len(r.lr_trace) > 0
    linear = r.param_counts[r.param_counts["type"] == "Linear"]
    assert len(linear) == len(hidden) - 1 + 3
    p = Predictor.load(r.artifact_dir, "cpu")
    assert p.model.input.out_features == hidden[0]
    verify_roundtrip(r, n_rows=200)


def test_inference_without_labels(fx, trained, tmp_path):
    out = tmp_path / "pred.parquet"
    res = run_inference(InferenceConfig(artifact_dir=trained.artifact_dir, input_path=fx["infer_no_labels"],
                                        output_path=str(out), report_dir=None, top_k=3, device="cpu", batch_size=64))
    df = pd.read_parquet(out)
    inp = pd.read_parquet(fx["infer_no_labels"])
    assert list(df.columns) == list(inp.columns) + output_columns(3)
    pd.testing.assert_frame_equal(df[inp.columns], inp)
    assert res.evaluation is None and not res.has_labels
    empty = df["name"].isna() | df["name"].fillna("").str.strip().eq("") | df["name"].fillna("").str.contains("ꙮ|⁣")
    assert (df.loc[empty, "reject_reason"] == "empty_features").all()
    assert df.loc[empty, "l3_top1_raw"].isna().all()
    assert df.loc[df["rejected"], "l3_pred"].isna().all()
    ok = ~df["rejected"]
    assert (df.loc[ok, "l3_pred"] == df.loc[ok, "l3_top1_raw"]).all()
    assert (df.loc[ok, "l3_prob"] >= res.rule.tau).all()
    assert set(df["reject_reason"].dropna()) <= {"empty_features", "low_confidence", "mismatch"}
    assert df["mismatch_l2"].dtype == bool


def test_inference_unknown_labels(fx, trained, tmp_path):
    res = run_inference(InferenceConfig(artifact_dir=trained.artifact_dir, input_path=fx["infer_unknown_labels"],
                                        output_path=str(tmp_path / "p.parquet"), report_dir=str(tmp_path / "rep"),
                                        device="cpu"))
    ev = res.evaluation
    assert ev is not None
    rep = ev["reports"]["no_reject__L3__head"]
    assert UNKNOWN_ROW in rep.index
    assert ev["n_unknown_labels"]["L3"] >= 18
    assert ev["n_empty_features"] >= 7
    assert os.path.exists(tmp_path / "rep" / "summary.csv")


def test_inference_overrides(fx, trained, tmp_path):
    common = dict(artifact_dir=trained.artifact_dir, input_path=fx["infer_no_labels"], report_dir=None, device="cpu")
    r0 = run_inference(InferenceConfig(output_path=str(tmp_path / "a.parquet"), tau=0.0, **common))
    r1 = run_inference(InferenceConfig(output_path=str(tmp_path / "b.parquet"), tau=1.0, **common))
    assert (r0.predictions["reject_reason"].dropna() == "empty_features").all()
    assert r1.predictions["rejected"].all()
    r2 = run_inference(InferenceConfig(output_path=str(tmp_path / "c.parquet"), tau=0.0,
                                       reject_on_mismatch_l2=True, **common))
    p = r2.predictions
    assert ((p["reject_reason"] == "mismatch") == (p["mismatch_l2"] & (p["reject_reason"] != "empty_features"))).all()
    np.testing.assert_array_equal(r0.predictions["l3_top1_raw"].to_numpy(dtype=object),
                                  r1.predictions["l3_top1_raw"].to_numpy(dtype=object))


def test_inference_on_saved_test_reproduces_metrics(trained, tmp_path):
    """User story 6: метрики инференса на сохранённом test = метрики обучения."""
    res = run_inference(InferenceConfig(artifact_dir=trained.artifact_dir,
                                        input_path=os.path.join(trained.output_dir, "splits", "test.parquet"),
                                        output_path=str(tmp_path / "t.parquet"), report_dir=None, device="cpu"))
    pd.testing.assert_frame_equal(res.evaluation["summary"], trained.summary)
