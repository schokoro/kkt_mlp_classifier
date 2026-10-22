"""Предсказание, отказ и метрики по размеченному входу — общий код для обучения и инференса (§4.6, §4.9)."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .artifact import Artifact, load_artifact
from .config import InferenceConfig
from .data import read_parquet
from .features import empty_rows
from .labels import LEVELS, LabelMaps
from .metrics import NO_PREDICTION, evaluate, f1_vs_frequency, save_eval
from .model import SparseMLP
from .reject import RejectRule, apply_reject, mismatch_flags
from .train import predict, resolve_device

log = logging.getLogger(__name__)


def output_columns(top_k: int) -> list[str]:
    cols = [
        "l3_pred", "l3_prob", "rejected", "reject_reason", "l3_top1_raw",
        "l2_from_l3", "l1_from_l3", "l2_head", "l2_head_prob", "l1_head", "l1_head_prob",
        "mismatch_l2", "mismatch_l1",
    ]
    for i in range(1, top_k + 1):
        cols += [f"l3_top{i}", f"l3_top{i}_prob"]
    return cols


def predict_frame(
    model: SparseMLP,
    X: sp.csr_matrix,
    maps: LabelMaps,
    rule: RejectRule,
    device: torch.device,
    batch_size: int = 8192,
    top_k: int = 0,
) -> pd.DataFrame:
    """Модель + отказ → таблица выходных колонок §4.9.

    Для строк с пустым вектором признаков модель не вызывается: их выходы
    модели — null, отказ ``empty_features``.
    """
    n = X.shape[0]
    empty = empty_rows(X)
    ok = np.flatnonzero(~empty)
    k = min(top_k, maps.n_classes("L3"))
    preds, _ = predict(model, X[ok], device, batch_size, max(k, 1))

    def full(values, fill, dtype):
        out = np.full(n, fill, dtype=dtype)
        out[ok] = values
        return out

    l3_ids = full(preds.l3_top1, -1, np.int64)
    l2_ids = full(preds.l2_head, -1, np.int64)
    l1_ids = full(preds.l1_head, -1, np.int64)
    l3_prob = full(preds.l3_prob, np.nan, np.float32)
    mm_l2 = np.zeros(n, dtype=bool)
    mm_l1 = np.zeros(n, dtype=bool)
    if len(ok):
        mm_l2[ok], mm_l1[ok] = mismatch_flags(preds.l3_top1, preds.l2_head, preds.l1_head, maps)
    rejected, reason = apply_reject(l3_prob, empty, mm_l2, mm_l1, rule)
    safe3 = np.where(l3_ids < 0, 0, l3_ids)
    l3_raw = maps.decode("L3", l3_ids)
    l3_pred = l3_raw.copy()
    l3_pred[rejected] = None
    out = {
        "l3_pred": l3_pred,
        "l3_prob": l3_prob,
        "rejected": rejected,
        "reject_reason": reason,
        "l3_top1_raw": l3_raw,
        "l2_from_l3": maps.decode("L2", np.where(l3_ids < 0, -1, maps.l2_of_l3(safe3))),
        "l1_from_l3": maps.decode("L1", np.where(l3_ids < 0, -1, maps.l1_of_l3(safe3))),
        "l2_head": maps.decode("L2", l2_ids),
        "l2_head_prob": full(preds.l2_prob, np.nan, np.float32),
        "l1_head": maps.decode("L1", l1_ids),
        "l1_head_prob": full(preds.l1_prob, np.nan, np.float32),
        "mismatch_l2": mm_l2,
        "mismatch_l1": mm_l1,
    }
    for i in range(k):
        out[f"l3_top{i + 1}"] = maps.decode("L3", full(preds.l3_topk[:, i], -1, np.int64))
        out[f"l3_top{i + 1}_prob"] = full(preds.l3_topk_prob[:, i], np.nan, np.float32)
    return pd.DataFrame(out)


def evaluate_frame(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    maps: LabelMaps,
    train_counts: dict[str, int] | None = None,
    top_n_errors: int = 50,
    noisy_support: int = 2,
) -> dict:
    """Метрики §4.7 по выходной таблице и истинным меткам L1..L3.

    Метки вне маппинга модели исключаются из метрик и выводятся отдельной
    строкой отчёта. Строки без предсказания (empty_features) в варианте «без
    отказа» считаются ошибкой.
    """
    true_labels = {lvl: labels[lvl].astype(object).to_numpy() for lvl in LEVELS}
    known = {lvl: maps.encode(lvl, true_labels[lvl]) >= 0 for lvl in LEVELS}

    def col(name):
        v = frame[name].to_numpy(dtype=object).copy()
        v[pd.isna(v)] = NO_PREDICTION
        return v

    pred = {
        "l1_head": col("l1_head"), "l2_head": col("l2_head"),
        "l1_from_l3": col("l1_from_l3"), "l2_from_l3": col("l2_from_l3"),
        "l3": col("l3_top1_raw"),
    }
    accepted = ~frame["rejected"].to_numpy(dtype=bool)
    has_pred = frame["reject_reason"].to_numpy(dtype=object) != "empty_features"
    ev = evaluate(
        true_labels, known, pred, accepted,
        frame["mismatch_l2"].to_numpy(dtype=bool), frame["mismatch_l1"].to_numpy(dtype=bool),
        top_n_errors, noisy_support, has_pred=has_pred,
    )
    if train_counts is not None:
        ev["f1_vs_freq"] = f1_vs_frequency(ev["reports"]["no_reject__L3__head"], train_counts)
    ev["n_empty_features"] = int((~has_pred).sum())
    ev["n_unknown_labels"] = {lvl: int((~known[lvl]).sum()) for lvl in LEVELS}
    return ev


class Predictor:
    """Восстановленный пайплайн: предобработка → векторизация → модель → отказ."""

    def __init__(self, artifact: Artifact, device: torch.device):
        self.artifact = artifact
        self.device = device
        self.model = artifact.model.to(device).eval()
        self.features = artifact.features
        self.maps = artifact.maps
        self.rule = artifact.rule

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "Predictor":
        dev = resolve_device(device)
        return cls(load_artifact(path, dev), dev)

    def make_rule(self, tau=None, reject_on_mismatch_l2=None, reject_on_mismatch_l1=None) -> RejectRule:
        r = self.rule
        return RejectRule(
            r.tau if tau is None else float(tau),
            r.reject_on_mismatch_l2 if reject_on_mismatch_l2 is None else bool(reject_on_mismatch_l2),
            r.reject_on_mismatch_l1 if reject_on_mismatch_l1 is None else bool(reject_on_mismatch_l1),
        )

    def predict(
        self,
        names: Sequence,
        batch_size: int = 50_000,
        top_k: int = 0,
        rule: RejectRule | None = None,
        model_batch_size: int = 8192,
    ) -> pd.DataFrame:
        """Батчевая векторизация и прогон; вся признаковая матрица не материализуется."""
        rule = rule or self.rule
        names = list(names)
        parts = []
        for start in range(0, len(names), batch_size):
            X = self.features.transform(names[start : start + batch_size], chunk_rows=batch_size)
            parts.append(predict_frame(self.model, X, self.maps, rule, self.device, model_batch_size, top_k))
        if not parts:
            return pd.DataFrame(columns=output_columns(min(top_k, self.maps.n_classes("L3"))))
        return pd.concat(parts, ignore_index=True)

    @property
    def train_counts(self) -> dict[str, int]:
        return self.artifact.class_weights.get("train_counts", {}).get("L3", {})


@dataclass
class InferenceResult:
    output_path: str
    predictions: pd.DataFrame
    rule: RejectRule
    has_labels: bool
    evaluation: dict | None = None
    report_paths: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def run_inference(cfg: InferenceConfig) -> InferenceResult:
    """Ноутбук инференса одним вызовом: чтение → предсказание → parquet → метрики (если есть метки)."""
    t0 = time.perf_counter()
    predictor = Predictor.load(cfg.artifact_dir, cfg.device)
    rule = predictor.make_rule(cfg.tau, cfg.reject_on_mismatch_l2, cfg.reject_on_mismatch_l1)
    df = read_parquet(cfg.input_path)
    if "name" not in df.columns:
        raise ValueError(f"Во входе {cfg.input_path} нет колонки 'name'")
    k = min(cfg.top_k, predictor.maps.n_classes("L3"))
    clash = [c for c in output_columns(k) if c in df.columns]
    if clash:
        raise ValueError(f"Во входе уже есть выходные колонки: {clash}")
    t_read = time.perf_counter()
    pred = predictor.predict(df["name"].tolist(), cfg.batch_size, cfg.top_k, rule)
    t_pred = time.perf_counter()
    out = pd.concat([df.reset_index(drop=True), pred], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)), exist_ok=True)
    out.to_parquet(cfg.output_path, index=False)
    has_labels = all(lvl in df.columns for lvl in LEVELS)
    res = InferenceResult(cfg.output_path, out, rule, has_labels)
    if not has_labels and any(lvl in df.columns for lvl in LEVELS):
        log.warning("Во входе есть не все колонки L1..L3 — метрики не считаются")
    if has_labels:
        res.evaluation = evaluate_frame(pred, df[list(LEVELS)], predictor.maps, predictor.train_counts)
        if cfg.report_dir:
            res.report_paths = save_eval(res.evaluation, cfg.report_dir)
    res.timings = {
        "load_and_read_s": t_read - t0,
        "predict_s": t_pred - t_read,
        "rows": int(len(df)),
        "rows_per_s": len(df) / max(t_pred - t_read, 1e-9),
    }
    log.info("Инференс: %d строк, %.1f с, отказов %.1f%%", len(df), t_pred - t_read,
             100 * float(pred["rejected"].mean()) if len(pred) else 0.0)
    return res
