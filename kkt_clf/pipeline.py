"""Сквозной обучающий прогон: данные → признаки → модель → τ → метрики → артефакт."""
from __future__ import annotations

import json
import logging
import os
import resource
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .artifact import save_artifact
from .config import HEADS, RunConfig, validate_config
from .data import PreparedData, file_fingerprint, prepare_data, save_splits
from .features import FeatureBuilder, matrix_stats
from .inference import Predictor, evaluate_frame, predict_frame
from .labels import LEVELS
from .metrics import save_eval
from .model import SparseMLP, param_counts
from .reject import RejectRule, select_tau, tau_table
from .train import compute_class_weights, resolve_device, set_seed, train_model

log = logging.getLogger(__name__)

LIMITATIONS = [
    "Калибровка вероятностей не делалась: softmax L3 используется как есть, τ — порог на некалиброванную уверенность.",
    "Классы L3 на границе min_count дают в val/test по 1–2 примера; их метрики шумные.",
    "Метрики на ODS — проверка работоспособности, а не целевые значения.",
]


def setup_logging(level: int = logging.INFO) -> None:
    """Вывод логов пакета в stdout (для ноутбуков и скриптов)."""
    logger = logging.getLogger("kkt_clf")
    logger.setLevel(level)
    if not any(getattr(h, "_kkt", False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        h._kkt = True
        logger.addHandler(h)
    logger.propagate = False


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class TrainResult:
    config: RunConfig
    output_dir: str
    artifact_dir: str
    report_dir: str
    data: PreparedData
    feature_stats: dict
    param_counts: pd.DataFrame
    history: pd.DataFrame
    best_epoch: dict
    tau_table: pd.DataFrame
    tau_selection: dict
    evaluation: dict
    test_predictions: pd.DataFrame
    timings: dict
    limitations: list[str] = field(default_factory=lambda: list(LIMITATIONS))

    @property
    def summary(self) -> pd.DataFrame:
        return self.evaluation["summary"]


def _y(df: pd.DataFrame, maps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(maps.encode(lvl, df[lvl].to_numpy()) for lvl in LEVELS)


def run_training(cfg: RunConfig) -> TrainResult:
    validate_config(cfg)
    t = {"start": time.perf_counter()}
    set_seed(cfg.seed)
    if cfg.train.num_threads:
        torch.set_num_threads(cfg.train.num_threads)
    device = resolve_device(cfg.train.device)
    log.info("Устройство: %s", device)
    out = cfg.output_dir
    art_dir = os.path.join(out, "artifact")
    rep_dir = os.path.join(out, "reports")
    os.makedirs(rep_dir, exist_ok=True)

    # 1. Данные (§4.1–4.3)
    data = prepare_data(cfg.data, cfg.seed)
    log.info("Данные: %s", json.dumps(data.report, ensure_ascii=False))
    save_splits(data.splits, os.path.join(out, "splits"))
    data.rare_classes.to_csv(os.path.join(rep_dir, "rare_l3_classes.csv"), index=False)
    data.duplicates.examples.to_csv(os.path.join(rep_dir, "duplicate_examples.csv"), index=False)
    t["data"] = time.perf_counter()

    # 2. Признаки (§4.4): обучение только на train
    fb = FeatureBuilder(cfg.features).fit(data.splits["train"]["name"].tolist(), seed=cfg.seed)
    t["vectorizer_fit"] = time.perf_counter()
    X = {k: fb.transform(v["name"].tolist()) for k, v in data.splits.items()}
    t["vectorize"] = time.perf_counter()
    feature_stats = {
        "n_features": fb.n_features,
        "n_char_features": fb.n_char_features,
        "n_word_features": fb.n_word_features,
        **{k: matrix_stats(v) for k, v in X.items()},
    }
    log.info("Признаки: %d (char %d, word %d); nnz/строку train %.1f; пустых: train %d, val %d, test %d",
             fb.n_features, fb.n_char_features, fb.n_word_features,
             feature_stats["train"]["nnz_per_row_mean"], feature_stats["train"]["empty_rows"],
             feature_stats["val"]["empty_rows"], feature_stats["test"]["empty_rows"])

    # 3. Модель и обучение (§4.5)
    maps = data.maps
    y = {k: _y(v, maps) for k, v in data.splits.items()}
    n_classes = tuple(maps.n_classes(lvl) for lvl in LEVELS)
    class_weights = {
        h: compute_class_weights(y["train"][i], n_classes[i], cfg.loss.class_weight_mode[h], cfg.loss.class_weight_alpha)
        for i, h in enumerate(HEADS)
    }
    set_seed(cfg.seed)
    model = SparseMLP(fb.n_features, n_classes, cfg.model)
    pc = pd.DataFrame(param_counts(model))
    log.info("Параметры модели:\n%s", pc.to_string(index=False))
    history, best = train_model(
        model, X["train"], y["train"], X["val"], y["val"], cfg.loss, cfg.train, class_weights, device, cfg.seed
    )
    t["train"] = time.perf_counter()
    hist = pd.DataFrame(history)

    # 4. Выбор τ на val (§4.6)
    rj = cfg.reject
    val_frame = predict_frame(model, X["val"], maps, RejectRule(rj.tau), device, cfg.train.eval_batch_size)
    l3_val = maps.encode("L3", val_frame["l3_top1_raw"].to_numpy(dtype=object))
    tt = tau_table(
        y["val"][2], l3_val, val_frame["l3_prob"].to_numpy(),
        (val_frame["reject_reason"] == "empty_features").to_numpy(),
        val_frame["mismatch_l2"].to_numpy(), val_frame["mismatch_l1"].to_numpy(),
        rj.tau_grid, n_classes[2], rj.reject_on_mismatch_l2, rj.reject_on_mismatch_l1,
    )
    sel = select_tau(tt, rj.tau_mode, rj.tau, rj.target_precision)
    rule = RejectRule(sel["tau"], rj.reject_on_mismatch_l2, rj.reject_on_mismatch_l1)
    log.info("τ = %.2f (%s)", rule.tau, rj.tau_mode)

    # 5. Метрики на test (§4.7)
    train_counts = {lvl: data.splits["train"][lvl].value_counts().sort_index().astype(int).to_dict() for lvl in LEVELS}
    test_frame = predict_frame(model, X["test"], maps, rule, device, cfg.train.eval_batch_size, top_k=3)
    ev = evaluate_frame(test_frame, data.splits["test"], maps, train_counts["L3"],
                        cfg.report.top_n_errors, cfg.report.noisy_support)
    t["evaluate"] = time.perf_counter()

    # 6. Артефакт (§4.8)
    cw_json = {
        "mode": cfg.loss.class_weight_mode,
        "alpha": cfg.loss.class_weight_alpha,
        "weights": {h: (None if w is None else w.tolist()) for h, w in class_weights.items()},
        "train_counts": train_counts,
    }
    timings = {
        "data_s": t["data"] - t["start"],
        "vectorizer_fit_s": t["vectorizer_fit"] - t["data"],
        "vectorize_s": t["vectorize"] - t["vectorizer_fit"],
        "train_s": t["train"] - t["vectorize"],
        "epochs_run": len(history),
        "mean_epoch_s": float(hist["epoch_time_s"].mean()) if len(hist) else 0.0,
        "evaluate_s": t["evaluate"] - t["train"],
        "peak_rss_mb": peak_rss_mb(),
        "device": str(device),
    }
    meta = {
        "seed": cfg.seed,
        "input": file_fingerprint(cfg.data.path),
        "data_report": data.report,
        "feature_stats": feature_stats,
        "best_epoch": best,
        "param_counts": pc.to_dict(orient="records"),
        "timings": timings,
    }
    save_artifact(art_dir, model, cfg.model, fb, maps, rule, sel, cw_json, cfg.to_dict(), meta)
    timings["total_s"] = time.perf_counter() - t["start"]

    # Отчёт на диск
    save_eval(ev, rep_dir, prefix="test__")
    hist.to_csv(os.path.join(rep_dir, "history.csv"), index=False)
    tt.to_csv(os.path.join(rep_dir, "tau_table_val.csv"), index=False)
    pc.to_csv(os.path.join(rep_dir, "param_counts.csv"), index=False)
    with open(os.path.join(rep_dir, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data": data.report, "features": feature_stats, "best_epoch": best,
            "tau_selection": sel, "coverage_test": ev["coverage"],
            "n_empty_features_test": ev["n_empty_features"],
            "n_noisy_l3_classes_test": ev["n_noisy_l3_classes"],
            "timings": timings, "limitations": LIMITATIONS,
        }, f, ensure_ascii=False, indent=2, default=str)
    log.info("Артефакт: %s; отчёт: %s; %.1f с, пик RSS %.0f МБ", art_dir, rep_dir, timings["total_s"], timings["peak_rss_mb"])
    return TrainResult(
        config=cfg, output_dir=out, artifact_dir=art_dir, report_dir=rep_dir, data=data,
        feature_stats=feature_stats, param_counts=pc, history=hist, best_epoch=best,
        tau_table=tt, tau_selection=sel, evaluation=ev, test_predictions=test_frame, timings=timings,
    )


def verify_roundtrip(result: TrainResult, n_rows: int = 2000) -> bool:
    """Загружает артефакт и сравнивает предсказания на test с полученными до сохранения."""
    pred = Predictor.load(result.artifact_dir, result.config.train.device)
    names = result.data.splits["test"]["name"].tolist()[:n_rows]
    after = pred.predict(names, top_k=3)
    before = result.test_predictions.iloc[: len(names)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-5, atol=1e-6)
    return True
