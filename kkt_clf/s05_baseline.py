"""S05 (D04–D08): контур обучения baseline MLP на едином полном TF-IDF.

Реализует обучающий контур спринта S05 поверх замороженных сплитов
(``outputs/S05/splits/manifest.json``, см. ``kkt_clf.freeze_split``) и одного
проверенного сохранённого full-TFIDF S02 (по умолчанию ``outputs/S02/tfidf_1m``).
Feature selection не выполняется; режим векторизатора — только ``pretrained``.

Протокол (границы частей):

- ``train_fit`` / ``train_monitor`` — единственные данные для выбора числа эпох
  и конфигурации (фаза selection, D05): MLP обучается на train_fit с early
  stopping по train_monitor, кривые train/monitor сохраняются (train-метрики
  считаются каждую эпоху — без них train-сторона кривых пуста и по history
  нельзя оценить train↔monitor gap, D05/D07).
- ``outer_train`` — финальный refit (D06/D08): отдельное обучение на всём
  outer_train ровно на зафиксированном числе эпох с **фиксированным заранее
  расписанием LR**, общим для selection и refit: адаптивный ``plateau``
  заменяется постоянным LR в сборке конфигурации (до любого обучения), а не
  молча только в refit; cosine/onecycle допустимы (зависят только от шага).
  Checkpoint выбора по monitor **нет** (``restore_best=False`` — веса
  финальной эпохи).
  Comparison validation до завершения всех refit не трансформируется и не
  оценивается; sealed test не используется ни разу. Monitor-кривые фазы refit
  — метрики на ``train_monitor`` (⊂ ``outer_train``), это train-internal
  протокол, а не независимая выборка (явно отражено в provenance).
- ``comparison_val`` — одна оценка raw top-1 L3 без отказа **после** refit
  каждого seed: accuracy и macro-F1 по фиксированной L3-вселенной label maps
  из manifest; классы с support=0 дают null, а не 0. Пустые векторы признаков
  считаются отдельно. Для контроля считается тривиальный majority-class
  baseline (большинство по outer train) на том же comparison_val.
- Опорный checkpoint для S06 (D08) выбирается **заранее заданным детерминированным
  train-only правилом** — ``reference_seed`` (по умолчанию 42), не по качеству
  на comparison_val; выполняется round-trip проверка загрузки: полный контракт
  признаков (словари/IDF/порядок столбцов), sha256 full TF-IDF и train-сплитов,
  число эпох, и независимый inference через ``transform`` векторизатора из
  checkpoint (без внешне построенной матрицы).
- Завершившиеся результаты не теряются при сбоях (D09): per-seed refit-история
  пишется сразу после обучения seed, агрегаты (runs/per_class/history/stages/
  majority/selected/manifest) — даже при падении оценки отдельных seed или
  проверки опорного checkpoint (контур при этом завершается ошибкой).

Все результаты пишутся только в новые файлы под ``output_dir`` (по умолчанию
``outputs/S05``): существующие файлы и каталоги не перезаписываются, при
обнаружении пересечения — отказ до любых записей. Крупные выходы (checkpoints)
— вне Git, каталог настраивается.

Реализация переиспользует существующий цикл обучения (``train.train_model``),
артефакты (``artifact.save_artifact`` / ``load_artifact``), замеры ресурсов и
метрики raw top-1 (``experiment._Measurement``, ``experiment._l3_val_metrics``);
тренировочный цикл не дублируется и не переписывается.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch

from .artifact import environment_versions, load_artifact, save_artifact
from .attribution import feature_contract_fingerprint
from .config import HEADS, LossConfig, ModelConfig, TrainConfig, _merge, build_config
from .data import file_fingerprint
from .experiment import _Measurement, _git_info, _l3_val_metrics, _sha256_file, _sha256_json, _write_json
from .features import matrix_stats
from .labels import LEVELS, LabelMaps
from .model import SparseMLP
from .reject import RejectRule
from .train import compute_class_weights, predict, resolve_device, set_seed, train_model
from .vectorizer import LoadedVectorizer, load_vectorizer_artifact

log = logging.getLogger(__name__)

S05_SCHEMA_VERSION = "s05-training@1"

PROTOCOL = {
    "selection": (
        "train-only выбор числа эпох/конфигурации: fit на train_fit, ранний "
        "останов/выбор best epoch по train_monitor (внутри outer_train)"
    ),
    "refit": (
        "финальный refit на всём outer_train: фиксированное число эпох из "
        "selection (или --fixed-epochs), фиксированное заранее расписание LR, "
        "общее с selection (plateau заменяется постоянным LR до обучения — "
        "адаптация по monitor для контура S05 недопустима), веса финальной "
        "эпохи без выбора checkpoint (restore_best=False); monitor ⊂ "
        "outer_train — train-internal протокол"
    ),
    "evaluation": (
        "одна оценка raw top-1 L3 без отказа на comparison_val после refit "
        "всех seed; macro-F1 по классам с support>0 фиксированной L3-вселенной "
        "label maps; support=0 → null; τ/отказ не применяются"
    ),
    "checkpoint_rule": (
        "опорный checkpoint для S06 — заранее заданный reference_seed (train-only "
        "правило), не по качеству на comparison_val; round-trip проверка загрузки"
    ),
    "boundaries": (
        "comparison_val не используется до завершения всех refit и никогда для "
        "выбора checkpoint; sealed_test не используется ни для чего (ни метрики, "
        "ни выбор); метки val/test не участвуют в fit TF-IDF (pretrained)"
    ),
    "transductive": (
        "режим pretrained трансдуктивен: тексты корпуса full TF-IDF могут "
        "совпадать с train/val/test; метки при этом не переносятся"
    ),
}

RUNS_COLUMNS = [
    "run_id", "config_id", "train_seed", "split_seed", "status", "reason",
    "n_train", "n_monitor", "n_val", "n_classes", "full_tfidf_id",
    "epochs_completed", "best_epoch", "l3_macro_f1", "l3_accuracy",
    "metric_evaluated_on", "train_seconds", "peak_rss_bytes", "checkpoint_path",
]
STAGES_COLUMNS = [
    "run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
    "measurement_note",
]
HISTORY_COLUMNS = [
    "run_id", "epoch", "split", "loss", "l3_macro_f1", "l3_accuracy", "lr",
]
PER_CLASS_COLUMNS = ["run_id", "L3", "support", "precision", "recall", "f1"]
SHARED_VECTORIZATION_RUN_ID = "shared_vectorization"


class S05BaselineError(RuntimeError):
    """Базовая ошибка контура обучения S05; файлы не записываются."""


@dataclass
class S05Config:
    """Конфигурация контура обучения S05.

    ``preset`` — базовая MLP/loss/train-конфигурация (по умолчанию ``ods_light``
    как ориентир относительно модели; источник признаков — только полный
    TF-IDF). ``model``/``loss``/``train`` — рекурсивные переопределения пресета.
    ``sweep`` — ограниченный список кандидатов фазы selection
    (``[{"name": ..., "model": {...}, "train": {...}}]``); выбор кандидата —
    только по train_monitor, каждый кандидат фиксируется в отчёте selection.
    """

    output_dir: str = "outputs/S05"
    splits_manifest: str = "outputs/S05/splits/manifest.json"
    full_tfidf_path: str = "outputs/S02/tfidf_1m"
    corpus_path: str = "data/corpus.pqt"
    preset: str = "ods_light"
    model: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    sweep: list[dict[str, Any]] = field(default_factory=list)
    features_chunk_rows: int = 50_000
    selection_seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    reference_seed: int = 42
    fixed_epochs: int | None = None
    device: str = "auto"

    def build_sections(self, model_over: dict | None = None,
                       train_over: dict | None = None,
                       loss_over: dict | None = None) -> tuple[ModelConfig, LossConfig, TrainConfig]:
        base = build_config(self.preset, {})
        model_merged = {**self.model, **(model_over or {})}
        train_merged = {**self.train, **(train_over or {})}
        loss_merged = {**self.loss, **(loss_over or {})}
        model_cfg = _merge(base.model, model_merged)
        loss_cfg = _merge(base.loss, loss_merged)
        train_cfg = _merge(base.train, train_merged)
        # Протокол S05 (D05/D07): train-метрики каждую эпоху — без них
        # train-сторона history.csv пуста (null) и train↔monitor gap по кривым
        # не оценить. Отключение не поддерживается: это часть протокола обучения.
        train_cfg.eval_train_each_epoch = True
        # Единая семантика macro-F1 контура S05: monitor/train-кривые обучения
        # считаются по классам с support>0, как каноническая оценка
        # experiment._l3_val_metrics (selection/refit и итоговая оценка — одна
        # метрика). Прежняя семантика baseline ("true_or_predicted") сохраняется
        # дефолтом конфигурации; здесь она включается явно конфигурацией S05.
        train_cfg.macro_f1_semantics = "supported"
        # Фиксированное заранее расписание LR (D05/D06), решается ДО любого
        # обучения: selection и refit обязаны идти с одним и тем же расписанием,
        # иначе выбранное число эпох относится к другому протоколу. Адаптивный
        # plateau (подстройка по loss на train_monitor) для контура S05
        # недопустим и заменяется на постоянный LR здесь, в сборке конфигурации
        # (для selection и refit одинаково), а не молча только в refit.
        # cosine/onecycle зависят только от номера шага — допустимы как есть.
        if train_cfg.scheduler == "plateau":
            train_cfg = dataclasses.replace(train_cfg, scheduler="none")
            self._lr_schedule_note = (
                "scheduler 'plateau' (адаптация по loss на train_monitor) заменён на "
                "постоянный LR 'none' до обучения: selection и refit идут с одним и "
                "тем же фиксированным заранее расписанием LR"
            )
        return model_cfg, loss_cfg, train_cfg

    def validate(self) -> None:
        if self.reference_seed not in self.seeds:
            raise S05BaselineError(
                f"reference_seed={self.reference_seed} должен входить в seeds={self.seeds}: "
                "опорный checkpoint выбирается заранее заданным правилом и обязан быть обучен"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise S05BaselineError(f"seeds должны быть различны: {self.seeds}")
        if self.features_chunk_rows <= 0:
            raise S05BaselineError("features_chunk_rows: > 0")
        for cand in self.sweep:
            if not isinstance(cand, dict) or not isinstance(cand.get("name"), str) or not cand["name"]:
                raise S05BaselineError(f"sweep: каждый кандидат — объект с непустым 'name': {cand!r}")
        if self.fixed_epochs is not None and self.fixed_epochs < 1:
            raise S05BaselineError("fixed_epochs: None или >= 1")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Загрузка и проверка входов (D04)
# ---------------------------------------------------------------------------


def load_split_manifest(path: str) -> dict[str, Any]:
    """Читает manifest замороженного сплита и сверяет части по sha256/числу строк.

    Fail closed: несовпадение хэша/счётчика, отсутствие файла или обязательных
    секций — S05BaselineError, работа не начинается.
    """
    if not os.path.isfile(path):
        raise S05BaselineError(f"manifest сплитов не найден: {path!r}")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    for key in ("parts", "label_maps", "policy", "inputs"):
        if key not in manifest:
            raise S05BaselineError(f"manifest {path!r}: нет обязательной секции {key!r}")
    for name, part in manifest["parts"].items():
        fp = part.get("file")
        if not fp or not os.path.isfile(fp):
            raise S05BaselineError(f"часть сплита {name!r}: файл не найден: {fp!r}")
        actual = file_fingerprint(fp)
        if actual["sha256"] != part.get("sha256") or actual["bytes"] != part.get("bytes"):
            raise S05BaselineError(
                f"часть сплита {name!r}: sha256/bytes изменились (ожидалось "
                f"{str(part.get('sha256'))[:16]}…/{part.get('bytes')}, фактически "
                f"{actual['sha256'][:16]}…/{actual['bytes']}): сплит не тот, что заморожен"
            )
        n_actual = pq.ParquetFile(fp).metadata.num_rows
        if n_actual != part.get("n_rows"):
            raise S05BaselineError(
                f"часть сплита {name!r}: число строк {n_actual} != зафиксированному {part.get('n_rows')}"
            )
    return manifest


def load_part(manifest: dict[str, Any], name: str) -> pd.DataFrame:
    return pd.read_parquet(manifest["parts"][name]["file"])


def verify_full_tfidf(full_tfidf_path: str, corpus_path: str) -> dict[str, Any]:
    """D04: загружает единый full-TFIDF с полной проверкой целостности и provenance.

    Проверки (отказ при любом нарушении, молчаливая подмена недопустима):
    стандартная верификация артефакта (метаданные, контрольные суммы файлов,
    fit-отчёт, словари, IDF) через ``load_vectorizer_artifact``; фактический
    корпус по ``corpus_path`` совпадает с зафиксированным источником по
    sha256/bytes. Возвращает сводку с идентификатором артефакта
    (sha256 features.joblib) для ссылок из runs/manifest.
    """
    loaded = load_vectorizer_artifact(full_tfidf_path)
    meta = loaded.meta
    if not os.path.isfile(corpus_path):
        raise S05BaselineError(f"D04: корпус full TF-IDF не найден: {corpus_path!r}")
    corpus_fp = file_fingerprint(corpus_path)
    src = meta.get("source") or {}
    corpus_matches = corpus_fp["sha256"] == src.get("sha256") and corpus_fp["bytes"] == src.get("bytes")
    if not corpus_matches:
        raise S05BaselineError(
            f"D04: корпус {corpus_path!r} не совпадает с источником full TF-IDF "
            f"(sha256/bytes: ожидалось {str(src.get('sha256'))[:16]}…/{src.get('bytes')}, "
            f"фактически {corpus_fp['sha256'][:16]}…/{corpus_fp['bytes']}): артефакт "
            "несовместим или корпус подменён, обучение не начинается."
        )
    artifact_id = (meta.get("files") or {}).get("features.joblib", {}).get("sha256")
    return {
        "schema_version": S05_SCHEMA_VERSION,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "artifact_dir": os.path.abspath(full_tfidf_path),
        "full_tfidf_id": artifact_id,
        "artifact_version": meta.get("artifact_version"),
        "seed": meta.get("seed"),
        "vocab": meta.get("vocab"),
        "fit": meta.get("fit"),
        "source": {k: v for k, v in src.items()},
        "corpus_fingerprint": corpus_fp,
        "corpus_matches_source": True,
        "files": {
            name: {"sha256": _sha256_file(os.path.join(full_tfidf_path, name)),
                   "bytes": os.path.getsize(os.path.join(full_tfidf_path, name))}
            for name in ("features.joblib", "vectorizer_meta.json")
            if os.path.exists(os.path.join(full_tfidf_path, name))
        },
        "mode": "pretrained (fit не выполняется, transform без повторного обучения)",
        "decision": "ok",
        "notes": [
            "Используется один сохранённый артефакт для всех seed/прогонов S05.",
            PROTOCOL["transductive"],
        ],
    }


def record_tfidf_verification(output_dir: str, verification: dict[str, Any]) -> str:
    """Сохраняет проверенный provenance full TF-IDF для отчёта (D04/gate).

    ``tfidf_verification.json`` пишется, только если отсутствует (существующие
    outputs не перезаписываются). Если файл уже есть, сверяет с текущей
    проверкой ``decision``/``full_tfidf_id``/версию артефакта и fingerprint
    корпуса: расхождение — отказ (fail closed) до любых записей, совпадение —
    существующий файл сохраняется. Благодаря этому раздельные команды
    (``verify-tfidf``/``select``, затем ``refit``) оставляют проверенный
    provenance для gate отчёта, а подмена артефакта/корпуса между фазами
    отклоняется.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tfidf_verification.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        problems = [
            key for key in ("decision", "full_tfidf_id", "artifact_version")
            if existing.get(key) != verification.get(key)
        ]
        ex_corpus = (existing.get("corpus_fingerprint") or {})
        new_corpus = (verification.get("corpus_fingerprint") or {})
        if (ex_corpus.get("sha256") != new_corpus.get("sha256")
                or ex_corpus.get("bytes") != new_corpus.get("bytes")):
            problems.append("corpus_fingerprint")
        if problems:
            raise S05BaselineError(
                f"{path} уже существует и не совпадает с текущей проверкой full "
                f"TF-IDF по полям {problems}: подмена артефакта/корпуса между "
                "фазами недопустима, отказ до любых записей."
            )
        log.info("%s уже существует и совпадает с текущей проверкой full TF-IDF", path)
        return path
    _write_json(verification, path)
    return path


# ---------------------------------------------------------------------------
# Общие helpers
# ---------------------------------------------------------------------------


def _check_new(paths: Sequence[str]) -> None:
    """Отказ, если любой целевой путь уже существует (ничего не перезаписываем)."""
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise S05BaselineError(
            f"Существующие outputs не перезаписываются; отказ до любых записей: {existing}"
        )


def _encode_labels(maps: LabelMaps, df: pd.DataFrame, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = []
    for lvl in LEVELS:
        ids = maps.encode(lvl, df[lvl].to_numpy(dtype=object))
        if (ids < 0).any():
            bad = sorted(set(df.loc[ids < 0, lvl].astype(str)))[:5]
            raise S05BaselineError(
                f"Часть {part!r}: метки вне зафиксированной вселенной {lvl}: {bad}"
            )
        out.append(ids)
    return tuple(out)  # type: ignore[return-value]


def _transform(loaded: LoadedVectorizer, names: list[str], chunk_rows: int) -> sp.csr_matrix:
    return loaded.builder.transform(names, chunk_rows=chunk_rows)


def _empty_rows(X: sp.csr_matrix) -> int:
    return int((np.diff(X.indptr) == 0).sum())


def _reject_stub() -> dict[str, Any]:
    return {
        "note": "S05: τ не калибровался; сравнение по raw top-1 на comparison_val без отказа",
        "tau_mode": "fixed",
    }


def _history_rows(run_id: str, history: list[dict]) -> list[dict[str, Any]]:
    """История эпох train_model → длинный формат train/monitor кривых.

    Train-сторона заполняется фактическими метриками обучающей части
    (``train_l3_*`` считаются при ``eval_train_each_epoch``), а не null —
    иначе кривые train пустые и gap train↔monitor по history.csv недостоверен.
    """
    rows: list[dict[str, Any]] = []
    for rec in history:
        rows.append({
            "run_id": run_id, "epoch": int(rec["epoch"]), "split": "train",
            "loss": rec["train_loss"],
            "l3_macro_f1": rec.get("train_l3_macro_f1"),
            "l3_accuracy": rec.get("train_l3_accuracy"),
            "lr": rec["lr"],
        })
        rows.append({
            "run_id": run_id, "epoch": int(rec["epoch"]), "split": "monitor",
            "loss": rec["val_loss"], "l3_macro_f1": rec["val_l3_macro_f1"],
            "l3_accuracy": rec["val_l3_accuracy"], "lr": rec["lr"],
        })
    return rows


# ---------------------------------------------------------------------------
# D05: train-only выбор числа эпох/конфигурации (train_fit / train_monitor)
# ---------------------------------------------------------------------------


def run_selection(
    cfg: S05Config,
    manifest: dict[str, Any],
    loaded: LoadedVectorizer,
    device: torch.device,
) -> dict[str, Any]:
    """Фаза selection: кандидаты обучаются на train_fit с early stopping по
    train_monitor; выбранное число эпох — best_epoch лучшего кандидата по
    метрике train.* (по умолчанию val_l3_macro_f1 на train_monitor).

    Записывает ``selection/selection_report.json`` и
    ``selection/curves_train_monitor.csv`` (оба — только новые файлы).
    Возвращает сводку для последующего refit.
    """
    cfg.validate()
    maps = LabelMaps.from_dict(manifest["label_maps"])
    fit_df = load_part(manifest, "train_fit")
    mon_df = load_part(manifest, "train_monitor")
    y_fit = _encode_labels(maps, fit_df, "train_fit")
    y_mon = _encode_labels(maps, mon_df, "train_monitor")

    out_dir = cfg.output_dir
    sel_dir = os.path.join(out_dir, "selection")
    report_path = os.path.join(sel_dir, "selection_report.json")
    curves_path = os.path.join(sel_dir, "curves_train_monitor.csv")
    _check_new([report_path, curves_path])
    os.makedirs(sel_dir, exist_ok=True)

    stage_rows: list[dict[str, Any]] = []
    with _Measurement(device, f"transform train_fit/train_monitor полным TF-IDF, chunk_rows={cfg.features_chunk_rows}") as m:
        X_fit = _transform(loaded, fit_df["name"].tolist(), cfg.features_chunk_rows)
        X_mon = _transform(loaded, mon_df["name"].tolist(), cfg.features_chunk_rows)
    stage_rows.append(m.stage_row("selection", "vectorize", "success"))
    log.info(
        "Selection: train_fit %d строк (пустых %d), train_monitor %d (пустых %d), признаков %d",
        X_fit.shape[0], _empty_rows(X_fit), X_mon.shape[0], _empty_rows(X_mon), X_fit.shape[1],
    )

    candidates = [{"name": "base", "model": {}, "train": {}}] + [dict(c) for c in cfg.sweep]
    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names):
        raise S05BaselineError(f"Имена кандидатов selection должны быть различны: {names}")

    # Кандидаты сравниваются по best_score на train_monitor, поэтому метрика
    # early stopping обязана быть одна и та же для всех: иначе выбор сводил бы
    # вместе несопоставимые числа (например, val_loss одного кандидата и
    # macro-F1 другого). Fail closed до любого обучения.
    cand_metrics = {
        cfg.build_sections(c.get("model"), c.get("train"))[2].early_stopping_metric
        for c in candidates
    }
    if len(cand_metrics) > 1:
        raise S05BaselineError(
            f"Кандидаты selection используют разные early_stopping_metric {sorted(cand_metrics)}: "
            "best_score несопоставимы, train-only выбор конфигурации не определён. "
            "Задайте одну и ту же метрику для всех кандидатов sweep."
        )

    results: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for cand in candidates:
        model_cfg, loss_cfg, train_cfg = cfg.build_sections(cand.get("model"), cand.get("train"))
        metric = train_cfg.early_stopping_metric
        set_seed(cfg.selection_seed)
        model = SparseMLP(
            loaded.builder.n_features,
            tuple(maps.n_classes(lvl) for lvl in LEVELS),
            model_cfg,
        )
        class_weights = {
            h: compute_class_weights(y_fit[i], maps.n_classes(lvl), loss_cfg.class_weight_mode[h],
                                     loss_cfg.class_weight_alpha)
            for i, (h, lvl) in enumerate(zip(HEADS, LEVELS))
        }
        run_id = f"selection:{cand['name']}"
        with _Measurement(
            device,
            f"обучение кандидата {cand['name']!r} на train_fit (n={X_fit.shape[0]}), "
            f"early stopping по train_monitor, метрика {metric}; выбор только по train-internal monitor",
        ) as meas:
            history, best, lr_trace = train_model(
                model, X_fit, y_fit, X_mon, y_mon,
                loss_cfg, train_cfg, class_weights, device, cfg.selection_seed,
            )
        stage_rows.append(meas.stage_row(run_id, "selection_train", "success"))
        curves.extend(_history_rows(run_id, history))
        last = history[-1] if history else {}
        results.append({
            "name": cand["name"],
            "model": dataclasses.asdict(model_cfg),
            "train": dataclasses.asdict(train_cfg),
            "loss": dataclasses.asdict(loss_cfg),
            "epochs_run": len(history),
            "best_epoch": int(best["epoch"]),
            "best_score": float(best["score"]),
            "early_stopping_metric": metric,
            "final_monitor_l3_macro_f1": last.get("val_l3_macro_f1"),
            "final_monitor_l3_accuracy": last.get("val_l3_accuracy"),
            "train_seconds": meas.seconds,
            "peak_rss_bytes": meas.peak_rss_bytes,
            "peak_vram_bytes": meas.peak_vram_bytes,
            "n_train_rows_used": int(best.get("n_train_rows_used", 0)),
        })
        log.info(
            "Кандидат %s: best_epoch=%d (%s=%.4f), эпох выполнено %d",
            cand["name"], best["epoch"], metric, best["score"], len(history),
        )

    higher_better = metric != "val_loss"
    chosen = max(results, key=lambda r: r["best_score"]) if higher_better else min(
        results, key=lambda r: r["best_score"]
    )
    report = {
        "schema_version": S05_SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": PROTOCOL["selection"],
        "metric": metric,
        "higher_better": higher_better,
        "selection_seed": int(cfg.selection_seed),
        "candidates": results,
        "chosen": {
            "name": chosen["name"],
            "fixed_epochs": int(chosen["best_epoch"]),
            "criterion": f"best {metric} на train_monitor (train-only; comparison_val не использовался)",
        },
        "splits": {
            "train_fit": manifest["parts"]["train_fit"]["sha256"],
            "train_monitor": manifest["parts"]["train_monitor"]["sha256"],
        },
        "full_tfidf_id": (loaded.meta.get("files") or {}).get("features.joblib", {}).get("sha256"),
        "stages": stage_rows,
    }
    _write_json(report, report_path)
    pd.DataFrame(curves, columns=HISTORY_COLUMNS).to_csv(curves_path, index=False)
    log.info("Selection: выбран кандидат %s, фиксированные эпохи %d", chosen["name"], chosen["best_epoch"])
    return report


def read_selection(output_dir: str) -> dict[str, Any]:
    path = os.path.join(output_dir, "selection", "selection_report.json")
    if not os.path.isfile(path):
        raise S05BaselineError(
            f"Нет {path!r}: сначала выполните фазу selection (train-only выбор эпох/конфигурации)"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _verify_selection_provenance(
    selection: dict[str, Any] | None,
    manifest: dict[str, Any],
    loaded: LoadedVectorizer,
) -> None:
    """Сверяет provenance selection-report с фактическими входами refit.

    Selection (число эпох/конфигурация) валиден только для того сплита и того
    full TF-IDF, на которых выполнялся: применять эпохи от артефакта A к
    сплиту/векторизатору B — молчаливое нарушение протокола. Проверка fail
    closed и выполняется **до любого обращения к данным и обучения**: при
    несовпадении sha256 сплитов или full_tfidf_id refit не начинается.
    """
    if selection is None:
        return
    problems: list[str] = []
    sel_splits = selection.get("splits") or {}
    for part in ("train_fit", "train_monitor"):
        expected = sel_splits.get(part)
        actual = (manifest["parts"].get(part) or {}).get("sha256")
        if expected != actual:
            problems.append(
                f"{part}: selection зафиксирован для sha256 {str(expected)[:16]}…, "
                f"текущий сплит {str(actual)[:16]}…"
            )
    sel_tfidf = selection.get("full_tfidf_id")
    actual_tfidf = (loaded.meta.get("files") or {}).get("features.joblib", {}).get("sha256")
    if sel_tfidf != actual_tfidf:
        problems.append(
            f"full_tfidf_id: selection относится к {str(sel_tfidf)[:16]}…, "
            f"загружен артефакт {str(actual_tfidf)[:16]}…"
        )
    if problems:
        raise S05BaselineError(
            "Selection report несовместим с текущими входами (сплит / full TF-IDF): "
            + "; ".join(problems)
        )


# ---------------------------------------------------------------------------
# D06–D08: финальный refit на outer_train (seed 42/43/44) + оценка + checkpoint
# ---------------------------------------------------------------------------


_VERIFICATION_UNSET: Any = object()
"""Сентинел 'проверка TF-IDF не передана' для run_refit (отличие от явного None)."""


def run_refit(
    cfg: S05Config,
    manifest: dict[str, Any],
    loaded: LoadedVectorizer,
    device: torch.device,
    selection: dict[str, Any] | None = None,
    tfidf_verification: dict[str, Any] | None | Any = _VERIFICATION_UNSET,
) -> dict[str, Any]:
    """Финальный контур refit+оценка (D06–D08).

    Порядок гарантирует границы протокола: проверка provenance full TF-IDF
    (fail closed до любых данных и обучения), проверка provenance selection
    (sha256 сплитов и full_tfidf_id против текущих входов), затем transform
    outer_train и обучение всех seed ровно на фиксированное число эпох.
    Расписание LR фиксируется заранее (в сборке конфигурации, до selection и
    refit одинаково): адаптивный ``plateau`` заменяется постоянным LR там, а не
    молча в refit; comparison_val до этого момента не читается и не
    оценивается. При использовании selection ВЕСЬ loss выбранного кандидата
    воспроизводится из сохранённого отчёта, а явные переопределения loss
    обязаны совпадать с ним, иначе отказ до обучения. Затем одна оценка raw
    top-1 каждого seed на comparison_val, majority baseline и выбор опорного
    checkpoint по правилу reference_seed с round-trip проверкой.

    Устойчивость к сбоям (D09): результаты каждого завершившегося этапа
    записываются независимо — per-seed history сразу после обучения seed,
    агрегаты (runs/per_class/history/stages/majority/selected/manifest) —
    даже если оценка отдельных seed на comparison_val, общая подготовка
    comparison_val или проверка опорного checkpoint завершились ошибкой; при
    этом контур в целом завершается ошибкой, а не объявляет протокол
    выполненным.

    Записывает (только новые файлы): ``checkpoints/seed_<s>/`` (артефакты
    save_artifact по каждому seed), ``refit/history_seed_<s>.csv``,
    ``refit/run_result_seed_<s>.json``, ``runs.csv``, ``per_class.parquet``,
    ``history.csv`` (включая кривые фазы selection с run_id ``selection:*``),
    ``stages.csv``, ``majority_baseline.json``, ``selected_checkpoint.json``,
    ``manifest.json``.

    ``tfidf_verification`` — результат :func:`verify_full_tfidf` вызывающей
    стороны. Обучение fail closed: явный ``None``, ``decision != 'ok'`` или
    несовпадение ``full_tfidf_id`` с загруженным артефактом — отказ до любых
    данных и обучения. Если параметр пропущен (прямой вызов API), проверка не
    отменяется: run_refit сам выполняет :func:`verify_full_tfidf` над текущими
    ``cfg.full_tfidf_path``/``cfg.corpus_path`` до чтения данных и обучения;
    любое несоответствие артефакта/корпуса — отказ. Проверка не подменяется
    словарём из metadata: сверяются контрольные суммы файлов артефакта и
    fingerprint корпуса. Результат сохраняется через
    :func:`record_tfidf_verification` на всех путях.
    """
    cfg.validate()
    actual_tfidf_id = (loaded.meta.get("files") or {}).get("features.joblib", {}).get("sha256")
    if tfidf_verification is _VERIFICATION_UNSET:
        # Параметр не передан: fail closed без специальных исключений — реальная
        # проверка выполняется здесь, до любого чтения данных и обучения.
        log.info("run_refit: tfidf_verification не передан — выполняется verify_full_tfidf")
        tfidf_verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
    if tfidf_verification is None:
        raise S05BaselineError(
            "run_refit: tfidf_verification=None передан явно — обучение без "
            "проверки full TF-IDF запрещено. Выполните verify_full_tfidf и "
            "передайте её результат (все CLI-пути делают это до обучения)."
        )
    else:
        if tfidf_verification.get("decision") != "ok":
            raise S05BaselineError(
                "run_refit: проверка full TF-IDF не пройдена "
                f"(decision={tfidf_verification.get('decision')!r}): обучение не начинается."
            )
        if tfidf_verification.get("full_tfidf_id") != actual_tfidf_id:
            raise S05BaselineError(
                "run_refit: full_tfidf_id проверки не совпадает с загруженным "
                f"артефактом ({str(tfidf_verification.get('full_tfidf_id'))[:16]}… != "
                f"{str(actual_tfidf_id)[:16]}…): обучение не начинается."
            )
        # Проверенный provenance сохраняется для gate отчёта по каждому пути
        # обучения (D04): при уже существующем файле — сверка совпадения, не
        # перезапись.
        record_tfidf_verification(cfg.output_dir, tfidf_verification)
    maps = LabelMaps.from_dict(manifest["label_maps"])
    split_seed = int(manifest["policy"]["split_seed"])
    chosen = (selection or {}).get("chosen") or {}
    fixed_epochs = int(cfg.fixed_epochs or chosen.get("fixed_epochs") or 0)
    if fixed_epochs < 1:
        raise S05BaselineError(
            "Фиксированное число эпох не задано: передайте fixed_epochs в конфиге "
            "или выполните фазу selection (selection/selection_report.json)"
        )
    loss_over: dict[str, Any] | None = None
    if selection is not None and cfg.fixed_epochs is None:
        cand = next(
            (c for c in selection["candidates"] if c["name"] == chosen["name"]), None
        )
        if cand is None:
            raise S05BaselineError(f"chosen-кандидат {chosen['name']!r} не найден в selection")
        model_over, train_over = cand["model"], cand["train"]
        # ВЕСЬ loss выбранного кандидата воспроизводится из сохранённого
        # отчёта selection (review-2, P1): число эпох и конфигурация модели
        # выбраны при loss фазы selection, поэтому финальный refit обязан
        # обучаться с тем же полным loss, а не заново собирать его из текущего
        # CLI-конфига. Явные переопределения loss текущего вызова обязаны
        # точно совпадать с сохранёнными — иначе отказ до любого обучения
        # (fail closed), а не молчаливая подмена.
        cand_loss = dict(cand.get("loss") or {})
        if cand_loss:
            cli_loss = dataclasses.asdict(_merge(build_config(cfg.preset, {}).loss, cfg.loss))
            if cli_loss != cand_loss:
                raise S05BaselineError(
                    f"loss-конфигурация текущего вызова не совпадает с сохранённой "
                    f"выбранным кандидатом selection {chosen['name']!r}: передайте тот "
                    "же --loss, что и в фазе select, либо не передавайте его "
                    f"(используется сохранённый loss). Ожидалось {cand_loss}, "
                    f"получено {cli_loss}."
                )
            loss_over = cand_loss
    else:
        model_over, train_over = {}, {}
    # До любого обращения к данным: selection применим только к своим входам.
    _verify_selection_provenance(selection, manifest, loaded)
    model_cfg, loss_cfg, train_cfg = cfg.build_sections(model_over, train_over, loss_over)
    # Расписание LR уже фиксировано заранее в build_sections (до selection и
    # refit одинаково); адаптивный plateau заменён там, а не только здесь.
    refit_scheduler_note: str | None = getattr(cfg, "_lr_schedule_note", None)
    if refit_scheduler_note:
        log.info("refit: %s", refit_scheduler_note)
    config_id = chosen.get("name", "base")
    full_tfidf_id = (loaded.meta.get("files") or {}).get("features.joblib", {}).get("sha256")
    log.info(
        "Refit: config_id=%s, фиксированные эпохи=%d, seed=%s, reference_seed=%d, "
        "scheduler=%s (restore_best=False)",
        config_id, fixed_epochs, cfg.seeds, cfg.reference_seed, train_cfg.scheduler,
    )

    train_df = load_part(manifest, "outer_train")
    mon_df = load_part(manifest, "train_monitor")
    y_train = _encode_labels(maps, train_df, "outer_train")
    y_mon = _encode_labels(maps, mon_df, "train_monitor")
    l3_labels = list(maps.l3)
    n_classes = tuple(maps.n_classes(lvl) for lvl in LEVELS)

    out_dir = cfg.output_dir
    ckpt_root = os.path.join(out_dir, "checkpoints")
    refit_dir = os.path.join(out_dir, "refit")
    targets = [
        os.path.join(out_dir, "runs.csv"),
        os.path.join(out_dir, "per_class.parquet"),
        os.path.join(out_dir, "history.csv"),
        os.path.join(out_dir, "stages.csv"),
        os.path.join(out_dir, "majority_baseline.json"),
        os.path.join(out_dir, "selected_checkpoint.json"),
        os.path.join(out_dir, "manifest.json"),
    ]
    for s in cfg.seeds:
        targets.append(os.path.join(refit_dir, f"history_seed_{s}.csv"))
        targets.append(os.path.join(refit_dir, f"run_result_seed_{s}.json"))
        targets.append(os.path.join(ckpt_root, f"seed_{s}"))
    _check_new(targets)
    os.makedirs(refit_dir, exist_ok=True)

    stage_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    # Кривые настоящего train-only selection (train_fit/train_monitor) —
    # первые строки общего history.csv: без них фаза выбора эпох в отчёте
    # невидима и сопоставить selection-кривые с refit нельзя.
    history_rows: list[dict[str, Any]] = []
    sel_curves_path = os.path.join(out_dir, "selection", "curves_train_monitor.csv")
    if selection is not None and os.path.isfile(sel_curves_path):
        history_rows.extend(pd.read_csv(sel_curves_path).to_dict("records"))
    run_results: dict[int, dict[str, Any]] = {}
    class_weights = {
        h: compute_class_weights(y_train[i], n_classes[i], loss_cfg.class_weight_mode[h],
                                 loss_cfg.class_weight_alpha)
        for i, h in enumerate(HEADS)
    }
    cw_json = {
        "mode": loss_cfg.class_weight_mode,
        "alpha": float(loss_cfg.class_weight_alpha),
        "weights": {h: (None if w is None else w.tolist()) for h, w in class_weights.items()},
        "train_counts": {
            lvl: train_df[lvl].value_counts().sort_index().astype(int).to_dict() for lvl in LEVELS
        },
    }

    # --- transform обучающих частей (comparison_val НЕ трогаем до конца refit)
    with _Measurement(device, f"transform outer_train/train_monitor полным TF-IDF, chunk_rows={cfg.features_chunk_rows}") as m:
        X_train = _transform(loaded, train_df["name"].tolist(), cfg.features_chunk_rows)
        X_mon = _transform(loaded, mon_df["name"].tolist(), cfg.features_chunk_rows)
    stage_rows.append(m.stage_row(SHARED_VECTORIZATION_RUN_ID, "vectorize_train", "success"))
    log.info(
        "Refit: outer_train %d строк (пустых %d), monitor %d (пустых %d), признаков %d",
        X_train.shape[0], _empty_rows(X_train), X_mon.shape[0], _empty_rows(X_mon), X_train.shape[1],
    )

    # --- refit каждого seed на фиксированное число эпох
    for seed in cfg.seeds:
        run_id = f"seed_{seed}"
        meas: _Measurement | None = None
        try:
            set_seed(seed)
            model = SparseMLP(loaded.builder.n_features, n_classes, model_cfg)
            fixed_cfg = dataclasses.replace(train_cfg, epochs=fixed_epochs, patience=fixed_epochs)
            with _Measurement(
                device,
                f"seed {seed}: финальный refit на outer_train (n={X_train.shape[0]}), ровно "
                f"{fixed_epochs} эпох, scheduler={fixed_cfg.scheduler}, restore_best=False "
                "(веса финальной эпохи; comparison_val не используется)",
            ) as meas:
                history, best, lr_trace = train_model(
                    model, X_train, y_train, X_mon, y_mon,
                    loss_cfg, fixed_cfg, class_weights, device, seed,
                    restore_best=False,
                )
            stage_rows.append(meas.stage_row(run_id, "mlp_train", "success"))
            ckpt_dir = os.path.join(ckpt_root, run_id)
            save_artifact(
                ckpt_dir, model, model_cfg, loaded.builder, maps,
                RejectRule(0.5, False, False), _reject_stub(), cw_json,
                {
                    "experiment_run": run_id,
                    "role": "s05_final_refit",
                    "config": {
                        "preset": cfg.preset, "model": dataclasses.asdict(model_cfg),
                        "loss": dataclasses.asdict(loss_cfg), "train": dataclasses.asdict(fixed_cfg),
                    },
                },
                {
                    "seed": int(seed),
                    "split_seed": int(split_seed),
                    "n_features": int(loaded.builder.n_features),
                    "device": str(device),
                    "protocol": S05_SCHEMA_VERSION,
                    "train": {
                        "epochs_completed": len(history),
                        "fixed_epochs": int(fixed_epochs),
                        "scheduler": str(train_cfg.scheduler),
                        "scheduler_note": refit_scheduler_note,
                        "best_epoch_monitor": int(best["epoch"]),
                        "early_stopping_metric": best["metric"],
                        "best_score_monitor": float(best["score"]),
                        "restored_best": bool(best.get("restored_best")),
                        "n_train_rows_used": int(best.get("n_train_rows_used", 0)),
                        "final_monitor_l3_macro_f1": (history[-1]["val_l3_macro_f1"] if history else None),
                        "final_monitor_l3_accuracy": (history[-1]["val_l3_accuracy"] if history else None),
                    },
                    "feature_stats": {
                        "outer_train": matrix_stats(X_train),
                        "train_monitor": matrix_stats(X_mon),
                    },
                    "train_fingerprints": {
                        "outer_train_sha256": manifest["parts"]["outer_train"]["sha256"],
                        "train_monitor_sha256": manifest["parts"]["train_monitor"]["sha256"],
                    },
                    "full_tfidf": {
                        "path": os.path.abspath(cfg.full_tfidf_path),
                        "artifact_version": loaded.meta.get("artifact_version"),
                        "full_tfidf_id": (loaded.meta.get("files") or {})
                        .get("features.joblib", {}).get("sha256"),
                    },
                    "selection": {
                        "selection_seed": int(cfg.selection_seed),
                        "config_id": config_id,
                        "criterion": chosen.get("criterion"),
                    },
                },
            )
            history_rows.extend(_history_rows(run_id, history))
            # Per-seed результат фиксируется сразу: при сбое последующих seed
            # или проверки checkpoint завершившийся refit остаётся на диске.
            pd.DataFrame(history).to_csv(
                os.path.join(refit_dir, f"history_seed_{seed}.csv"), index=False
            )
            run_results[seed] = {
                "run_id": run_id, "seed": seed, "status": "trained",
                "checkpoint_dir": os.path.abspath(ckpt_dir),
                "history": history, "best": best,
                "train_seconds": meas.seconds,
                "peak_rss_bytes": meas.peak_rss_bytes,
                "peak_vram_bytes": meas.peak_vram_bytes,
                "lr_trace": lr_trace,
            }
            log.info(
                "seed %d: refit завершён (%d эпох, %.1f с), checkpoint %s",
                seed, len(history), meas.seconds or 0.0, ckpt_dir,
            )
        except Exception as e:  # noqa: BLE001 — фиксируем failed seed, остальные продолжают
            log.exception("seed %d: refit failed", seed)
            if meas is not None and meas.seconds is not None:
                row = meas.stage_row(run_id, "mlp_train", "failed")
                row["measurement_note"] += f" Этап прерван: {type(e).__name__}: {e}"
                stage_rows.append(row)
            else:
                stage_rows.append({
                    "run_id": run_id, "stage": "mlp_train", "status": "failed",
                    "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
                    "measurement_note": f"refit прерван: {type(e).__name__}: {e}"[:500],
                })
            run_rows.append({
                "run_id": f"seed_{seed}", "config_id": config_id, "train_seed": seed,
                "split_seed": split_seed, "status": "failed",
                "reason": f"{type(e).__name__}: {e}"[:500],
                "n_train": int(len(train_df)), "n_monitor": int(len(mon_df)),
                "n_val": int(manifest["parts"]["comparison_val"]["n_rows"]), "n_classes": int(n_classes[2]),
                "full_tfidf_id": full_tfidf_id,
                "epochs_completed": None, "best_epoch": None,
                "l3_macro_f1": None, "l3_accuracy": None,
                "metric_evaluated_on": "comparison_val",
                "train_seconds": None, "peak_rss_bytes": None, "checkpoint_path": None,
            })
            run_results[seed] = {"run_id": f"seed_{seed}", "seed": seed, "status": "failed",
                                 "reason": f"{type(e).__name__}: {e}"[:500]}

    git_commit, git_dirty = _git_info()
    eval_rows: dict[int, dict[str, Any]] = {}

    def build_manifest(seed_stats: dict[str, Any] | None,
                       selected_status: str | None) -> dict[str, Any]:
        limitations = [
            "ODS/PoC: результаты на comparison_val диагностические, не оценка промышленной применимости.",
            "Сеaled_test не использовался ни для чего; test-метрик нет.",
            PROTOCOL["transductive"],
            "Режим pretrained: идентичность словаря/IDF всех seed обеспечена одним артефактом; "
            "fit векторизатора между seed не выполнялся.",
            # Честность monitor-кривых refit (train_monitor ⊂ outer_train):
            # это train-internal протокол выбора, а не независимая выборка.
            "Кривые refit: split 'monitor' — метрики на train_monitor (⊂ outer_train), "
            "т.е. train-internal протокол; независимая оценка — comparison_val после "
            "всех refit. Split 'train' — фактические метрики обучающей части.",
        ]
        if refit_scheduler_note:
            limitations.append(refit_scheduler_note)
        return {
            "schema_version": S05_SCHEMA_VERSION,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "protocol": dict(PROTOCOL),
            "config": cfg.to_dict(),
            "config_id": config_id,
            "fixed_epochs": fixed_epochs,
            "selection": (selection or {}).get("chosen"),
            "seeds": {"planned": list(cfg.seeds), "trained": trained,
                      "failed": [s for s in cfg.seeds if s not in trained],
                      "reference_seed": int(cfg.reference_seed)},
            "seed_stats_comparison_val": seed_stats,
            "inputs": {
                "splits_manifest": file_fingerprint(cfg.splits_manifest),
                "splits": {
                    name: {"sha256": part["sha256"], "bytes": part["bytes"], "n_rows": part["n_rows"]}
                    for name, part in manifest["parts"].items()
                },
                "labeled_parquet": manifest["inputs"]["labeled_parquet"],
                "full_tfidf": {
                    "path": os.path.abspath(cfg.full_tfidf_path),
                    "artifact_version": loaded.meta.get("artifact_version"),
                    "full_tfidf_id": full_tfidf_id,
                    "vocab": loaded.meta.get("vocab"),
                    "fit": loaded.meta.get("fit"),
                    # Проверка целостности выполняется вызывающей стороной до
                    # обучения (все CLI-пути); здесь фиксируется её результат,
                    # а не перепроверка задним числом с пропуском при отсутствии
                    # корпуса.
                    "files": (tfidf_verification or {}).get("files"),
                },
                "corpus": file_fingerprint(cfg.corpus_path) if os.path.isfile(cfg.corpus_path) else None,
            },
            "label_maps": {
                "n_classes": {lvl: maps.n_classes(lvl) for lvl in LEVELS},
                "l3_universe_sha256": _sha256_json(maps.to_dict()),
            },
            "environment": {
                "versions": environment_versions(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "device": str(device),
                "torch_cuda_available": torch.cuda.is_available(),
            },
            "git": {"commit": git_commit, "dirty": git_dirty},
            "statuses": {
                "refit": {f"seed_{s}": run_results[s]["status"] for s in cfg.seeds},
                "comparison_val_eval": {
                    f"seed_{s}": ("success" if s in eval_rows else "failed") for s in trained
                },
                "selected_checkpoint": selected_status,
            },
            "limitations": limitations,
        }

    trained = [s for s in cfg.seeds if run_results[s]["status"] == "trained"]
    if not trained:
        # Фиксируем провал до ошибки: все planned seed остаются видимыми в
        # runs.csv (status=failed, null-метрики), стадии — в stages.csv,
        # кривые selection — в history.csv; per_class.parquet и manifest.json
        # пишутся (пустыми/частичными), чтобы агрегированный отчёт D09
        # строился из сохранённых файлов. comparison_val не читался и не
        # оценивался (успешных моделей нет).
        pd.DataFrame(run_rows, columns=RUNS_COLUMNS).to_csv(os.path.join(out_dir, "runs.csv"), index=False)
        pd.DataFrame(history_rows, columns=HISTORY_COLUMNS).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        pd.DataFrame(stage_rows, columns=STAGES_COLUMNS).to_csv(os.path.join(out_dir, "stages.csv"), index=False)
        pd.DataFrame({
            "run_id": pd.Series(dtype="str"), "L3": pd.Series(dtype="str"),
            "support": pd.Series(dtype="int64"),
            "precision": pd.Series(dtype="float64"),
            "recall": pd.Series(dtype="float64"), "f1": pd.Series(dtype="float64"),
        }).to_parquet(os.path.join(out_dir, "per_class.parquet"), index=False)
        _write_json(build_manifest(None, None), os.path.join(out_dir, "manifest.json"))
        raise S05BaselineError(
            "Ни один seed не обучен; failed-запуски зафиксированы в runs.csv/stages.csv, "
            "comparison_val не читался: см. логи"
        )

    # --- Одна оценка raw top-1 на comparison_val ПОСЛЕ refit всех seed
    # (первое чтение/кодирование val-данных — только здесь, и только если
    # есть успешно обученные seed). Сбой общей подготовки validation (чтение,
    # кодирование меток, transform — напр. нехватка памяти) не должен
    # уничтожать агрегированную отчётность D09: failed-строки по всем
    # обученным seed, агрегаты и manifest записываются, контур завершается
    # ошибкой ниже.
    val_df: pd.DataFrame | None = None
    y_val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    X_val: sp.csr_matrix | None = None
    n_empty_val = 0
    try:
        val_df = load_part(manifest, "comparison_val")
        y_val = _encode_labels(maps, val_df, "comparison_val")
        with _Measurement(device, "transform comparison_val полным TF-IDF (первое обращение к val — после всех refit)") as m:
            X_val = _transform(loaded, val_df["name"].tolist(), cfg.features_chunk_rows)
        stage_rows.append(m.stage_row(SHARED_VECTORIZATION_RUN_ID, "vectorize_comparison_val", "success"))
        n_empty_val = _empty_rows(X_val)
        val_ok = True
    except Exception as e:  # noqa: BLE001 — диагностика сохраняется, контур падает ниже
        val_ok = False
        log.exception("comparison_val preparation failed")
        stage_rows.append({
            "run_id": SHARED_VECTORIZATION_RUN_ID, "stage": "vectorize_comparison_val",
            "status": "failed", "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
            "measurement_note": f"подготовка comparison_val прервана: {type(e).__name__}: {e}"[:500],
        })
        for seed in trained:
            rr = run_results[seed]
            reason = f"comparison_val preparation: {type(e).__name__}: {e}"[:500]
            run_rows.append({
                "run_id": f"seed_{seed}", "config_id": config_id, "train_seed": seed,
                "split_seed": split_seed, "status": "failed", "reason": reason,
                "n_train": int(len(train_df)), "n_monitor": int(len(mon_df)),
                "n_val": int(manifest["parts"]["comparison_val"]["n_rows"]), "n_classes": int(n_classes[2]),
                "full_tfidf_id": full_tfidf_id,
                "epochs_completed": len(rr["history"]), "best_epoch": int(rr["best"]["epoch"]),
                "l3_macro_f1": None, "l3_accuracy": None,
                "metric_evaluated_on": "comparison_val",
                "train_seconds": rr["train_seconds"], "peak_rss_bytes": rr["peak_rss_bytes"],
                "checkpoint_path": rr["checkpoint_dir"],
            })
            run_results[seed]["status"] = "failed"
            run_results[seed]["reason"] = reason

    for seed in [s for s in trained if run_results[s]["status"] == "trained"]:
        run_id = f"seed_{seed}"
        meas: _Measurement | None = None
        try:
            with _Measurement(device, f"seed {seed}: predict raw top-1 на comparison_val (n={X_val.shape[0]}), без отказа") as meas:
                art = load_artifact(run_results[seed]["checkpoint_dir"], device)
                preds, _ = predict(art.model, X_val, device, train_cfg.eval_batch_size, top_k=1)
                macro, acc, per_class = _l3_val_metrics(y_val[2], preds.l3_top1, l3_labels)
            stage_rows.append(meas.stage_row(run_id, "comparison_val_inference", "success"))
        except Exception as e:  # noqa: BLE001 — фиксируем failed seed, остальные оцениваются
            log.exception("seed %d: comparison_val evaluation failed", seed)
            if meas is not None and meas.seconds is not None:
                row = meas.stage_row(run_id, "comparison_val_inference", "failed")
                row["measurement_note"] += f" Этап прерван: {type(e).__name__}: {e}"
                stage_rows.append(row)
            else:
                stage_rows.append({
                    "run_id": run_id, "stage": "comparison_val_inference", "status": "failed",
                    "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
                    "measurement_note": f"оценка на comparison_val прервана: {type(e).__name__}: {e}"[:500],
                })
            reason = f"comparison_val evaluation: {type(e).__name__}: {e}"[:500]
            rr = run_results[seed]
            run_rows.append({
                "run_id": run_id, "config_id": config_id, "train_seed": seed,
                "split_seed": split_seed, "status": "failed", "reason": reason,
                "n_train": int(len(train_df)), "n_monitor": int(len(mon_df)),
                "n_val": int(len(val_df)), "n_classes": int(n_classes[2]),
                "full_tfidf_id": full_tfidf_id,
                "epochs_completed": len(rr["history"]),
                "best_epoch": int(rr["best"]["epoch"]),
                "l3_macro_f1": None, "l3_accuracy": None,
                "metric_evaluated_on": "comparison_val",
                "train_seconds": rr["train_seconds"],
                "peak_rss_bytes": rr["peak_rss_bytes"],
                "checkpoint_path": rr["checkpoint_dir"],
            })
            run_results[seed]["status"] = "failed"
            run_results[seed]["reason"] = reason
            continue
        frame = pd.DataFrame(per_class)[PER_CLASS_COLUMNS[1:]]
        frame.insert(0, "run_id", run_id)
        per_class_frames.append(frame)
        rr = run_results[seed]
        eval_rows[seed] = {
            "l3_macro_f1": macro, "l3_accuracy": acc,
            "val_inference_seconds": meas.seconds,
            "n_empty_features_val": n_empty_val,
            "per_class": per_class,
        }
        run_rows.append({
            "run_id": run_id, "config_id": config_id, "train_seed": seed,
            "split_seed": split_seed, "status": "success", "reason": None,
            "n_train": int(len(train_df)), "n_monitor": int(len(mon_df)),
            "n_val": int(len(val_df)), "n_classes": int(n_classes[2]),
            "full_tfidf_id": full_tfidf_id,
            "epochs_completed": len(rr["history"]),
            "best_epoch": int(rr["best"]["epoch"]),
            "l3_macro_f1": macro, "l3_accuracy": acc,
            "metric_evaluated_on": "comparison_val",
            "train_seconds": rr["train_seconds"],
            "peak_rss_bytes": rr["peak_rss_bytes"],
            "checkpoint_path": rr["checkpoint_dir"],
        })
        log.info("seed %d: comparison_val raw top-1 — acc=%.4f macro-F1=%.4f", seed, acc, macro)

    # --- Majority-class baseline (большинство по outer train) на том же comparison_val
    # (только если comparison_val подготовлен: иначе агрегаты пишутся без него,
    # gate честно получает not_ready по отсутствующему контролю).
    majority_report: dict[str, Any] | None = None
    if val_ok:
        counts_l3 = np.bincount(y_train[2], minlength=n_classes[2])
        majority_id = int(np.argmax(counts_l3))
        majority_label = l3_labels[majority_id]
        majority_pred = np.full(len(y_val[2]), majority_id, dtype=np.int64)
        maj_macro, maj_acc, maj_per_class = _l3_val_metrics(y_val[2], majority_pred, l3_labels)
        majority_report = {
            "run_id": "majority_outer_train",
            "rule": "наиболее частая L3 в outer_train (ties → наименьший id; train-only)",
            "majority_label": majority_label,
            "majority_id": majority_id,
            "majority_count_outer_train": int(counts_l3[majority_id]),
            "n_outer_train": int(len(train_df)),
            "evaluated_on": "comparison_val",
            "l3_accuracy": maj_acc,
            "l3_macro_f1": maj_macro,
            "n_empty_features_val": n_empty_val,
            "per_class": maj_per_class,
            "note": "тривиальный контроль; качество моделей сопоставляется с ним, а не с историческими ods_light метриками",
        }
        log.info("Majority (%s): comparison_val acc=%.4f macro-F1=%.4f", majority_label, maj_acc, maj_macro)

    # --- Опорный checkpoint для S06: правило reference_seed, round-trip проверка.
    # Сбой проверки (или отсутствие успешно оценённого reference seed) не должен
    # терять уже полученные результаты: selected_checkpoint.json фиксируется со
    # status=failed, все агрегаты пишутся, и только затем контур падает.
    evaluated = list(eval_rows)
    selection_error: Exception | None = None
    if not evaluated:
        selection_error = S05BaselineError(
            "ни один обученный seed не удалось оценить на comparison_val: опорный "
            "checkpoint для S06 подтвердить нельзя"
        )
        selected: dict[str, Any] = {
            "run_id": f"seed_{cfg.reference_seed}",
            "selection_rule": {
                "kind": "reference_seed", "value": int(cfg.reference_seed),
                "note": "заранее заданное детерминированное train-only правило; "
                        "качество на comparison_val в выборе не участвует",
            },
            "roundtrip": {"status": "failed",
                          "reason": f"{type(selection_error).__name__}: {selection_error}"},
            "expected_use": "опорная MLP для Gradient × Input (S06)",
        }
    else:
        try:
            selected = _select_and_verify_checkpoint(
                cfg, manifest, loaded, maps, ckpt_root, run_results, eval_rows,
                val_df["name"].tolist(), y_val, l3_labels, device, n_classes,
                fixed_epochs,
            )
        except Exception as e:  # noqa: BLE001 — фиксируем провал проверки, результаты сохраняем
            selection_error = e
            log.exception("Проверка опорного checkpoint не пройдена")
            selected = {
                "run_id": f"seed_{cfg.reference_seed}",
                "selection_rule": {
                    "kind": "reference_seed", "value": int(cfg.reference_seed),
                    "note": "заранее заданное детерминированное train-only правило; "
                            "качество на comparison_val в выборе не участвует",
                },
                "roundtrip": {"status": "failed",
                              "reason": f"{type(e).__name__}: {e}"[:1000]},
                "expected_use": "опорная MLP для Gradient × Input (S06)",
            }

    # --- Сводные агрегаты по seed (mean/std/range) — только по оценённым
    f1s = [eval_rows[s]["l3_macro_f1"] for s in evaluated]
    accs = [eval_rows[s]["l3_accuracy"] for s in evaluated]
    seed_stats = {
        "n_seed_success": len(evaluated),
        "n_seed_failed": len(cfg.seeds) - len(evaluated),
        "l3_macro_f1": _mean_std_range(f1s) if f1s else None,
        "l3_accuracy": _mean_std_range(accs) if accs else None,
    }

    # --- Записи результатов (все пути проверены на отсутствие до любых записей)
    for seed in trained:
        rr = run_results[seed]
        if seed not in eval_rows:
            _write_json({
                "run_id": rr["run_id"], "seed": seed, "status": "failed",
                "reason": rr.get("reason"),
                "config_id": config_id, "fixed_epochs": fixed_epochs,
                "checkpoint_dir": rr.get("checkpoint_dir"),
                "train_seconds": rr.get("train_seconds"),
                "peak_rss_bytes": rr.get("peak_rss_bytes"),
                "peak_vram_bytes": rr.get("peak_vram_bytes"),
            }, os.path.join(refit_dir, f"run_result_seed_{seed}.json"))
            continue
        ev = eval_rows[seed]
        _write_json({
            "run_id": rr["run_id"], "seed": seed, "status": "success",
            "config_id": config_id,
            "fixed_epochs": fixed_epochs,
            "selection": {"selection_seed": cfg.selection_seed, "criterion": chosen.get("criterion")},
            "train": {
                "epochs_completed": len(rr["history"]),
                "scheduler": train_cfg.scheduler,
                "scheduler_note": refit_scheduler_note,
                "best_epoch_monitor": int(rr["best"]["epoch"]),
                "early_stopping_metric": rr["best"]["metric"],
                "best_score_monitor": float(rr["best"]["score"]),
                "restored_best": bool(rr["best"].get("restored_best")),
                "final_monitor_l3_macro_f1": rr["history"][-1]["val_l3_macro_f1"] if rr["history"] else None,
                "final_monitor_l3_accuracy": rr["history"][-1]["val_l3_accuracy"] if rr["history"] else None,
                "final_train_l3_macro_f1": rr["history"][-1].get("train_l3_macro_f1") if rr["history"] else None,
                "final_train_l3_accuracy": rr["history"][-1].get("train_l3_accuracy") if rr["history"] else None,
                "n_train_rows_used": int(rr["best"].get("n_train_rows_used", 0)),
            },
            "train_seconds": rr["train_seconds"],
            "peak_rss_bytes": rr["peak_rss_bytes"],
            "peak_vram_bytes": rr["peak_vram_bytes"],
            "checkpoint_dir": rr["checkpoint_dir"],
            "comparison_val_eval": {
                "evaluated_after": "все refit seed завершены",
                "protocol": "raw top-1 L3 без отказа",
                "metric_evaluated_on": "comparison_val",
                "l3_macro_f1": ev["l3_macro_f1"],
                "l3_accuracy": ev["l3_accuracy"],
                "n_empty_features_val": ev["n_empty_features_val"],
                "val_inference_seconds": ev["val_inference_seconds"],
                "fixed_l3_universe": len(l3_labels),
            },
            "device": str(device),
        }, os.path.join(refit_dir, f"run_result_seed_{seed}.json"))

    pd.DataFrame(run_rows, columns=RUNS_COLUMNS).to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    if per_class_frames:
        pd.concat(per_class_frames, ignore_index=True).to_parquet(
            os.path.join(out_dir, "per_class.parquet"), index=False)
    else:
        pd.DataFrame({
            "run_id": pd.Series(dtype="str"), "L3": pd.Series(dtype="str"),
            "support": pd.Series(dtype="int64"),
            "precision": pd.Series(dtype="float64"),
            "recall": pd.Series(dtype="float64"), "f1": pd.Series(dtype="float64"),
        }).to_parquet(os.path.join(out_dir, "per_class.parquet"), index=False)
    pd.DataFrame(history_rows, columns=HISTORY_COLUMNS).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    pd.DataFrame(stage_rows, columns=STAGES_COLUMNS).to_csv(os.path.join(out_dir, "stages.csv"), index=False)
    if majority_report is not None:
        _write_json(majority_report, os.path.join(out_dir, "majority_baseline.json"))
    _write_json(selected, os.path.join(out_dir, "selected_checkpoint.json"))

    manifest_out = build_manifest(seed_stats, selected["roundtrip"]["status"])
    _write_json(manifest_out, os.path.join(out_dir, "manifest.json"))
    log.info("Refit контур завершён: %d/%d seed успешно (%d оценено), отчёты в %s",
             len(trained), len(cfg.seeds), len(evaluated), out_dir)
    if selection_error is not None:
        raise S05BaselineError(
            f"Опорный checkpoint для S06 не подтверждён (D08): {selection_error}. "
            "Завершившиеся результаты сохранены в runs.csv/per_class.parquet/history.csv/"
            "stages.csv/majority_baseline.json/manifest.json."
        ) from selection_error
    return manifest_out


def _mean_std_range(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
        "range": float(arr.max() - arr.min()), "n": int(len(arr)),
    }


def _select_and_verify_checkpoint(
    cfg: S05Config,
    manifest: dict[str, Any],
    loaded: LoadedVectorizer,
    maps: LabelMaps,
    ckpt_root: str,
    run_results: dict[int, dict[str, Any]],
    eval_rows: dict[int, dict[str, Any]],
    val_names: list[str],
    y_val: tuple[np.ndarray, np.ndarray, np.ndarray],
    l3_labels: list[str],
    device: torch.device,
    n_classes: tuple[int, int, int],
    fixed_epochs: int,
) -> dict[str, Any]:
    """D08: выбор опорного checkpoint заранее заданным правилом reference_seed.

    Round-trip (fail closed): артефакт загружается с диска через load_artifact
    и сверяется по полному контракту — seed, размерность, label maps, число
    эпох (равное зафиксированному fixed_epochs), restored_best=False,
    sha256 full TF-IDF, sha256 фактических train-сплитов и проверяемый
    отпечаток контракта признаков (словари char/word с порядком столбцов, IDF,
    параметры transform — :func:`feature_contract_fingerprint`): векторизатор с
    той же размерностью, но другим словарём/IDF отклоняется. Затем выполняется
    независимый inference **через transform векторизатора из checkpoint**
    (``art.features.transform`` на comparison names, без внешне построенной
    матрицы); метрики должны совпасть с сохранёнными (allclose). Правило выбора
    — train-only, качество на comparison_val в выборе не участвует.
    """
    seed = int(cfg.reference_seed)
    rr = run_results.get(seed)
    if rr is None or rr["status"] != "trained":
        raise S05BaselineError(
            f"reference_seed={seed} не обучен (status={rr and rr['status']}): опорный "
            "checkpoint заранее заданным правилом получить нельзя"
        )
    if seed not in eval_rows:
        raise S05BaselineError(
            f"reference_seed={seed} обучен, но не оценён на comparison_val: round-trip "
            "метрики сверить не с чем, опорный checkpoint не подтверждён"
        )
    ckpt_dir = os.path.join(ckpt_root, f"seed_{seed}")
    art = load_artifact(ckpt_dir, device)
    art_meta = art.meta or {}
    art_train = art_meta.get("train") or {}
    checks: dict[str, Any] = {}
    checks["seed_matches"] = bool(art_meta.get("seed") == seed)
    checks["n_features_matches"] = bool(art.model.n_features == loaded.builder.n_features)
    checks["n_classes_match"] = bool(tuple(art.model.n_classes) == tuple(n_classes))
    checks["label_maps_match"] = bool(art.maps.to_dict() == maps.to_dict())
    checks["protocol"] = art_meta.get("protocol")
    checks["epochs_completed"] = int(art_train.get("epochs_completed", -1))
    checks["fixed_epochs_matches"] = bool(checks["epochs_completed"] == int(fixed_epochs))
    checks["restored_best"] = bool(art_train.get("restored_best"))
    # Проверяемый отпечаток контракта признаков: словари (с порядком столбцов),
    # IDF и параметры transform. Подмена векторизатора на совместимый по
    # размерности меняет отпечаток и отклоняется.
    checks["feature_contract_matches"] = bool(
        feature_contract_fingerprint(art.features) == feature_contract_fingerprint(loaded.builder)
    )
    # Provenance источников: full TF-IDF и фактические train-сплиты обучения.
    art_tfidf = art_meta.get("full_tfidf") or {}
    checks["full_tfidf_id_matches"] = bool(
        art_tfidf.get("full_tfidf_id")
        == (loaded.meta.get("files") or {}).get("features.joblib", {}).get("sha256")
    )
    art_fp = art_meta.get("train_fingerprints") or {}
    checks["train_fingerprint_matches"] = bool(
        art_fp.get("outer_train_sha256") == manifest["parts"]["outer_train"]["sha256"]
        and art_fp.get("train_monitor_sha256") == manifest["parts"]["train_monitor"]["sha256"]
    )
    identity_checks = {
        k: checks[k]
        for k in (
            "seed_matches", "n_features_matches", "n_classes_match",
            "label_maps_match", "feature_contract_matches",
            "full_tfidf_id_matches", "train_fingerprint_matches",
            "fixed_epochs_matches",
        )
    }
    if not all(identity_checks.values()):
        raise S05BaselineError(f"Round-trip проверка provenance не пройдена: {checks}")
    if checks["restored_best"] is not False:
        raise S05BaselineError(
            f"Опорный checkpoint обязан быть финальным refit без выбора по monitor "
            f"(restored_best=False), получено {checks['restored_best']!r}"
        )

    # Независимый inference через transform векторизатора, сохранённого
    # в checkpoint: реальный путь инференса S06, без внешне построенной
    # матрицы признаков.
    X_roundtrip = art.features.transform(list(val_names), chunk_rows=cfg.features_chunk_rows)
    preds, _ = predict(art.model, X_roundtrip, device, 8192, top_k=1)
    macro, acc, _ = _l3_val_metrics(y_val[2], preds.l3_top1, l3_labels)
    saved = eval_rows[seed]
    metrics_match = bool(
        np.isclose(macro, saved["l3_macro_f1"], rtol=1e-6, atol=1e-8)
        and np.isclose(acc, saved["l3_accuracy"], rtol=1e-6, atol=1e-8)
    )
    result = {
        "run_id": f"seed_{seed}",
        "selection_rule": {
            "kind": "reference_seed",
            "value": seed,
            "note": "заранее заданное детерминированное train-only правило; "
                    "качество на comparison_val в выборе не участвует",
        },
        "checkpoint_dir": os.path.abspath(ckpt_dir),
        "checkpoint_files_sha256": {
            name: _sha256_file(os.path.join(ckpt_dir, name))
            for name in sorted(os.listdir(ckpt_dir))
        },
        "provenance_checks": checks,
        "roundtrip": {
            "status": "success" if metrics_match else "failed",
            "inference": "load_artifact → art.features.transform(comparison_val names) → predict "
                         "(векторизатор загруженного checkpoint, без внешней матрицы признаков)",
            "l3_macro_f1_recomputed": macro,
            "l3_accuracy_recomputed": acc,
            "l3_macro_f1_saved": saved["l3_macro_f1"],
            "l3_accuracy_saved": saved["l3_accuracy"],
            "metrics_match": metrics_match,
        },
        "expected_use": "опорная MLP для Gradient × Input (S06); G×I извлекается из этой модели",
    }
    if not metrics_match:
        raise S05BaselineError(f"Round-trip метрики не совпали: {result['roundtrip']}")
    return result


# ---------------------------------------------------------------------------
# Полный контур (verify → select → refit+eval)
# ---------------------------------------------------------------------------


def run_training_contour(cfg: S05Config) -> dict[str, Any]:
    """verify (D04) → selection (D05) → refit+eval+checkpoint (D06–D08).

    Проверенный provenance сохраняется через
    :func:`record_tfidf_verification` (только новый файл; при существующем —
    сверка совпадения).
    """
    cfg.validate()
    device = resolve_device(cfg.device)
    manifest = load_split_manifest(cfg.splits_manifest)
    verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)

    record_tfidf_verification(cfg.output_dir, verification)

    selection = run_selection(cfg, manifest, loaded, device)
    return run_refit(cfg, manifest, loaded, device, selection=selection,
                     tfidf_verification=verification)
