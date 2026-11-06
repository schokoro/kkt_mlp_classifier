"""Единый сравнительный эксперимент S03 и восстановимый агрегированный отчёт (D02, D08–D10).

Оркестрация сравнения стратегий supervised-отбора признаков при фиксированных
данных и модели: один раз готовятся общие splits и label maps, затем выполняются
плановые прогоны — контрольный полный TF-IDF (``none``) и
``(global | class_aware) × бюджеты K``. Для каждого прогона: отбор только по
размеченному train, повторный fit выбранных словарей на том же корпусе,
обучение MLP с общими гиперпараметрами и оценка **raw top-1 L3 без отказа на
validation**. Метки test в отборе и сравнении не участвуют (test-сплит
сохраняется только как provenance).

Результаты каждого прогона — ``outputs/SNN/runs/<run_id>/run_result.json``
плюс артефакты (отбора с полными рейтингами, ``tfidf_selected``, модели).
Агрегированный отчёт (``report.html``, ``runs.csv``, ``per_class.parquet``,
``stages.csv``, ``manifest.json``) строится функцией :func:`build_report` **по
сохранённым результатам** и восстанавливается без повторного обучения: HTML
генерируется из четырёх файлов отчёта, поэтому регенерация работает, даже если
остались только они. Неуспешные и не выполненные прогоны отражаются
статусами: неуспешный этап фиксируется в ``stages.csv`` со статусом
``failed`` (с измеренными до ошибки временем и пиком RSS), не достигнутые
этапы — со статусом ``skipped``, а не выдуманными числами.

Замеры ресурсов (D10): длительность и пик RSS каждого этапа (selection,
selected_tfidf_refit, vectorize, mlp_train, validation_inference). RSS
сэмплируется из ``/proc/self/status`` (VmRSS) внутри окна этапа; пики разных
этапов не складываются. В CPU-режиме ``peak_vram_bytes`` — null с пояснением.
Отсутствующие измерения — null, а не нули.

Сравнение прогонов (дельты, baseline-поля per_class, сравнительные графики)
допускается только при подтверждённой идентичности прогона и baseline.
Семантика подтверждения задаётся явной схемой версии отчёта: ``s04-identity@1``
требует все семь непустых идентификаторов у обеих сторон (validation, состав
L3, train, полный TF-IDF, конфигурация MLP, протокол оценки, seed; fail
closed — отсутствие поля у обеих сторон идентичность не доказывает),
``s03-identity@1`` — прежнее поведение отчётов S03 (совпадение val/labels).
Схема выводится из ``schema_version`` эксперимента/отчёта: маркер ``s03-*``
включает legacy-семантику, всё остальное — строгую S04. Регенерация HTML из
сохранённых таблиц использует только зафиксированные флаги сопоставимости;
при их отсутствии в S04-отчёте сравнение не считается подтверждённым.

Стоимость сценариев (раздел «Качество ↔ ресурсы») строится по **всем плановым**
прогонам (включая failed/skipped: неполный бюджет делает итог метода null, а не
суммой оставшихся успешных запусков). Собственные успешные этапы прогона плюс
два разделяемых one-shot-источника, каждый учтённый один раз на метод: общий
transform полного TF-IDF train/validation (``run_id=shared_vectorization``,
нужен всем методам, поэтому стоимость методов не зависит от порядка прогонов)
и, только для ``gradient_x_input``, разделяемая подготовка атрибуции
(``run_id=shared_attribution``). На none и Chi²-сценарии общие затраты
атрибуции не распределяются. Неизвестные времена — null, а не ноль: успешный
этап без замера, неуспешный бюджет или загруженный извне checkpoint (стоимость
его обучения не измерена) делают полный итог null, а не частичной суммой.

Smoke- и обычный режим — один и тот же код и CLI, различаются только
параметры конфигурации (бюджеты, данные, MLP).

Это инструментология PoC: результаты на ODS диагностические и не являются
оценкой промышленной применимости.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import html
import json
import logging
import math
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import precision_recall_fscore_support

from .artifact import environment_versions, load_artifact, save_artifact
from .attribution import (
    METHOD as GRADIENT_X_INPUT,
    ATTRIBUTION_PROVENANCE_FILENAME,
    AttributionConfig,
    AttributionError,
    build_gxi_ratings,
    checkpoint_fingerprint,
    compute_margin_attributions,
    feature_contract_fingerprint,
    save_attribution_report,
    select_attributed_rows,
    select_features_gradient_x_input,
    verify_attribution_checkpoint,
)
from .config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    RunConfig,
    TrainConfig,
    _merge,
    validate_config,
)
from .data import PreparedData, file_fingerprint, prepare_data, save_splits
from .features import matrix_stats
from .labels import LEVELS, LabelMaps
from .model import SparseMLP
from .reject import RejectRule
from .selection import (
    SELECTION_METHODS,
    SelectionConfig,
    save_selection_artifact,
    select_features,
)
from .train import compute_class_weights, predict, resolve_device, set_seed, train_model
from .vectorizer import LoadedVectorizer, load_vectorizer_artifact, refit_selected_vectorizer

log = logging.getLogger(__name__)

EXPERIMENT_SCHEMA_VERSION = "s04-experiment@1"
REPORT_SCHEMA_VERSION = "s04-report@1"
RUNS_DIRNAME = "runs"
REPORT_DIRNAME = "report"
SHARED_DIRNAME = "shared"
SHARED_STAGES_FILENAME = "attribution_stages.json"
SHARED_RUN_ID = "shared_attribution"
# Разделяемый transform полного TF-IDF train/validation: выполняется один раз
# при подготовке контекста и учитывается в stages.csv отдельным run_id, чтобы
# стоимость методов не зависела от порядка прогонов (ревью 2, находка 2).
SHARED_VECTORIZATION_RUN_ID = "shared_vectorization"
BASELINE_RUN_ID = "none"
# Методы плановых прогонов: Chi²-отбор S03 + Gradient × Input S04. Режим
# 'none' — контрольный прогон полного TF-IDF без отбора.
PLAN_METHODS = (*SELECTION_METHODS, GRADIENT_X_INPUT)

RUNS_COLUMNS = [
    "run_id", "status", "error", "method", "requested_k",
    "n_features", "n_char", "n_word",
    "n_train", "n_val", "n_classes_eval", "seed",
    "l3_macro_f1", "l3_accuracy", "delta_macro_f1", "delta_accuracy",
    "train_seconds", "val_inference_seconds", "model_size_bytes",
    "vectorizer_artifact", "model_artifact", "selection_artifact",
]
PER_CLASS_COLUMNS = [
    "run_id", "L3", "support", "precision", "recall", "f1",
    "baseline_f1", "delta_f1", "baseline_recall", "delta_recall",
]
STAGES_COLUMNS = [
    "run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
    "measurement_note",
]

RSS_SAMPLING_NOTE = (
    "RSS: VmRSS из /proc/self/status, сэмпл каждые ~20 мс внутри окна этапа; "
    "короткие этапы могут пропустить истинный пик; пики разных этапов не аддитивны."
)
VRAM_NULL_NOTE = "CPU-режим: VRAM не использовалась."
RSS_UNAVAILABLE_NOTE = (
    "RSS не измерен: чтение VmRSS из /proc/self/status было недоступно; "
    "peak_rss_bytes=null."
)
RSS_PARTIAL_NOTE = (
    "часть сэмплов RSS была недоступна; пик — максимум по удавшимся сэмплам "
    "и может быть занижен."
)

METRIC_PROTOCOL = {
    "metric": "raw top-1 L3 без отказа на validation",
    "macro_f1": "среднее F1 по фиксированному набору L3 label_map с support>0 на общей validation",
    "accuracy": "доля верных top-1 среди всех строк validation (отказ не применяется)",
    "zero_division": 0,
    "support_0": "precision/recall/f1 и дельты = null",
    "reject": "τ и правила отказа в эксперименте не калибруются и не используются",
    "test_labels": "метки test не участвуют в selection и сравнении",
}

LIMITATION_NOTES = [
    "ODS/PoC не является оценкой промышленной применимости; результаты диагностические.",
    "Эксперимент выполняется в одном процессе: замеры RAM/времени отражают локальную машину "
    "и зависят от параллелизма sklearn/torch; повторные прогоны могут давать иные абсолютные значения.",
    "Режим features.mode=pretrained трансдуктивен: тексты корпуса могут совпадать с train/val/test.",
]


class ExperimentError(RuntimeError):
    """Базовая ошибка экспериментальной оркестрации."""


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


@dataclass
class SelectionPlanConfig:
    """План отбора: методы, бюджеты, квота char/word.

    ``include_full`` добавляет контрольный прогон ``none`` (полный TF-IDF без
    отбора). Бюджеты и методы не зашиты в алгоритм: задаются конфигурацией.
    """

    methods: list[str] = field(default_factory=lambda: ["global", "class_aware"])
    budgets: list[int] = field(default_factory=lambda: [50_000, 100_000, 200_000])
    char_share: float = 0.75
    include_full: bool = True

    def validate(self) -> None:
        bad = [m for m in self.methods if m not in PLAN_METHODS]
        if bad:
            raise ExperimentError(f"selection.methods: одно из {PLAN_METHODS}, получено {bad}")
        if not self.budgets:
            raise ExperimentError("selection.budgets: нужен хотя бы один бюджет K")
        for k in self.budgets:
            if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
                raise ExperimentError(f"selection.budgets: целые > 0, получено {k!r}")
        if not 0.0 < self.char_share < 1.0:
            raise ExperimentError(
                f"selection.char_share: доля строго в (0, 1), получено {self.char_share!r}; "
                "граничные значения обнуляют одну из веток (char/word), а пустую ветку "
                "невозможно корректно переобучить — задайте обеим веткам ненулевую квоту"
            )


@dataclass
class ExperimentConfig:
    """Конфигурация единого эксперимента (D02).

    Источники: автономный артефакт полного TF-IDF (``full_tfidf_path``), тот же
    корпус, что в его метаданных (``corpus_path``, для повторного fit), размеченный
    Parquet (``data``). Сплиты, label maps, seed и параметры MLP общие для всех
    плановых прогонов. Выходы: ``output_dir/runs/<run_id>`` и
    ``output_dir/report``.
    """

    output_dir: str = "outputs/S03"
    seed: int = 42
    full_tfidf_path: str = ""
    corpus_path: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    features_chunk_rows: int = 50_000
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selection: SelectionPlanConfig = field(default_factory=SelectionPlanConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    noisy_support: int = 2

    def validate(self) -> None:
        if not self.full_tfidf_path:
            raise ExperimentError("full_tfidf_path: обязателен")
        if not self.corpus_path:
            raise ExperimentError("corpus_path: обязателен (повторный fit выбранных словарей)")
        if not self.data.path:
            raise ExperimentError("data.path: обязателен")
        if self.features_chunk_rows <= 0:
            raise ExperimentError("features_chunk_rows: > 0")
        if self.noisy_support < 0:
            raise ExperimentError("noisy_support: >= 0")
        self.selection.validate()
        try:
            self.attribution.validate()
        except AttributionError as e:
            raise ExperimentError(str(e)) from e
        # Валидация вложенных разделов через стандартный путь конфига прогона.
        validate_config(RunConfig(
            output_dir=self.output_dir, seed=self.seed, data=self.data,
            model=self.model, loss=self.loss, train=self.train,
        ))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return _merge(cls(), d)


@dataclass(frozen=True)
class PlannedRun:
    """Один плановый прогон: стабильный run_id, метод и бюджет."""

    run_id: str
    method: str          # none | global | class_aware
    requested_k: int | None


def planned_runs(cfg: ExperimentConfig) -> list[PlannedRun]:
    """7 (или меньше) плановых прогонов в детерминированном порядке: сначала
    контрольный ``none``, затем methods × budgets в порядке конфигурации."""
    out: list[PlannedRun] = []
    if cfg.selection.include_full:
        out.append(PlannedRun(BASELINE_RUN_ID, "none", None))
    for m in cfg.selection.methods:
        for k in cfg.selection.budgets:
            out.append(PlannedRun(f"{m}_{k}", m, int(k)))
    return out


# ---------------------------------------------------------------------------
# Замеры ресурсов
# ---------------------------------------------------------------------------


def _rss_bytes() -> int | None:
    """VmRSS текущего процесса в байтах; None, если /proc недоступен.

    Недоступный замер не заменяется нулём: отсутствие измерения и нулевой
    расход RAM — разные состояния, смешивать их нельзя."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class _PeakRssSampler:
    """Сэмплирует VmRSS внутри окна измерения; пик — максимум сэмплов.

    Недоступный сэмпл (None) не превращается в ноль: он учитывается в
    ``failures``. ``peak`` остаётся None, только если ни один сэмпл не удался;
    при частичной недоступности пик — максимум по удавшимся сэмплам, и
    ``_Measurement`` обязан пояснить это в ``measurement_note``."""

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.peak: int | None = None
        self.failures = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        v = _rss_bytes()
        if v is None:
            self.failures += 1
            return
        self.peak = v if self.peak is None else max(self.peak, v)

    def __enter__(self) -> "_PeakRssSampler":
        self.peak = None
        self.failures = 0
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __exit__(self, *exc: Any) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()
        return False


class _Measurement:
    """Контекст замера одного этапа: длительность, пик RSS, пик VRAM (CUDA)."""

    def __init__(self, device: torch.device, note: str, rss_interval_s: float = 0.02):
        self.device = device
        self.note = note
        self.rss_interval_s = rss_interval_s
        self.seconds: float | None = None
        self.peak_rss_bytes: int | None = None
        self.peak_vram_bytes: int | None = None

    def __enter__(self) -> "_Measurement":
        self._t0 = time.perf_counter()
        self._vram_before: int | None = None
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._vram_before = torch.cuda.memory_allocated()
        self._rss = _PeakRssSampler(self.rss_interval_s)
        self._rss.__enter__()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._rss.__exit__()
        self.seconds = time.perf_counter() - self._t0
        # Полная недоступность RSS -> None (null), не ноль. При частичной
        # недоступности пик сохраняется по удавшимся сэмплам, но с пояснением:
        # пропущенные сэмплы могли зафиксировать больший пик.
        self.peak_rss_bytes = self._rss.peak
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.peak_vram_bytes = max(
                0, int(torch.cuda.max_memory_allocated()) - int(self._vram_before or 0)
            )
        return False

    def stage_row(self, run_id: str, stage: str, status: str) -> dict[str, Any]:
        notes = [self.note]
        if self.peak_rss_bytes is None:
            notes.append(RSS_UNAVAILABLE_NOTE)
        elif self._rss.failures:
            notes.append(RSS_PARTIAL_NOTE)
        else:
            notes.append(RSS_SAMPLING_NOTE)
        if self.device.type != "cuda":
            notes.append(VRAM_NULL_NOTE)
        return {
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "seconds": self.seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "measurement_note": " ".join(notes),
        }


def _skipped_stage(run_id: str, stage: str, reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id, "stage": stage, "status": "skipped",
        "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
        "measurement_note": reason,
    }


# Канонический порядок этапов прогона: используется для фиксации не
# выполненных (skipped) этапов при сбое/пропуске прогона и при построении
# stages.csv по сохранённым результатам.
STAGES_SELECTED = ("selection", "selected_tfidf_refit", "vectorize", "mlp_train", "validation_inference")
STAGES_NONE = ("selection", "selected_tfidf_refit", "full_tfidf_fit", "vectorize",
               "mlp_train", "validation_inference")


def _canonical_stage_names(method: str) -> tuple[str, ...]:
    return STAGES_NONE if method == "none" else STAGES_SELECTED


@contextmanager
def _stage(device: torch.device, note: str, run_result: dict, stage: str):
    """Замер одного этапа с гарантированной строкой в ``stages.csv``: при
    успехе — статус ``success``, при ошибке — ``failed`` с измеренными до
    прерывания длительностью и пиком RSS и пояснением; исключение пробрасывается
    дальше, чтобы прогон получил статус ``failed`` с причиной."""
    run_id = run_result.get("run_id") or "run"
    m = _Measurement(device, note)
    try:
        with m:
            yield m
    except Exception as e:
        # Замер уже завершён: фиксируем затраты до прерывания.
        row = m.stage_row(run_id, stage, "failed")
        row["measurement_note"] += f" Этап прерван: {type(e).__name__}: {e}"
        run_result["stages"].append(row)
        raise
    run_result["stages"].append(m.stage_row(run_id, stage, "success"))


def _append_missing_stages(run_result: dict, method: str, reason: str) -> None:
    """Дописывает ``skipped``-строки для этапов, которые не зафиксированы
    в ``run_result['stages']`` (прогон прерван раньше или не выполнялся)."""
    reached = {row.get("stage") for row in run_result.get("stages") or []}
    for name in _canonical_stage_names(method):
        if name not in reached:
            run_result["stages"].append(_skipped_stage(run_result["run_id"], name, reason))


# ---------------------------------------------------------------------------
# Метрики raw top-1 L3 на validation
# ---------------------------------------------------------------------------


def _l3_val_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, l3_labels: list[str]
) -> tuple[float, float, list[dict[str, Any]]]:
    """Raw top-1 L3 без отказа: macro-F1 по support>0, accuracy, per-class.

    Per-class считается по фиксированному набору L3 (все классы label_map,
    одинаковые для каждого прогона); ``zero_division=0``; для ``support=0``
    метрики и будущие дельты — null.
    """
    labels = np.arange(len(l3_labels), dtype=np.int64)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class: list[dict[str, Any]] = []
    for i, name in enumerate(l3_labels):
        row: dict[str, Any] = {"L3": name, "support": int(sup[i])}
        if sup[i] > 0:
            row.update(precision=float(p[i]), recall=float(r[i]), f1=float(f1[i]))
        else:
            row.update(precision=None, recall=None, f1=None)
        per_class.append(row)
    mask = sup > 0
    macro = float(f1[mask].mean()) if mask.any() else float("nan")
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean()) if len(y_true) else float("nan")
    return macro, acc, per_class


# ---------------------------------------------------------------------------
# JSON-утилиты и окружение
# ---------------------------------------------------------------------------


def _json_default(o: Any) -> Any:
    if o is pd.NA:
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _write_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_json(payload: Any) -> str:
    """Детерминированный sha256 канонического JSON-представления объекта."""
    import hashlib

    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _model_config_fingerprint(cfg: "ExperimentConfig") -> str:
    """Отпечаток конфигурации обучения MLP (model/loss/train).

    Два прогона сопоставимы только при одинаковой архитектуре, функции потерь
    и параметрах обучения финальной MLP (seed сравнивается отдельным полем).
    """
    return _sha256_json({
        "model": dataclasses.asdict(cfg.model),
        "loss": dataclasses.asdict(cfg.loss),
        "train": dataclasses.asdict(cfg.train),
    })


def _full_tfidf_sha256(path: str) -> str | None:
    """Отпечаток артефакта полного TF-IDF: sha256 файлов артефакта.

    Фиксирует фактическое содержимое признакового контракта, а не путь: замена
    артефакта по тому же пути изменяет отпечаток. None, если файлов нет.
    """
    import hashlib

    digest = hashlib.sha256()
    found = False
    for name in ("features.joblib", "vectorizer_meta.json"):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            digest.update(_sha256_file(fp).encode("ascii"))
            digest.update(b"\x00")
            found = True
    return digest.hexdigest() if found else None


def _git_info() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        ).stdout.strip())
        return commit, dirty
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Контекст эксперимента: общие данные, признаки, матрицы
# ---------------------------------------------------------------------------


class _ExperimentContext:
    def __init__(self, cfg: ExperimentConfig, loaded: LoadedVectorizer, data: PreparedData,
                 device: torch.device):
        self.cfg = cfg
        self.loaded = loaded
        self.full_builder = loaded.builder
        self.full_meta = loaded.meta
        self.data = data
        self.maps = data.maps
        self.device = device
        self.out = cfg.output_dir
        self.y_train = tuple(self.maps.encode(lvl, data.splits["train"][lvl].to_numpy()) for lvl in LEVELS)
        self.y_val = tuple(self.maps.encode(lvl, data.splits["val"][lvl].to_numpy()) for lvl in LEVELS)
        self.class_weights = {
            h: compute_class_weights(
                self.y_train[i], self.maps.n_classes(lvl),
                cfg.loss.class_weight_mode[h], cfg.loss.class_weight_alpha,
            )
            for i, (h, lvl) in enumerate(zip(("l1", "l2", "l3"), LEVELS))
        }
        self._X_full_train: sp.csr_matrix | None = None
        self._X_full_val: sp.csr_matrix | None = None
        # S04: разделяемые между gradient_x_input_* результаты атрибуции
        # (модель, рейтинги, диагностика) либо зафиксированная ошибка подготовки.
        self.attribution: AttributionBundle | None = None
        self.attribution_error: str | None = None
        self.attribution_meta: dict[str, Any] | None = None
        self.shared_stage_rows: list[dict[str, Any]] = []

    @property
    def n_char_full(self) -> int:
        return int(self.full_builder.n_char_features)

    @property
    def l3_labels(self) -> list[str]:
        return list(self.maps.l3)

    def ensure_full_train(self) -> tuple[sp.csr_matrix, bool]:
        """Матрица train полного TF-IDF; флаг — выполнен ли transform сейчас."""
        if self._X_full_train is None:
            self._X_full_train = self.full_builder.transform(
                self.data.splits["train"]["name"].tolist(), chunk_rows=self.cfg.features_chunk_rows
            )
            return self._X_full_train, True
        return self._X_full_train, False

    def ensure_full_val(self) -> tuple[sp.csr_matrix, bool]:
        if self._X_full_val is None:
            self._X_full_val = self.full_builder.transform(
                self.data.splits["val"]["name"].tolist(), chunk_rows=self.cfg.features_chunk_rows
            )
            return self._X_full_val, True
        return self._X_full_val, False


# ---------------------------------------------------------------------------
# Gradient × Input: разделяемая attribution MLP, атрибуция и рейтинги (S04)
# ---------------------------------------------------------------------------


@dataclass
class AttributionBundle:
    """Разделяемые между прогонами ``gradient_x_input_*`` результаты подготовки.

    Модель атрибуции (checkpoint на полном TF-IDF), каталог полных G×I-рейтингов
    (S03-формат), сводная диагностика и проверенные идентификаторы источников.
    Стоимость подготовки измеряется один раз и фиксируется отдельными строками
    ``stages.csv`` с ``run_id=shared_attribution`` (без двойного счёта между
    бюджетами).
    """

    model: SparseMLP
    checkpoint_dir: str
    checkpoint: dict[str, Any]
    ratings_dir: str
    ratings_manifest: dict
    report_path: str
    report: dict
    sampling: dict[str, Any]
    verification: dict[str, Any]

    def run_link(self) -> dict[str, Any]:
        """Компактная аудируемая ссылка прогона на разделяемую атрибуцию."""
        return {
            "ratings_dir": os.path.abspath(self.ratings_dir),
            "report_path": os.path.abspath(self.report_path),
            "checkpoint_dir": os.path.abspath(self.checkpoint_dir),
            "checkpoint": self.checkpoint,
            "sampling": dict(self.sampling),
            "verification": dict(self.verification),
        }


def _shared_measurement(
    ctx: _ExperimentContext, stage: str, note: str, fn: Any,
    run_id: str = SHARED_RUN_ID,
) -> Any:
    """Замер разделяемого этапа подготовки: строка stages.csv с заданным
    run_id (по умолчанию shared_attribution); ошибка пробрасывается после
    фиксации строки."""
    m = _Measurement(ctx.device, note)
    try:
        with m:
            result = fn()
    except Exception as e:
        row = m.stage_row(run_id, stage, "failed")
        row["measurement_note"] += f" Этап прерван: {type(e).__name__}: {e}"
        ctx.shared_stage_rows.append(row)
        raise
    ctx.shared_stage_rows.append(m.stage_row(run_id, stage, "success"))
    return result


def _reject_info_stub() -> dict[str, Any]:
    return {
        "note": "S04: τ не калибровался; сравнение по raw top-1 на validation",
        "tau_mode": "fixed",
    }


def _train_attribution_model(ctx: _ExperimentContext, out_dir: str) -> dict[str, Any]:
    """Обучение attribution MLP на полном TF-IDF (train, early stopping по
    validation — метрики валидации не используются для самого отбора)."""
    cfg = ctx.cfg
    checkpoint_dir = os.path.join(out_dir, "attribution_model")

    def work() -> dict[str, Any]:
        X_train, _ = ctx.ensure_full_train()
        X_val, _ = ctx.ensure_full_val()
        set_seed(cfg.seed)
        model = SparseMLP(
            ctx.full_builder.n_features,
            tuple(ctx.maps.n_classes(lvl) for lvl in LEVELS),
            cfg.model,
        )
        history, best, _trace = train_model(
            model, X_train, ctx.y_train, X_val, ctx.y_val,
            cfg.loss, cfg.train, ctx.class_weights, ctx.device, cfg.seed,
        )
        train_counts = {
            lvl: ctx.data.splits["train"][lvl].value_counts().sort_index().astype(int).to_dict()
            for lvl in LEVELS
        }
        save_artifact(
            checkpoint_dir, model, cfg.model, ctx.full_builder, ctx.maps,
            RejectRule(0.5, False, False), _reject_info_stub(),
            {
                "mode": cfg.loss.class_weight_mode,
                "alpha": cfg.loss.class_weight_alpha,
                "weights": {h: (None if w is None else w.tolist()) for h, w in ctx.class_weights.items()},
                "train_counts": train_counts,
            },
            {
                "experiment_run": "attribution",
                "role": "gradient_x_input_attribution",
                "config": cfg.to_dict(),
            },
            {
                "seed": cfg.seed,
                "n_features": int(ctx.full_builder.n_features),
                "device": str(ctx.device),
                "train": {
                    "epochs_run": len(history),
                    "best_epoch": int(best["epoch"]),
                    "early_stopping_metric": best["metric"],
                    "best_score": float(best["score"]),
                    "final_lr": float(best["final_lr"]),
                    "n_train_rows_used": int(best["n_train_rows_used"]),
                },
            },
        )
        # Проверяемый provenance checkpoint'а (S04, ревью 1): отпечаток полного
        # контракта признаков (словари char/word + IDF и все параметры
        # преобразования TF-IDF) и фактические сплиты, на которых обучена
        # модель, плюс sha256 файла весов model.pt: загрузка checkpoint'а в
        # другом эксперименте сверяет фактические значения с сохранёнными, а
        # подмена совместимого по размерности model.pt меняет хеш и
        # отклоняется (ревью 2, находка 1). Без этого файла загрузка
        # отклоняется (fail closed), см.
        # kkt_clf.attribution.verify_attribution_checkpoint.
        _write_json(
            {
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "seed": int(cfg.seed),
                "n_features": int(ctx.full_builder.n_features),
                "full_tfidf_fingerprint": feature_contract_fingerprint(ctx.full_builder),
                "train_split_sha256": (ctx.split_fingerprints.get("train") or {}).get("sha256"),
                "val_split_sha256": (ctx.split_fingerprints.get("val") or {}).get("sha256"),
                "label_map_sha256": ctx.label_map_sha256,
                "model_config_sha256": _model_config_fingerprint(cfg),
                "model_weights_sha256": _sha256_file(os.path.join(checkpoint_dir, "model.pt")),
                "data_config": json.loads(
                    json.dumps(dataclasses.asdict(cfg.data), ensure_ascii=False, default=str)
                ),
            },
            os.path.join(checkpoint_dir, ATTRIBUTION_PROVENANCE_FILENAME),
        )
        last = history[-1] if history else {}
        return {
            "model": model,
            "checkpoint_dir": checkpoint_dir,
            "train_info": {
                "epochs_run": len(history),
                "best_epoch": int(best["epoch"]),
                "early_stopping_metric": best["metric"],
                "best_score": float(best["score"]),
                "final_lr": float(best["final_lr"]),
                "last_epoch_val_l3_macro_f1": last.get("val_l3_macro_f1"),
                "last_epoch_val_l3_accuracy": last.get("val_l3_accuracy"),
            },
        }

    result = _shared_measurement(
        ctx, "attribution_model_train",
        "обучение attribution MLP на полном TF-IDF (train, early stopping по "
        "validation в рамках существующего протокола); разделяемый этап S04",
        work,
    )
    result["fingerprint"] = checkpoint_fingerprint(checkpoint_dir)
    return result


def _load_attribution_model(ctx: _ExperimentContext) -> dict[str, Any]:
    """Загрузка проверенного checkpoint'а attribution MLP из конфигурации."""
    cfg = ctx.cfg
    art_dir = cfg.attribution.checkpoint_path

    def work() -> dict[str, Any]:
        art = load_artifact(art_dir, ctx.device)
        data_cfg = json.loads(
            json.dumps(dataclasses.asdict(cfg.data), ensure_ascii=False, default=str)
        )
        verification = verify_attribution_checkpoint(
            art_dir, art.model, art.config,
            n_features_full=ctx.full_builder.n_features,
            label_map_sha256=ctx.label_map_sha256,
            data_cfg=data_cfg,
            seed=cfg.seed,
            full_tfidf_fp=feature_contract_fingerprint(ctx.full_builder),
            train_split_sha256=(ctx.split_fingerprints.get("train") or {}).get("sha256"),
        )
        return {
            "model": art.model,
            "checkpoint_dir": os.path.abspath(art_dir),
            "train_info": (art.meta or {}).get("train"),
            "verification": verification,
            "artifact_config": art.config,
        }

    result = _shared_measurement(
        ctx, "attribution_model_load",
        "загрузка checkpoint'а attribution MLP с проверкой идентичности: размерность "
        "полного TF-IDF, label maps, data-конфиг/seed (splits), проверяемый fingerprint "
        "полного контракта признаков (словари char/word + IDF и параметры преобразования), "
        "sha256 весов model.pt и sha256 фактического train-сплита из "
        "attribution_provenance.json (fail closed)",
        work,
    )
    # Стоимость обучения загруженного извне checkpoint'а в этом процессе не
    # измерена: фиксируем attribution_model_train с seconds=None, чтобы итог
    # стоимости G×I честно оставался неизвестным (null), а не полным по
    # собственным затратам прогона (ревью 2, находка 2). Строка одна — дублей
    # при повторных вызовах не появляется.
    ctx.shared_stage_rows = [
        row for row in ctx.shared_stage_rows
        if not (row.get("stage") == "attribution_model_train" and row.get("run_id") == SHARED_RUN_ID)
    ]
    ctx.shared_stage_rows.append({
        "run_id": SHARED_RUN_ID,
        "stage": "attribution_model_train",
        "status": "success",
        "seconds": None,
        "peak_rss_bytes": None,
        "peak_vram_bytes": None,
        "measurement_note": (
            "checkpoint attribution MLP загружен извне: обучение в этом процессе "
            "не выполнялось, стоимость обучения неизвестна (null); полная стоимость "
            "сценария gradient_x_input без замера обучения не заявляется"
        ),
    })
    result["fingerprint"] = checkpoint_fingerprint(art_dir)
    return result


def _prepare_attribution(ctx: _ExperimentContext) -> None:
    """Подготовка разделяемой атрибуции для прогонов ``gradient_x_input_*``.

    Два сценария модели (S04): загрузить проверенный checkpoint из
    ``attribution.checkpoint_path`` или обучить из конфигурации эксперимента.
    Затем Gradient × Input по margin на (возможно, сэмплированном) размеченном
    train и полные рейтинги на диск. Результаты — в ``output_dir/shared/``;
    строки этапов — в ``ctx.shared_stage_rows`` (run_id=shared_attribution).
    При ошибке фиксируется ``ctx.attribution_error``: прогоны G×I получат
    статус failed с этой причиной, остальные методы не затрагиваются.
    """
    cfg = ctx.cfg
    out_dir = os.path.join(ctx.out, SHARED_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    # Строки разделяемых этапов накапливаются с подготовки контекста (общий
    # transform полного TF-IDF, run_id=shared_vectorization) — не обнуляем.
    model = None
    checkpoint: dict[str, Any] = {}
    train_info = None
    verification: dict[str, Any] = {}
    try:
        if cfg.attribution.checkpoint_path:
            loaded = _load_attribution_model(ctx)
            model = loaded["model"]
            checkpoint = {
                "path": loaded["checkpoint_dir"],
                "source": "loaded",
                **loaded["fingerprint"],
            }
            train_info = loaded.get("train_info")
            verification = loaded.get("verification") or {}
        else:
            trained = _train_attribution_model(ctx, out_dir)
            model = trained["model"]
            checkpoint = {
                "path": os.path.abspath(trained["checkpoint_dir"]),
                "source": "trained_in_experiment",
                **trained["fingerprint"],
            }
            train_info = trained["train_info"]
        ctx.attribution_error = None
    except Exception as e:  # noqa: BLE001 — фиксируем причину; прогоны G×I упадут осмысленно
        ctx.attribution_error = f"{type(e).__name__}: {e}"[:1000]
        log.exception("attribution model: failed")
    if model is None:
        ctx.attribution_meta = {"error": ctx.attribution_error}
        _write_json(
            {"stages": ctx.shared_stage_rows, "error": ctx.attribution_error},
            os.path.join(out_dir, SHARED_STAGES_FILENAME),
        )
        return

    ratings_dir = os.path.join(out_dir, "attribution", "ratings")
    report_path = os.path.join(out_dir, "attribution", "report.json")

    def attribute() -> dict[str, Any]:
        X_full, built = ctx.ensure_full_train()
        n_train = int(X_full.shape[0])
        rows = select_attributed_rows(n_train, cfg.attribution, cfg.seed)
        X_att = X_full[rows]
        y_att = ctx.y_train[2][rows]
        att = compute_margin_attributions(
            model, X_att, y_att, ctx.device, batch_size=cfg.attribution.batch_size
        )
        manifest, report = build_gxi_ratings(att, ctx.n_char_full, ratings_dir, ctx.l3_labels)
        report["sampling"] = {
            "method": cfg.attribution.sampling,
            "max_rows": cfg.attribution.max_rows,
            "seed": int(cfg.seed),
            "n_train_rows_total": n_train,
            "n_rows_used": int(len(rows)),
            "resampled_transform": bool(built),
        }
        report["checkpoint"] = dict(checkpoint)
        report["train_info"] = train_info
        save_attribution_report(report_path, report)
        return {"manifest": manifest, "report": report, "rows": rows}

    try:
        done = _shared_measurement(
            ctx, "attribution",
            "Gradient × Input по margin истинной L3 против сильнейшего конкурента "
            "на train-only (sparse CSR value gradients, model.eval(), без обновления "
            "весов); агрегация class-aware рейтингов на диск; метки val/test не используются",
            attribute,
        )
    except Exception as e:  # noqa: BLE001
        ctx.attribution_error = f"{type(e).__name__}: {e}"[:1000]
        log.exception("attribution: failed")
        ctx.attribution_meta = {"error": ctx.attribution_error}
        _write_json(
            {"stages": ctx.shared_stage_rows, "error": ctx.attribution_error},
            os.path.join(out_dir, SHARED_STAGES_FILENAME),
        )
        return
    sampling = dict(done["report"]["sampling"])
    ctx.attribution = AttributionBundle(
        model=model,
        checkpoint_dir=checkpoint["path"],
        checkpoint=checkpoint,
        ratings_dir=ratings_dir,
        ratings_manifest=done["manifest"],
        report_path=report_path,
        report=done["report"],
        sampling=sampling,
        verification=verification,
    )
    full_fp = ctx.split_fingerprints.get("train", {}) if hasattr(ctx, "split_fingerprints") else {}
    ctx.attribution_meta = {
        "target": done["report"]["target"],
        "formula": done["report"]["formula"],
        "mode": (
            "model.eval(); batch gradient по ненулевым CSR-значениям; веса не "
            "обновлялись; competitor фиксировался до градиентного шага"
        ),
        "batch_size": int(cfg.attribution.batch_size),
        "checkpoint": checkpoint,
        "train_info": train_info,
        "verification": verification,
        "sampling": sampling,
        "positive_filter": "rankings: max(0, a) > 0; negative/zero — только диагностика",
        "empty_policy": (
            "пустые TF-IDF-строки: margin вычисляется через bias-путь, вкладов нет; "
            "в N_c знаменателя класса входят, поэтому класс только из пустых строк "
            "сохраняется с пустым списком положительных кандидатов; их число — в диагностике"
        ),
        "ties_policy": (
            "равные логиты конкурента → наименьший индекс класса; равные оценки "
            "признака → наименьший индекс признака"
        ),
        "early_stopping_note": (
            "attribution MLP обучалась с обычным early stopping по validation "
            "(метрика train.*); метки validation/test в ранжирование не входят"
        ),
        "full_tfidf": {
            "path": os.path.abspath(cfg.full_tfidf_path),
            "artifact_version": ctx.full_meta.get("artifact_version"),
            "source_sha256": (ctx.full_meta.get("source") or {}).get("sha256"),
        },
        "train_split": {
            "sha256": (full_fp or {}).get("sha256"),
            "label_map_sha256": getattr(ctx, "label_map_sha256", None),
            "rows": sampling.get("n_train_rows_total"),
        },
        "report_path": os.path.abspath(report_path),
        "ratings_manifest": done["manifest"],
        "summary": {
            "n_attributed_rows": done["report"]["n_attributed_rows"],
            "n_nonempty_rows": done["report"]["n_nonempty_rows"],
            "n_empty_rows": done["report"]["n_empty_rows"],
            "n_classes": done["report"]["n_classes"],
            "n_classes_with_positive": done["report"]["n_classes_with_positive"],
            "correct": done["report"]["correct"],
            "incorrect": done["report"]["incorrect"],
            "margins": done["report"]["margins"],
            "contributions": done["report"]["contributions"],
        },
        # Число фактически использованных train-строк по каждой L3 — прямо в
        # метаданных (manifest.json), а не только во внешнем report.json: без
        # пути к файлу значения остаются проверяемой частью контракта отчёта.
        "per_class_counts": {
            str(row["label"]): int(row["n_attributed"]) for row in done["report"]["per_class"]
        },
        "n_classes_without_positive": done["report"]["n_classes_without_positive"],
    }
    _write_json(
        {"stages": ctx.shared_stage_rows, "error": None},
        os.path.join(out_dir, SHARED_STAGES_FILENAME),
    )
    log.info("Атрибуция подготовлена: %s", report_path)


# ---------------------------------------------------------------------------
# Выполнение одного прогона
# ---------------------------------------------------------------------------


def _train_mlp(
    ctx: _ExperimentContext, n_features: int, X_train: sp.csr_matrix, X_val: sp.csr_matrix,
    run_result: dict, run_dir: str, features_builder: Any, tag: str,
) -> None:
    """Обучение MLP (общие гиперпараметры/seed, отличается только n_features),
    сохранение model-артефакта, замер этапа mlp_train."""
    cfg = ctx.cfg
    with _stage(ctx.device, "инициализация SparseMLP + train_model (early stopping по validation)",
                run_result, "mlp_train") as m:
        set_seed(cfg.seed)
        model = SparseMLP(n_features, tuple(ctx.maps.n_classes(lvl) for lvl in LEVELS), cfg.model)
        train_model(
            model, X_train, ctx.y_train, X_val, ctx.y_val,
            cfg.loss, cfg.train, ctx.class_weights, ctx.device, cfg.seed,
        )
        rule = RejectRule(0.5, False, False)
        reject_info = {
            "note": "S03-эксперимент: τ не калибровался; сравнение по raw top-1 на validation",
            "tau_mode": "fixed",
        }
        train_counts = {
            lvl: ctx.data.splits["train"][lvl].value_counts().sort_index().astype(int).to_dict()
            for lvl in LEVELS
        }
        cw_json = {
            "mode": cfg.loss.class_weight_mode,
            "alpha": cfg.loss.class_weight_alpha,
            "weights": {h: (None if w is None else w.tolist()) for h, w in ctx.class_weights.items()},
            "train_counts": train_counts,
        }
        art_dir = os.path.join(run_dir, "model")
        save_artifact(
            art_dir, model, cfg.model, features_builder, ctx.maps, rule, reject_info,
            cw_json, {"experiment_run": tag, "config": cfg.to_dict()},
            {
                "seed": cfg.seed,
                "n_features": int(n_features),
                "device": str(ctx.device),
                "feature_stats": {
                    "train": matrix_stats(X_train),
                    "val": matrix_stats(X_val),
                },
            },
        )
    # Строка этапа mlp_train уже добавлена контекстом _stage (success/failed):
    # повторно её не дублируем, здесь — только производные поля результата.
    run_result["train_seconds"] = m.seconds
    run_result["artifacts"]["model"] = os.path.abspath(art_dir)
    run_result["model_size_bytes"] = int(os.path.getsize(os.path.join(art_dir, "model.pt")))


def _validate_and_store_metrics(
    ctx: _ExperimentContext, model: SparseMLP, X_val: sp.csr_matrix, run_result: dict
) -> None:
    """Этап validation_inference: raw top-1 L3 без отказа, метрики и per-class."""
    with _stage(
        ctx.device, "predict по всей validation, без отказа; τ не применяется",
        run_result, "validation_inference",
    ) as m:
        preds, _ = predict(
            model, X_val, ctx.device, ctx.cfg.train.eval_batch_size, top_k=1
        )
        macro, acc, per_class = _l3_val_metrics(ctx.y_val[2], preds.l3_top1, ctx.l3_labels)
    run_result["val_inference_seconds"] = m.seconds
    run_result["metrics"] = {"l3_macro_f1": macro, "l3_accuracy": acc}
    run_result["per_class"] = per_class


def _run_none(plan: PlannedRun, ctx: _ExperimentContext, run_result: dict, run_dir: str) -> None:
    """Контрольный прогон без отбора: полный TF-IDF как есть."""
    run_result["stages"].append(_skipped_stage(
        plan.run_id, "selection", "не применимо: контрольный прогон без отбора (полный TF-IDF)"))
    run_result["stages"].append(_skipped_stage(
        plan.run_id, "selected_tfidf_refit", "не применимо: отбор не выполнялся"))
    fit_meta = ctx.full_meta.get("fit") or {}
    src = ctx.full_meta.get("source") or {}
    run_result["stages"].append({
        "run_id": plan.run_id, "stage": "full_tfidf_fit", "status": "reused",
        "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
        "measurement_note": (
            "полный TF-IDF построен ранее и только используется; fit в этом процессе не выполнялся; "
            f"артефакт {os.path.abspath(ctx.cfg.full_tfidf_path)}; "
            f"rows_fit={fit_meta.get('rows_fit')}, seed={ctx.full_meta.get('seed')}, "
            f"max_rows={src.get('max_rows')}, strategy={src.get('strategy')}"
        ),
    })
    with _stage(
        ctx.device,
        f"transform train+val полным TF-IDF, chunk_rows={ctx.cfg.features_chunk_rows}",
        run_result, "vectorize",
    ) as m:
        X_train, built_tr = ctx.ensure_full_train()
        X_val, built_va = ctx.ensure_full_val()
        if not (built_tr or built_va):
            m.note += ("; матрицы получены из общего кэша: первичный transform выполнен и "
                       f"измерен один раз под run_id={SHARED_VECTORIZATION_RUN_ID} (без двойного счёта)")
    run_result["n_features"] = int(ctx.full_builder.n_features)
    run_result["n_char"] = int(ctx.full_builder.n_char_features)
    run_result["n_word"] = int(ctx.full_builder.n_word_features)
    run_result["artifacts"]["vectorizer"] = os.path.abspath(ctx.cfg.full_tfidf_path)
    _train_mlp(ctx, ctx.full_builder.n_features, X_train, X_val, run_result, run_dir,
               ctx.full_builder, tag=plan.run_id)
    model = load_artifact(run_result["artifacts"]["model"], ctx.device).model
    _validate_and_store_metrics(ctx, model, X_val, run_result)


def _select_for_run(
    plan: PlannedRun, ctx: _ExperimentContext, run_result: dict, run_dir: str, sel_cfg: SelectionConfig
) -> tuple[Any, str]:
    """Этап selection прогона: Chi² (S03) или Gradient × Input (S04).

    Оба варианта — только по размеченному train полного TF-IDF; метки val/test
    не используются. Для G×I рейтинги построены один раз (shared/): здесь
    выполняется class-aware Top-K round-robin и сохранение артефакта отбора с
    полным provenance (источник TF-IDF, размеченные данные, checkpoint
    атрибуции). Возвращает (SelectionResult, абсолютный путь артефакта отбора).
    """
    cfg = ctx.cfg
    if plan.method == GRADIENT_X_INPUT:
        if ctx.attribution is None:
            raise ExperimentError(
                f"атрибуция Gradient × Input недоступна: {ctx.attribution_error}"
            )
        bundle = ctx.attribution
        note = (
            "class-aware Top-K round-robin из предварительно вычисленных G×I-рейтингов "
            "(shared, train-only, см. run_id=shared_attribution); метки val/test не используются"
        )
    else:
        bundle = None
        note = (
            "supervised Chi²-отбор только по размеченному train полного TF-IDF; "
            "метки val/test не используются"
        )
    with _stage(ctx.device, note, run_result, "selection") as m:
        if plan.method == GRADIENT_X_INPUT:
            result = select_features_gradient_x_input(
                bundle.ratings_dir, sel_cfg, ctx.n_char_full,
                int(ctx.full_builder.n_features), l3_labels=ctx.l3_labels,
            )
        else:
            X_full, built = ctx.ensure_full_train()
            if built:
                m.note += "; включён первичный transform полного train (результат закэширован)"
            else:
                m.note += (f"; матрица получена из общего кэша: первичный transform учтён под "
                           f"run_id={SHARED_VECTORIZATION_RUN_ID} (без двойного счёта)")
            result = select_features(
                X_full, ctx.y_train[2], sel_cfg, ctx.n_char_full, l3_labels=ctx.l3_labels
            )
        sel_dir = os.path.join(run_dir, "selection")
        provenance: dict[str, Any] = {
            "vectorizer": {
                "path": os.path.abspath(cfg.full_tfidf_path),
                "source_sha256": (ctx.full_meta.get("source") or {}).get("sha256"),
                "artifact_version": ctx.full_meta.get("artifact_version"),
            },
            "labeled": {"path": os.path.abspath(cfg.data.path), "split": "train"},
            "experiment": {"run_id": plan.run_id, "seed": cfg.seed},
        }
        if bundle is not None:
            provenance["attribution"] = bundle.run_link()
        save_selection_artifact(
            sel_dir, result, provenance=provenance, overwrite=True,
            ratings_copy=plan.method == GRADIENT_X_INPUT,
        )
    return result, os.path.abspath(sel_dir)


def _run_selected(plan: PlannedRun, ctx: _ExperimentContext, run_result: dict, run_dir: str) -> None:
    """Прогон с отбором: selection (train) → refit на том же корпусе → MLP → метрики."""
    cfg = ctx.cfg
    sel_cfg = SelectionConfig(
        method=plan.method, k=int(plan.requested_k),
        char_share=cfg.selection.char_share, seed=cfg.seed,
    )
    result, sel_dir = _select_for_run(plan, ctx, run_result, run_dir, sel_cfg)
    run_result["selection_report"] = result.report
    run_result["artifacts"]["selection"] = sel_dir
    if plan.method == GRADIENT_X_INPUT and ctx.attribution is not None:
        run_result["attribution"] = ctx.attribution.run_link()

    with _stage(
        ctx.device,
        "новые фиксированные словари char/word, fit на той же подвыборке корпуса, что полный TF-IDF; "
        "проверка sha256 корпуса, IDF и L2-нормировки",
        run_result, "selected_tfidf_refit",
    ) as m:
        refit = refit_selected_vectorizer(
            cfg.corpus_path, ctx.full_builder, ctx.full_meta, result,
            out_dir=os.path.join(run_dir, "tfidf_selected"), overwrite=True,
        )
    run_result["refit_checks"] = refit.checks
    run_result["artifacts"]["vectorizer"] = os.path.abspath(os.path.join(run_dir, "tfidf_selected"))
    run_result["n_features"] = int(refit.builder.n_features)
    run_result["n_char"] = int(refit.builder.n_char_features)
    run_result["n_word"] = int(refit.builder.n_word_features)

    with _stage(
        ctx.device, f"transform train+val выбранным TF-IDF, chunk_rows={cfg.features_chunk_rows}",
        run_result, "vectorize",
    ) as m:
        X_train = refit.builder.transform(
            ctx.data.splits["train"]["name"].tolist(), chunk_rows=cfg.features_chunk_rows
        )
        X_val = refit.builder.transform(
            ctx.data.splits["val"]["name"].tolist(), chunk_rows=cfg.features_chunk_rows
        )

    _train_mlp(ctx, refit.builder.n_features, X_train, X_val, run_result, run_dir,
               refit.builder, tag=plan.run_id)
    model = load_artifact(run_result["artifacts"]["model"], ctx.device).model
    _validate_and_store_metrics(ctx, model, X_val, run_result)


def _fresh_run_result(plan: PlannedRun, ctx: _ExperimentContext) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "method": plan.method,
        "requested_k": plan.requested_k,
        "status": "pending",
        "error": None,
        "seed": ctx.cfg.seed,
        "n_train": int(len(ctx.data.splits["train"])),
        "n_val": int(len(ctx.data.splits["val"])),
        "n_classes_eval": int(ctx.maps.n_classes("L3")),
        "label_map_sha256": ctx.label_map_sha256,
        "val_sha256": ctx.val_sha256,
        "train_sha256": (ctx.split_fingerprints.get("train") or {}).get("sha256"),
        "full_tfidf_sha256": getattr(ctx, "full_tfidf_sha256", None),
        "model_config_sha256": getattr(ctx, "model_config_sha256", None),
        "eval_protocol_sha256": getattr(ctx, "eval_protocol_sha256", None),
        "n_features": None,
        "n_char": None,
        "n_word": None,
        "metrics": None,
        "per_class": None,
        "stages": [],
        "artifacts": {"vectorizer": None, "model": None, "selection": None},
        "model_size_bytes": None,
        "train_seconds": None,
        "val_inference_seconds": None,
        "selection_report": None,
        "refit_checks": None,
        "attribution": None,
        "device": str(ctx.device),
        "versions": environment_versions(),
    }


# ---------------------------------------------------------------------------
# Эксперимент целиком
# ---------------------------------------------------------------------------


def _measure_shared_full_vectorization(ctx: _ExperimentContext) -> None:
    """Разделяемый transform полного TF-IDF train/validation (ревью 2, находка 2).

    Матрицы нужны каждому прогону (none, Chi², G×I) и разделяемой атрибуции,
    поэтому строятся один раз при подготовке контекста и замеряются отдельной
    строкой stages.csv с ``run_id=shared_vectorization``: стоимость методов не
    зависит от порядка прогонов, а учёт сценариев добавляет эту работу каждому
    методу ровно один раз (прогонные этапы vectorize при этом получают закэшированные
    матрицы и не дублируют затраты)."""
    def work() -> None:
        ctx.ensure_full_train()
        ctx.ensure_full_val()

    _shared_measurement(
        ctx, "vectorize",
        "единоразовый transform train+validation полным TF-IDF "
        f"(chunk_rows={ctx.cfg.features_chunk_rows}); результат закэширован и "
        "используется всеми прогонами — собственные этапы vectorize прогонов "
        "получают готовые матрицы и время не тратят",
        work,
        run_id=SHARED_VECTORIZATION_RUN_ID,
    )


def _prepare_context(cfg: ExperimentConfig, device: torch.device) -> _ExperimentContext:
    if not os.path.isdir(cfg.full_tfidf_path):
        raise ExperimentError(f"full_tfidf_path {cfg.full_tfidf_path!r}: каталог артефакта не найден")
    if not os.path.isfile(cfg.corpus_path):
        raise ExperimentError(f"corpus_path {cfg.corpus_path!r}: файл не найден")
    if not os.path.isfile(cfg.data.path):
        raise ExperimentError(f"data.path {cfg.data.path!r}: файл не найден")
    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
    log.info(
        "Полный TF-IDF: %s (словарь char %d, word %d; источник %s, rows_fit=%s)",
        cfg.full_tfidf_path, loaded.builder.n_char_features, loaded.builder.n_word_features,
        (loaded.meta.get("source") or {}).get("path"),
        (loaded.meta.get("fit") or {}).get("rows_fit"),
    )
    data = prepare_data(cfg.data, cfg.seed)
    ctx = _ExperimentContext(cfg, loaded, data, device)
    # Provenance: сплиты и label maps сохраняются один раз и общие для всех прогонов.
    os.makedirs(ctx.out, exist_ok=True)
    split_paths = save_splits(data.splits, os.path.join(ctx.out, "splits"))
    ctx.split_paths = split_paths  # type: ignore[attr-defined]
    ctx.split_fingerprints = {k: file_fingerprint(p) for k, p in split_paths.items()}  # type: ignore[attr-defined]
    maps_path = os.path.join(ctx.out, "label_maps.json")
    _write_json(data.maps.to_dict(), maps_path)
    ctx.label_maps_path = maps_path  # type: ignore[attr-defined]
    ctx.label_map_sha256 = _sha256_file(maps_path)  # type: ignore[attr-defined]
    ctx.val_sha256 = ctx.split_fingerprints["val"]["sha256"]  # type: ignore[attr-defined]
    ctx.train_counts = {  # type: ignore[attr-defined]
        lvl: data.splits["train"][lvl].value_counts().sort_index().astype(int).to_dict()
        for lvl in LEVELS
    }
    # Идентификаторы сопоставимости прогонов (S04, ревью 1): фактический train,
    # полный TF-IDF, конфигурация обучения MLP и протокол оценки. Дельты и
    # объединение в отчёте допускаются только при полном совпадении.
    ctx.full_tfidf_sha256 = _full_tfidf_sha256(cfg.full_tfidf_path)  # type: ignore[attr-defined]
    ctx.model_config_sha256 = _model_config_fingerprint(cfg)  # type: ignore[attr-defined]
    ctx.eval_protocol_sha256 = _sha256_json(METRIC_PROTOCOL)  # type: ignore[attr-defined]
    _measure_shared_full_vectorization(ctx)
    return ctx


def _write_experiment_meta(ctx: _ExperimentContext, planned: list[PlannedRun]) -> str:
    cfg = ctx.cfg
    git_commit, git_dirty = _git_info()
    full_dir = os.path.abspath(cfg.full_tfidf_path)
    label = "S04" if any(p.method == GRADIENT_X_INPUT for p in planned) else "S03"
    meta = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": f"{label}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "environment": {
            "versions": environment_versions(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "device": str(ctx.device),
        },
        "config": cfg.to_dict(),
        "planned_runs": [dataclasses.asdict(p) for p in planned],
        "inputs": {
            "full_tfidf": {
                "path": full_dir,
                "artifact_version": ctx.full_meta.get("artifact_version"),
                "seed": ctx.full_meta.get("seed"),
                "vocab": ctx.full_meta.get("vocab"),
                "tfidf_config": ctx.full_meta.get("tfidf_config"),
                "fit": ctx.full_meta.get("fit"),
                "files": {
                    name: {"sha256": _sha256_file(os.path.join(full_dir, name)),
                           "bytes": os.path.getsize(os.path.join(full_dir, name))}
                    for name in ("features.joblib", "vectorizer_meta.json")
                    if os.path.exists(os.path.join(full_dir, name))
                },
            },
            "corpus": {
                "path": os.path.abspath(cfg.corpus_path),
                "sha256": _sha256_file(cfg.corpus_path),
                "bytes": os.path.getsize(cfg.corpus_path),
                "expected_source_sha256": (ctx.full_meta.get("source") or {}).get("sha256"),
                "row_selection": {
                    "max_rows": (ctx.full_meta.get("source") or {}).get("max_rows"),
                    "strategy": (ctx.full_meta.get("source") or {}).get("strategy"),
                    "rows_used": (ctx.full_meta.get("source") or {}).get("rows_used"),
                    "vectorizer_fit_rows": (ctx.full_meta.get("tfidf_config") or {}).get(
                        "vectorizer_fit_rows"
                    ),
                    "seed": ctx.full_meta.get("seed"),
                    "source_id": (ctx.full_meta.get("source") or {}).get("source_id"),
                },
            },
            "labeled": file_fingerprint(cfg.data.path),
        },
        "splits": {
            k: {"path": os.path.abspath(v), "sha256": fp["sha256"], "bytes": fp["bytes"]}
            for k, v in ctx.split_paths.items() for fp in [ctx.split_fingerprints[k]]  # type: ignore[attr-defined]
        },
        "label_maps": {
            "path": os.path.abspath(ctx.label_maps_path),  # type: ignore[attr-defined]
            "sha256": ctx.label_map_sha256,  # type: ignore[attr-defined]
            "n_classes": {lvl: ctx.maps.n_classes(lvl) for lvl in LEVELS},
            "l3_labels": ctx.l3_labels,
            "l3_train_counts": ctx.train_counts["L3"],  # type: ignore[attr-defined]
        },
        "data_report": ctx.data.report,
        "metric_protocol": dict(METRIC_PROTOCOL),
        "attribution": ctx.attribution_meta,
        "limitations": list(LIMITATION_NOTES) + (
            [
                "gradient_x_input: интерпретация обученной attribution MLP не доказывает "
                "полезность признака после переобучения; одноэпоховый smoke не является "
                "доказательством качества метода."
            ]
            if ctx.attribution_meta is not None else []
        ),
    }
    path = os.path.join(ctx.out, "experiment.json")
    _write_json(meta, path)
    return path


def run_experiment(
    cfg: ExperimentConfig,
    only: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Выполняет плановые прогоны и сохраняет их результаты (без построения отчёта).

    ``only`` ограничивает множество выполняемых прогонов; остальные плановые
    сохраняются со статусом ``skipped`` (с причиной), чтобы отчёт покрывал все 7.
    Ошибка одного прогона не останавливает остальные. Существующие результаты
    без ``force`` не затираются.
    """
    cfg.validate()
    out = cfg.output_dir
    if os.path.exists(out):
        if not os.path.isdir(out):
            raise ExperimentError(f"output_dir {out!r} существует и не является каталогом")
        if os.listdir(out) and not force:
            raise ExperimentError(
                f"output_dir {out!r} не пуст: результаты не затираются. Выберите новый путь "
                "или передайте force=True / --force."
            )
    os.makedirs(out, exist_ok=True)

    device = resolve_device(cfg.train.device)
    if cfg.train.num_threads:
        torch.set_num_threads(cfg.train.num_threads)
    log.info("Эксперимент: устройство %s, seed %d, выход %s", device, cfg.seed, out)

    planned = planned_runs(cfg)
    if only is not None:
        unknown = sorted(set(only) - {p.run_id for p in planned})
        if unknown:
            raise ExperimentError(f"--only: неизвестные run_id {unknown}; план: {[p.run_id for p in planned]}")
    ctx = _prepare_context(cfg, device)
    # S04: разделяемая attribution MLP и G×I-рейтинги готовятся один раз до
    # прогонов, если среди выполняемых есть gradient_x_input_*.
    gxi_planned = [
        p for p in planned
        if p.method == GRADIENT_X_INPUT and (only is None or p.run_id in only)
    ]
    if gxi_planned:
        _prepare_attribution(ctx)
    meta_path = _write_experiment_meta(ctx, planned)
    log.info("Мета эксперимента: %s", meta_path)

    results: dict[str, dict[str, Any]] = {}
    for plan in planned:
        run_dir = os.path.join(out, RUNS_DIRNAME, plan.run_id)
        os.makedirs(run_dir, exist_ok=True)
        result = _fresh_run_result(plan, ctx)
        if only is not None and plan.run_id not in only:
            result["status"] = "skipped"
            result["error"] = "не выполнялся в этом запуске (--only)"
            _append_missing_stages(result, plan.method, "не выполнялся в этом запуске (--only)")
        else:
            t0 = time.perf_counter()
            try:
                log.info("=== run %s (method=%s, k=%s) ===", plan.run_id, plan.method, plan.requested_k)
                if plan.method == "none":
                    _run_none(plan, ctx, result, run_dir)
                else:
                    _run_selected(plan, ctx, result, run_dir)
                result["status"] = "success"
            except Exception as e:  # noqa: BLE001 — статус failed с причиной, прогоны независимы
                result["status"] = "failed"
                result["error"] = f"{type(e).__name__}: {e}"[:1000]
                # Этап с ошибкой уже зафиксирован со статусом failed (с затратами
                # до прерывания); недостигнутые этапы помечаем skipped.
                _append_missing_stages(result, plan.method,
                                       "не выполнен: прогон завершился ошибкой (см. error в runs.csv/manifest.json)")
                log.exception("run %s: failed", plan.run_id)
            result["wall_seconds"] = time.perf_counter() - t0
        _write_json(result, os.path.join(run_dir, "run_result.json"))
        results[plan.run_id] = result
        log.info(
            "run %s: %s%s (%.1f с)",
            plan.run_id, result["status"],
            f" — {result['error']}" if result["error"] else "", result.get("wall_seconds", 0.0),
        )
    summary = {r: results[r]["status"] for r in results}
    log.info("Итог эксперимента: %s", summary)
    # Разделяемые этапы (общий transform полного TF-IDF и, при наличии G×I,
    # подготовка атрибуции) сохраняются в любом случае: без них стоимость
    # сценариев в отчёте потеряла бы общую векторизацию.
    shared_rows = getattr(ctx, "shared_stage_rows", None)
    if shared_rows is not None:
        os.makedirs(os.path.join(out, SHARED_DIRNAME), exist_ok=True)
        _write_json(
            {"stages": shared_rows, "error": getattr(ctx, "attribution_error", None)},
            os.path.join(out, SHARED_DIRNAME, SHARED_STAGES_FILENAME),
        )
    return {"output_dir": out, "statuses": summary, "results": results}


# ---------------------------------------------------------------------------
# Агрегированный отчёт по сохранённым результатам
# ---------------------------------------------------------------------------


_RUN_STATUSES = ("success", "failed", "skipped")
_RUN_NUMERIC_FIELDS = (
    "n_features", "n_char", "n_word", "n_train", "n_val", "n_classes_eval",
    "seed", "model_size_bytes", "train_seconds", "val_inference_seconds",
)


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _sanitize_per_class(rows: Any, l3_labels: list[str] | None) -> list[dict[str, Any]] | None:
    """Проверяет построчные метрики: список dict с меткой L3 и целым
    ``support >= 0``, числовые precision/recall/f1 (или None). Состав L3 должен
    быть без дубликатов и в точности совпадать с общим label map, когда тот
    известен. Любое несоответствие — None: метрики не выдумываем и не чиним."""
    if not isinstance(rows, list) or not rows:
        return None
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        l3, support = row.get("L3"), row.get("support")
        if not isinstance(l3, str) or not isinstance(support, int) or isinstance(support, bool) or support < 0:
            return None
        clean: dict[str, Any] = {"L3": l3, "support": support}
        for key in ("precision", "recall", "f1"):
            v = row.get(key)
            if v is not None and not _is_finite_number(v):
                return None
            clean[key] = None if v is None else float(v)
        out.append(clean)
    l3s = [r["L3"] for r in out]
    if len(set(l3s)) != len(l3s):
        return None
    if l3_labels is not None and (len(out) != len(l3_labels) or set(l3s) != set(l3_labels)):
        return None
    return out


def _sanitize_run_result(raw: Any, run_id: str, l3_labels: list[str] | None) -> dict[str, Any]:
    """Приводит прочитанный с диска run_result.json к безопасной форме.

    Доверенной границей является только код, записавший файл. При чтении
    содержимое проверяется: нечисловые метрики/измерения, недопустимый статус,
    неполный или несовпадающий с label map per_class, повреждённое поле
    ``stages`` или его элементы у успешного прогона деградируют его статус
    в ``failed`` с понятной причиной, а недостоверные поля обнуляются —
    повреждённый запуск не должен ломать генерацию отчёта."""
    if not isinstance(raw, dict):
        return {
            "run_id": run_id, "status": "failed",
            "error": "run_result.json повреждён: ожидался объект верхнего уровня",
            "metrics": None, "per_class": None, "stages": [], "artifacts": {},
        }
    r = dict(raw)
    r["run_id"] = run_id
    if r.get("status") not in _RUN_STATUSES:
        r["status"] = "failed"
        r["error"] = f"статус {r.get('status')!r} недопустим"
    if r.get("error") is not None and not isinstance(r["error"], str):
        r["error"] = str(r["error"])
    for key in _RUN_NUMERIC_FIELDS:
        if key in r and not _is_finite_number(r[key]):
            r[key] = None
    for key in ("val_sha256", "label_map_sha256", "train_sha256", "full_tfidf_sha256",
                "model_config_sha256", "eval_protocol_sha256"):
        if r.get(key) is not None and not isinstance(r[key], str):
            r[key] = None
    artifacts = r.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    r["artifacts"] = {
        key: (artifacts.get(key) if isinstance(artifacts.get(key), str) else None)
        for key in ("vectorizer", "model", "selection")
    }
    stages_raw = r.get("stages")
    stage_problems: list[str] = []
    if not isinstance(stages_raw, list):
        stage_problems.append("stages повреждены: ожидался список этапов")
        stages_raw = []
    stages: list[dict[str, Any]] = []
    for row in stages_raw:
        if not isinstance(row, dict) or not isinstance(row.get("stage"), str):
            stage_problems.append("этап пропущен: запись повреждена (нет поля stage)")
            continue
        row = dict(row)
        row["run_id"] = run_id
        if row.get("status") not in _RUN_STATUSES:
            row["status"] = "skipped"
        for key in ("seconds", "peak_rss_bytes", "peak_vram_bytes"):
            if key in row and not _is_finite_number(row[key]):
                row[key] = None
        if not isinstance(row.get("measurement_note"), str):
            row["measurement_note"] = "" if row.get("measurement_note") is None else str(row["measurement_note"])
        stages.append(row)
    r["stages"] = stages
    if r["status"] == "success":
        problems: list[str] = list(stage_problems)
        metrics = r.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not _is_finite_number(metrics.get("l3_macro_f1"))
            or not _is_finite_number(metrics.get("l3_accuracy"))
        ):
            problems.append("metrics отсутствуют или содержат нечисловые значения")
        else:
            r["metrics"] = {
                "l3_macro_f1": float(metrics["l3_macro_f1"]),
                "l3_accuracy": float(metrics["l3_accuracy"]),
            }
        per_class = _sanitize_per_class(r.get("per_class"), l3_labels)
        if per_class is None:
            problems.append("per_class отсутствует, неполон или не соответствует общему label map")
        r["per_class"] = per_class
        if problems:
            # Содержимое успеха недостоверно: статус success сохранять нельзя.
            r["status"] = "failed"
            r["error"] = "; ".join(problems)
            r["metrics"] = None
    else:
        r["per_class"] = None
        if stage_problems and not r.get("error"):
            r["error"] = "; ".join(stage_problems)
        metrics = r.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not _is_finite_number(metrics.get("l3_macro_f1"))
            or not _is_finite_number(metrics.get("l3_accuracy"))
        ):
            r["metrics"] = None
    return r


def _load_run_result(output_dir: str, run_id: str,
                     l3_labels: list[str] | None = None) -> dict[str, Any] | None:
    path = os.path.join(output_dir, RUNS_DIRNAME, run_id, "run_result.json")
    if not os.path.isfile(path):
        return None
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError) as e:
        return {"run_id": run_id, "status": "failed", "error": f"run_result.json не читается: {e}",
                "stages": [], "artifacts": {}, "metrics": None, "per_class": None}
    return _sanitize_run_result(raw, run_id, l3_labels)


# Ключи идентичности сопоставимости прогонов. Семантика сравнения задаётся
# явной схемой версии (см. IDENTITY_SCHEMA_*), а не неявным видом данных:
# строгая схема S04 требует все семь непустых полей у обеих сторон, legacy-схема
# S03 — совпадения validation и состава L3 (прежнее поведение отчётов S03).
_IDENTITY_REQUIRED_KEYS = ("val_sha256", "label_map_sha256")
_IDENTITY_EXTENDED_KEYS = (
    "train_sha256", "full_tfidf_sha256", "model_config_sha256",
    "eval_protocol_sha256", "seed",
)
IDENTITY_ALL_KEYS = _IDENTITY_REQUIRED_KEYS + _IDENTITY_EXTENDED_KEYS
IDENTITY_SCHEMA_S04 = "s04-identity@1"
IDENTITY_SCHEMA_LEGACY = "s03-identity@1"


def _identity_schema_for_experiment(schema_version: Any) -> str:
    """Схема сопоставимости по ``schema_version`` эксперимента.

    Явный маркер ``s03-*`` включает семантику legacy-отчётов S03. Всё
    остальное (``s04-*``, отсутствующая или неизвестная версия) получает
    строгую семантику S04: без явного заявления legacy данные сопоставимости
    подтверждёнными не считаются (fail closed).
    """
    if str(schema_version or "").startswith("s03"):
        return IDENTITY_SCHEMA_LEGACY
    return IDENTITY_SCHEMA_S04


def _identity_schema_for_manifest(manifest: dict) -> str:
    """Схема сопоставимости сохранённого отчёта.

    Явное поле ``comparability_schema`` (пишется в manifest с S04) имеет
    приоритет; иначе вывод по ``schema_version`` манифеста.
    """
    explicit = (manifest or {}).get("comparability_schema")
    if explicit in (IDENTITY_SCHEMA_S04, IDENTITY_SCHEMA_LEGACY):
        return str(explicit)
    return _identity_schema_for_experiment((manifest or {}).get("schema_version"))


def _identity_value_missing(v: Any) -> bool:
    """Отсутствие значения идентификатора сопоставимости.

    Отсутствует — только None (для строк также пустая строка); ``seed=0`` —
    допустимое значение и идентичность не отменяет (ревью 2, находка 5)."""
    return v is None or (isinstance(v, str) and not v)


def _comparable_identity(
    result: dict | None,
    baseline: dict | None,
    identity_schema: str = IDENTITY_SCHEMA_S04,
) -> bool:
    """Подтверждена ли идентичность прогона и baseline для сравнения.

    ``IDENTITY_SCHEMA_S04`` (fail closed): все семь идентификаторов непусты у
    обеих сторон и попарно равны — validation, состав L3, фактический train,
    полный TF-IDF, конфигурация обучения MLP, протокол оценки и seed.
    Отсутствие поля у обеих сторон идентичность не доказывает: сопоставимость
    не выводится из совпадения val/labels в отсутствие остальных
    идентификаторов; подставленный чужой run_result.json дельт не получает.
    Нули (в частности ``seed=0``) от допустимых значений не отличаются —
    отсутствие значения — только None/пустая строка.
    ``IDENTITY_SCHEMA_LEGACY`` (отчёты S03): требуются непустые равные
    ``val_sha256`` и ``label_map_sha256``; семантика S03 сохраняется по явной
    схеме версии, а не ослаблением правил S04.
    """
    if not result or not baseline:
        return False
    keys = IDENTITY_ALL_KEYS if identity_schema == IDENTITY_SCHEMA_S04 else _IDENTITY_REQUIRED_KEYS
    for key in keys:
        a, b = result.get(key), baseline.get(key)
        if _identity_value_missing(a) or _identity_value_missing(b) or a != b:
            return False
    return True


def _runs_frame(
    planned: list[dict],
    results: dict[str, dict],
    baseline: dict | None,
    identity_schema: str = IDENTITY_SCHEMA_S04,
) -> pd.DataFrame:
    rows = []
    for p in planned:
        r = results.get(p["run_id"]) or {
            "status": "skipped", "error": "run_result.json отсутствует", "stages": [],
        }
        metrics = r.get("metrics") or {}
        row = {
            "run_id": p["run_id"],
            "status": r.get("status"),
            "error": r.get("error"),
            "method": p["method"],
            "requested_k": p["requested_k"],
            "n_features": r.get("n_features"),
            "n_char": r.get("n_char"),
            "n_word": r.get("n_word"),
            "n_train": r.get("n_train"),
            "n_val": r.get("n_val"),
            "n_classes_eval": r.get("n_classes_eval"),
            "seed": r.get("seed"),
            "l3_macro_f1": metrics.get("l3_macro_f1"),
            "l3_accuracy": metrics.get("l3_accuracy"),
            "delta_macro_f1": None,
            "delta_accuracy": None,
            "train_seconds": r.get("train_seconds"),
            "val_inference_seconds": r.get("val_inference_seconds"),
            "model_size_bytes": r.get("model_size_bytes"),
            "vectorizer_artifact": (r.get("artifacts") or {}).get("vectorizer"),
            "model_artifact": (r.get("artifacts") or {}).get("model"),
            "selection_artifact": (r.get("artifacts") or {}).get("selection"),
        }
        rows.append(row)
    df = pd.DataFrame(rows, columns=RUNS_COLUMNS)
    # Дельты — только если baseline и прогон успешны и подтверждена идентичность
    # в семантике схемы отчёта: S04 — все семь непустых идентификаторов
    # (validation, label maps, train, full TF-IDF, конфигурация MLP, протокол,
    # seed; см. _comparable_identity), legacy S03 — val/labels. Колонка
    # сопоставимости сохраняется в runs.csv и используется графиками HTML.
    df["comparable_with_baseline"] = [
        bool(
            baseline is not None
            and baseline.get("status") == "success"
            and _comparable_identity(results.get(rid), baseline, identity_schema)
        )
        for rid in df["run_id"]
    ]
    if baseline is not None and baseline.get("status") == "success":
        b = baseline.get("metrics") or {}
        for i, row in df.iterrows():
            r = results.get(row["run_id"]) or {}
            comparable = (
                row["status"] == "success"
                and bool(row["comparable_with_baseline"])
                and b.get("l3_macro_f1") is not None
                and row["l3_macro_f1"] is not None
                and b.get("l3_accuracy") is not None
                and row["l3_accuracy"] is not None
            )
            if comparable:
                df.at[i, "delta_macro_f1"] = row["l3_macro_f1"] - b["l3_macro_f1"]
                df.at[i, "delta_accuracy"] = row["l3_accuracy"] - b["l3_accuracy"]
    for col in ("requested_k", "n_features", "n_char", "n_word", "n_train", "n_val",
                "n_classes_eval", "seed", "model_size_bytes"):
        df[col] = df[col].astype("Int64")
    return df


def _per_class_frame(
    planned: list[dict],
    results: dict[str, dict],
    baseline: dict | None,
    identity_schema: str = IDENTITY_SCHEMA_S04,
) -> pd.DataFrame:
    baseline_ok = bool(baseline) and baseline.get("status") == "success"
    base_rows = {
        row["L3"]: row for row in baseline.get("per_class") or []
    } if baseline_ok else {}
    out = []
    for p in planned:
        r = results.get(p["run_id"]) or {}
        if r.get("status") != "success" or not r.get("per_class"):
            continue  # статус неуспешного run — в runs.csv/manifest, построчных метрик не выдумываем
        # baseline-поля и дельты — только при подтверждённой в семантике схемы
        # отчёта идентичности; иначе строки сохраняются без них (null).
        comparable = baseline_ok and _comparable_identity(r, baseline, identity_schema)
        for row in r["per_class"]:
            b = base_rows.get(row["L3"]) or {} if comparable else {}
            b_f1 = b.get("f1")
            b_rec = b.get("recall")
            f1 = row.get("f1")
            rec = row.get("recall")
            out.append({
                "run_id": p["run_id"],
                "L3": row["L3"],
                "support": row["support"],
                "precision": row.get("precision"),
                "recall": rec,
                "f1": f1,
                "baseline_f1": b_f1,
                "delta_f1": (f1 - b_f1) if (f1 is not None and b_f1 is not None) else None,
                "baseline_recall": b_rec,
                "delta_recall": (rec - b_rec) if (rec is not None and b_rec is not None) else None,
            })
    return pd.DataFrame(out, columns=PER_CLASS_COLUMNS)


def _load_shared_stage_rows(output_dir: str) -> list[dict[str, Any]]:
    """Читает строки разделяемых этапов атрибуции (shared/attribution_stages.json).

    Повреждённый или отсутствующий файл не ломает генерацию отчёта: возвращается
    пустой список. Содержимое проверяется по минимальной структуре строки.
    """
    path = os.path.join(output_dir, SHARED_DIRNAME, SHARED_STAGES_FILENAME)
    if not os.path.isfile(path):
        return []
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Разделяемые этапы атрибуции не прочитаны (%s): пропускаю", e)
        return []
    rows = raw.get("stages") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("stage"), str):
            continue
        row = dict(row)
        # run_id сохраняется из файла: там могут быть как shared_attribution,
        # так и shared_vectorization. Подстановка SHARED_RUN_ID — только для
        # строк без run_id (файлы, записанные старым кодом/вручную).
        if not isinstance(row.get("run_id"), str) or not row["run_id"]:
            row["run_id"] = SHARED_RUN_ID
        if row.get("status") not in _RUN_STATUSES:
            row["status"] = "skipped"
        for key in ("seconds", "peak_rss_bytes", "peak_vram_bytes"):
            if key in row and not _is_finite_number(row[key]):
                row[key] = None
        if not isinstance(row.get("measurement_note"), str):
            row["measurement_note"] = (
                "" if row.get("measurement_note") is None else str(row.get("measurement_note"))
            )
        out.append(row)
    return out


def _stages_frame(
    planned: list[dict], results: dict[str, dict], output_dir: str | None = None
) -> pd.DataFrame:
    """Строки этапов всех плановых прогонов.

    Для прогонов без полного набора этапов (failed/skipped, старые или
    частичные ``run_result.json``) недостигнутые этапы дописываются строками
    ``skipped`` с пояснением: так стоимость неуспешной работы не скрывается и
    контракт stages.csv выполняется для каждого планового прогона. При заданном
    ``output_dir`` добавляются разделяемые этапы атрибуции
    (``run_id=shared_attribution``): их затраты учтены один раз и не дублируются
    в прогонах ``gradient_x_input_*``.
    """
    rows = []
    for p in planned:
        raw = results.get(p["run_id"])
        r = raw or {}
        stages = list(r.get("stages") or [])
        rows.extend(stages)
        reached = {row.get("stage") for row in stages}
        missing = [s for s in STAGES_SELECTED if s not in reached]
        if missing:
            status = r.get("status")
            if raw is None:
                reason = "run_result.json отсутствует: этапы прогона не зафиксированы"
            elif status == "failed":
                reason = "не выполнен: прогон завершился ошибкой (см. error в runs.csv/manifest.json)"
            elif status == "skipped":
                reason = "не выполнялся в этом запуске (run пропущен)"
            else:
                reason = "этап не зафиксирован в сохранённых результатах прогона"
            for name in missing:
                rows.append(_skipped_stage(p["run_id"], name, reason))
    if output_dir is not None:
        rows.extend(_load_shared_stage_rows(output_dir))
    return pd.DataFrame(rows, columns=STAGES_COLUMNS)


def build_manifest(
    output_dir: str,
    exp: dict,
    runs_df: pd.DataFrame,
    identity_schema: str = IDENTITY_SCHEMA_S04,
) -> dict[str, Any]:
    env = exp.get("environment") or {}
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "comparability_schema": identity_schema,
        "experiment_id": exp.get("experiment_id"),
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "experiment_created_at": exp.get("created_at"),
        "git_commit": exp.get("git_commit"),
        "git_dirty": exp.get("git_dirty"),
        "versions": env.get("versions"),
        "device": env.get("device"),
        "platform": env.get("platform"),
        "cpu_count": env.get("cpu_count"),
        "resource_limitations": list(exp.get("limitations") or []),
        "inputs": exp.get("inputs"),
        "splits": exp.get("splits"),
        "label_maps": {k: v for k, v in (exp.get("label_maps") or {}).items() if k != "l3_labels"},
        "l3_composition_hash": exp.get("label_maps", {}).get("sha256"),
        "data_report": exp.get("data_report"),
        "configs": {
            "experiment": exp.get("config"),
            "selection": (exp.get("config") or {}).get("selection"),
            "data": (exp.get("config") or {}).get("data"),
            "model": (exp.get("config") or {}).get("model"),
            "train": (exp.get("config") or {}).get("train"),
            "loss": (exp.get("config") or {}).get("loss"),
        },
        "metric_protocol": exp.get("metric_protocol"),
        "attribution": exp.get("attribution"),
        "planned_runs": [
            {
                "run_id": row.run_id, "method": row.method, "requested_k": row.requested_k,
                "status": row.status, "error": row.error,
                "comparable_with_baseline": bool(getattr(row, "comparable_with_baseline", False)),
                "vectorizer_artifact": row.vectorizer_artifact,
                "model_artifact": row.model_artifact,
                "selection_artifact": row.selection_artifact,
            }
            for row in runs_df.itertuples()
        ],
        "notes": [
            "Отчёт построен из сохранённых run_result.json без повторного обучения.",
            "Метки test не участвовали в selection и сравнении; test-сплит сохранён только как provenance.",
            "Все отсутствующие измерения — null, не нули.",
        ],
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _fmt_mb(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x / (1024 * 1024):.1f}"


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _table_html(df: pd.DataFrame, columns: list[str], formats: dict[str, Any] | None = None,
                caption: str | None = None) -> str:
    formats = formats or {}
    parts = ["<table>", "<thead><tr>"]
    for c in columns:
        parts.append(f"<th>{_esc(c)}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        parts.append("<tr>")
        for c in columns:
            v = row[c]
            txt = formats[c](v) if c in formats and v is not None and not pd.isna(v) else _fmt(v)
            parts.append(f"<td>{_esc(txt)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if caption:
        parts.append(f'<p class="caption">{_esc(caption)}</p>')
    return "".join(parts)


def _svg_line_chart(
    title: str,
    xs: list[float],
    series: list[tuple[str, str, dict[float, float]]],
    baseline: tuple[str, float] | None = None,
    y_fmt: str = "{:.4f}",
    w: int = 700,
    h: int = 380,
) -> str:
    """Линейный SVG-график. series: (имя, цвет, {x: y}).

    Линия каждой серии разрывается на пропусках плановых точек ``xs``:
    соединяются только соседние запланированные K, обе имеющие значение;
    участок «через» незапущенный/неуспешный бюджет не дорисовывается.
    Точки (круги) отмечаются для всех имеющихся значений.
    """
    ml, mr, mt, mb = 64, 18, 40, 52
    pw, ph = w - ml - mr, h - mt - mb
    ys = [y for _, _, pts in series for y in pts.values() if y is not None]
    if baseline is not None:
        ys.append(baseline[1])
    if not xs or not ys:
        return f'<p class="muted">Нет данных для графика «{_esc(title)}».</p>'
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    pad = (y_max - y_min) * 0.08
    y_min, y_max = y_min - pad, y_max + pad
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_max = x_min + 1.0

    def X(v: float) -> float:
        return ml + (v - x_min) / (x_max - x_min) * pw

    def Y(v: float) -> float:
        return mt + ph - (v - y_min) / (y_max - y_min) * ph

    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_esc(title)}">']
    s.append(f'<text x="{w / 2}" y="20" text-anchor="middle" class="chart-title">{_esc(title)}</text>')
    # оси и сетка
    for i in range(5):
        gy = y_min + (y_max - y_min) * i / 4
        s.append(f'<line x1="{ml}" y1="{Y(gy):.1f}" x2="{w - mr}" y2="{Y(gy):.1f}" class="grid"/>')
        s.append(f'<text x="{ml - 6}" y="{Y(gy) + 4:.1f}" text-anchor="end" class="tick">{y_fmt.format(gy)}</text>')
    for x in xs:
        s.append(f'<line x1="{X(x):.1f}" y1="{mt}" x2="{X(x):.1f}" y2="{mt + ph}" class="grid"/>')
        s.append(f'<text x="{X(x):.1f}" y="{mt + ph + 18}" text-anchor="middle" class="tick">{x:g}</text>')
    s.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" class="axis"/>')
    s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" class="axis"/>')
    if baseline is not None:
        name, by = baseline
        s.append(f'<line x1="{ml}" y1="{Y(by):.1f}" x2="{w - mr}" y2="{Y(by):.1f}" class="baseline"/>')
        s.append(
            f'<text x="{w - mr - 4}" y="{Y(by) - 6:.1f}" text-anchor="end" class="tick baseline-label">'
            f'{_esc(name)} = {y_fmt.format(by)}</text>'
        )
    lx = ml + 8
    for name, color, pts in series:
        present = {x: pts[x] for x in xs if pts.get(x) is not None}
        # Сегменты непрерывных запланированных K: разрыв на любом пропуске.
        run: list[tuple[float, float]] = []

        def _emit(seg: list[tuple[float, float]]) -> None:
            if len(seg) < 2:
                return
            path = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in seg)
            s.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')

        for x in xs:
            v = pts.get(x)
            if v is None:
                _emit(run)
                run = []
            else:
                run.append((x, v))
        _emit(run)
        for x, y in present.items():
            s.append(
                f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{color}">'
                f'<title>{_esc(name)}, K={x:g}: {y_fmt.format(y)}</title></circle>'
            )
        s.append(f'<rect x="{lx}" y="{h - 16}" width="10" height="4" fill="{color}"/>')
        s.append(f'<text x="{lx + 14}" y="{h - 12}" class="tick">{_esc(name)}</text>')
        lx += 14 + 7 * len(name) + 24
    s.append("</svg>")
    return "".join(s)


def _svg_bar_chart(
    title: str,
    groups: list[float],
    series: list[tuple[str, str, dict[float, float]]],
    y_fmt: str = "{:.1f}",
    w: int = 700,
    h: int = 360,
) -> str:
    """Группированные столбцы: groups по X, series — ряды методов."""
    ml, mr, mt, mb = 64, 18, 40, 52
    pw, ph = w - ml - mr, h - mt - mb
    vals = [y for _, _, pts in series for y in pts.values() if y is not None]
    if not groups or not vals:
        return f'<p class="muted">Нет данных для графика «{_esc(title)}».</p>'
    y_max = max(vals) * 1.15 or 1.0

    def Y(v: float) -> float:
        return mt + ph - v / y_max * ph

    slot = pw / len(groups)
    bw = min(48.0, slot / max(len(series), 1) * 0.8)
    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_esc(title)}">']
    s.append(f'<text x="{w / 2}" y="20" text-anchor="middle" class="chart-title">{_esc(title)}</text>')
    for i in range(5):
        gy = y_max * i / 4
        s.append(f'<line x1="{ml}" y1="{Y(gy):.1f}" x2="{w - mr}" y2="{Y(gy):.1f}" class="grid"/>')
        s.append(f'<text x="{ml - 6}" y="{Y(gy) + 4:.1f}" text-anchor="end" class="tick">{y_fmt.format(gy)}</text>')
    s.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" class="axis"/>')
    s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" class="axis"/>')
    for gi, g in enumerate(groups):
        cx = ml + slot * gi + slot / 2
        s.append(f'<text x="{cx:.1f}" y="{mt + ph + 18}" text-anchor="middle" class="tick">{g:g}</text>')
        for si, (name, color, pts) in enumerate(series):
            v = pts.get(g)
            if v is None:
                continue
            x0 = cx - (len(series) * bw) / 2 + si * bw
            s.append(f'<rect x="{x0:.1f}" y="{Y(v):.1f}" width="{bw - 4:.1f}" height="{mt + ph - Y(v):.1f}" fill="{color}">'
                     f'<title>{_esc(name)}, K={g:g}: {y_fmt.format(v)}</title></rect>')
    lx = ml + 8
    for name, color, _ in series:
        s.append(f'<rect x="{lx}" y="{h - 18}" width="10" height="10" fill="{color}"/>')
        s.append(f'<text x="{lx + 14}" y="{h - 9}" class="tick">{_esc(name)}</text>')
        lx += 14 + 7 * len(name) + 24
    s.append("</svg>")
    return "".join(s)


def _svg_scatter_chart(
    title: str,
    points: list[tuple[str, str, float, float]],  # (имя, цвет, x, y)
    x_label: str,
    y_label: str,
    y_fmt: str = "{:.4f}",
    w: int = 700,
    h: int = 380,
) -> str:
    """Точечный SVG-график «качество ↔ ресурсы»: points — (имя, цвет, x, y)."""
    ml, mr, mt, mb = 64, 18, 40, 64
    pw, ph = w - ml - mr, h - mt - mb
    if not points:
        return f'<p class="muted">Нет данных для графика «{_esc(title)}».</p>'
    xs = [p[2] for p in points]
    ys = [p[3] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= max(0.5, abs(x_min) * 0.1)
        x_max += max(0.5, abs(x_max) * 0.1)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def X(v: float) -> float:
        return ml + (v - x_min) / (x_max - x_min) * pw

    def Y(v: float) -> float:
        return mt + ph - (v - y_min) / (y_max - y_min) * ph

    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_esc(title)}">']
    s.append(f'<text x="{w / 2}" y="20" text-anchor="middle" class="chart-title">{_esc(title)}</text>')
    for i in range(5):
        gy = y_min + (y_max - y_min) * i / 4
        gx = x_min + (x_max - x_min) * i / 4
        s.append(f'<line x1="{ml}" y1="{Y(gy):.1f}" x2="{w - mr}" y2="{Y(gy):.1f}" class="grid"/>')
        s.append(f'<text x="{ml - 6}" y="{Y(gy) + 4:.1f}" text-anchor="end" class="tick">{y_fmt.format(gy)}</text>')
        s.append(f'<line x1="{X(gx):.1f}" y1="{mt}" x2="{X(gx):.1f}" y2="{mt + ph}" class="grid"/>')
        s.append(f'<text x="{X(gx):.1f}" y="{mt + ph + 18}" text-anchor="middle" class="tick">{gx:.1f}</text>')
    s.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{w - mr}" y2="{mt + ph}" class="axis"/>')
    s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" class="axis"/>')
    s.append(f'<text x="{ml + pw / 2}" y="{h - 26}" text-anchor="middle" class="tick">{_esc(x_label)}</text>')
    s.append(
        f'<text x="16" y="{mt + ph / 2}" text-anchor="middle" class="tick" '
        f'transform="rotate(-90 16 {mt + ph / 2})">{_esc(y_label)}</text>'
    )
    for name, color, x, y in points:
        s.append(
            f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5" fill="{color}">'
            f'<title>{_esc(name)}: {x:.2f} c, {y_fmt.format(y)}</title></circle>'
        )
    seen: dict[str, str] = {}
    for name, color, _, _ in points:
        seen.setdefault(name, color)
    lx = ml + 8
    for name, color in seen.items():
        s.append(f'<rect x="{lx}" y="{h - 14}" width="10" height="4" fill="{color}"/>')
        s.append(f'<text x="{lx + 14}" y="{h - 10}" class="tick">{_esc(name)}</text>')
        lx += 14 + 7 * len(name) + 24
    s.append("</svg>")
    return "".join(s)


def _shared_stage_cost(stages_df: pd.DataFrame, run_id: str, absent_ok: bool) -> float | None:
    """Стоимость разделяемого run_id: сумма успешных измеренных этапов.

    ``absent_ok=True`` — разделяемый этап не выполнялся и не требовался
    (старые отчёты без него): вклад 0.0, а не неизвестность. Иначе (absent_ok=
    False) отсутствие строк означает, что стоимость не зафиксирована — None.
    Любой успешный этап без измеренного времени или полное отсутствие успешных
    этапов делает итог неизвестным (None), а не частичной суммой."""
    rows = stages_df[stages_df["run_id"] == run_id]
    if len(rows) == 0:
        return 0.0 if absent_ok else None
    succ = rows[rows["status"] == "success"]
    secs = pd.to_numeric(succ["seconds"], errors="coerce")
    if len(succ) == 0 or secs.isna().any():
        return None
    return float(secs.sum())


def _scenario_cost_rows(
    stages_df: pd.DataFrame, planned_runs: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Строки стоимости сценариев: по всем плановым прогонам и агрегаты по методам.

    В ``planned_runs`` передаются **все** плановые прогоны (включая
    failed/skipped и не сопоставимые с baseline): сопоставимость — требование к
    сравнению качества, а бюджет сценария обязан быть виден неполным, а не
    суммой оставшихся успешных запусков (ревью 2, находка 3).

    Разделяемые затраты выделяются из прогонов и учитываются один раз на метод:

    - ``run_id=shared_vectorization`` — общий transform полного TF-IDF
      train/validation; его работу используют все методы, поэтому стоимость
      добавляется каждому сценарию ровно один раз (независимо от порядка
      прогонов; раньше первый прогон «платил» за transform внутри своего
      этапа vectorize, а остальные получали ~0 — графики были несопоставимы);
    - ``run_id=shared_attribution`` — обучение attribution MLP и сама атрибуция;
      нужны только ``gradient_x_input``, поэтому на none и Chi²-сценарии
      (global, class_aware) не распределяются и учитываются ровно один раз в
      итоге метода (без двойного счёта между бюджетами).

    Неизвестные времена нулём не заменяются: успешный этап без измеренного
    времени, неуспешный/пропущенный плановый бюджет или неизмеренная
    разделяемая подготовка делают ``scenario_total_seconds`` прогона/метода
    равным null, а не частичной сумме; у агрегата метода поле ``complete``
    фиксирует полноту бюджетов явно. Загруженный извне checkpoint фиксирует
    ``attribution_model_train`` с временем null: стоимость его обучения в этом
    процессе неизвестна, и полный итог G×I честно остаётся null (ревью 2,
    находка 2).
    """
    success_stages = stages_df[stages_df["status"] == "success"].copy()
    success_stages["_sec"] = pd.to_numeric(success_stages["seconds"], errors="coerce")
    own_by_run = (
        success_stages.dropna(subset=["_sec"]).groupby("run_id")["_sec"].sum().to_dict()
    )
    # Прогоны/разделяемые run_id, у которых хотя бы один успешный этап без
    # измеренного времени: полная стоимость такого сценария неизвестна —
    # частичная сумма по измеренным этапам в итог не подставляется.
    unknown_time_runs = set(success_stages.loc[success_stages["_sec"].isna(), "run_id"])
    vec_seconds = _shared_stage_cost(stages_df, SHARED_VECTORIZATION_RUN_ID, absent_ok=True)
    attr_seconds = _shared_stage_cost(stages_df, SHARED_RUN_ID, absent_ok=False)

    def _status_of(row: Any) -> str:
        status = getattr(row, "status", None)
        return "success" if status is None else str(status)

    run_rows: list[dict[str, Any]] = []
    for r in planned_runs.itertuples():
        own = own_by_run.get(r.run_id)
        status = _status_of(r)
        is_gxi = r.method == GRADIENT_X_INPUT
        shared = attr_seconds if is_gxi else None
        # Полная стоимость прогона известна только для успешного прогона с
        # полностью измеренными собственными этапами, измеренным общим
        # transform и (для G×I) разделяемой атрибуцией; иначе — null.
        total = None
        if (status == "success" and own is not None and r.run_id not in unknown_time_runs
                and vec_seconds is not None and (shared is not None or not is_gxi)):
            total = float(own) + vec_seconds + (shared or 0.0)
        run_rows.append({
            "run_id": r.run_id, "method": r.method, "requested_k": r.requested_k,
            "status": status,
            "own_seconds": (float(own) if own is not None else None),
            "vectorization_seconds": vec_seconds,
            "shared_seconds": shared,
            "scenario_total_seconds": total,
            "l3_macro_f1": getattr(r, "l3_macro_f1", None),
        })

    total_rows: list[dict[str, Any]] = []
    for method in dict.fromkeys(planned_runs["method"]):
        sub = planned_runs[planned_runs["method"] == method]
        is_gxi = method == GRADIENT_X_INPUT
        shared = attr_seconds if is_gxi else None
        owns, n_success = [], 0
        complete = True
        for rid, status in zip(sub["run_id"], sub.get("status", ["success"] * len(sub))):
            own = own_by_run.get(rid)
            owns.append(own)
            if status == "success":
                n_success += 1
            if status != "success" or own is None or rid in unknown_time_runs:
                complete = False
        own_sum = float(sum(owns)) if owns and all(v is not None for v in owns) else None
        total = None
        if complete and own_sum is not None and vec_seconds is not None and (
            shared is not None or not is_gxi
        ):
            total = own_sum + vec_seconds + (shared or 0.0)
        total_rows.append({
            "method": method, "n_budgets": int(len(sub)), "n_successful": int(n_success),
            "complete": bool(complete),
            "own_seconds": own_sum,
            "vectorization_seconds": vec_seconds,
            "shared_seconds": shared,
            "scenario_total_seconds": total,
        })
    return run_rows, total_rows


def _html_report(
    output_dir: str, exp: dict, runs_df: pd.DataFrame, per_class_df: pd.DataFrame,
    stages_df: pd.DataFrame, manifest: dict, report_dir: str
) -> str:
    cfg = exp.get("config") or {}
    noisy = cfg.get("noisy_support", 2)
    baseline_id = BASELINE_RUN_ID
    succ = runs_df[runs_df["status"] == "success"]
    base = runs_df[runs_df["run_id"] == baseline_id]
    base_metrics = {}
    if len(base) and base.iloc[0]["status"] == "success":
        base_metrics = {"macro_f1": base.iloc[0]["l3_macro_f1"], "accuracy": base.iloc[0]["l3_accuracy"]}

    sel_runs = succ[succ["method"] != "none"].copy()
    # Графики сравнивают методы между собой и с baseline: только прогоны с
    # подтверждённой сопоставимостью (run_result.json с иными train, full
    # TF-IDF, конфигурацией MLP, seed, label maps или протоколом исключаются;
    # см. _comparable_identity). Семантика подтверждения задаётся явной схемой
    # отчёта: s03-identity@1 — прежнее поведение отчётов S03, s04-identity@1 —
    # все семь непустых идентификаторов.
    identity_schema = _identity_schema_for_manifest(manifest)
    comparability_note = ""
    if "comparable_with_baseline" in succ.columns:
        chart_runs = sel_runs[sel_runs["comparable_with_baseline"].astype(bool)].copy()
    elif identity_schema == IDENTITY_SCHEMA_LEGACY:
        # Файлы отчёта S03 без колонки сопоставимости: сохраняем прежнюю
        # семантику — все успешные прогоны участвуют в сравнении.
        chart_runs = sel_runs
    else:
        # S04 без данных о сопоставимости в сохранённых таблицах: отсутствие
        # данных не считается подтверждением — сравнительные графики пустые,
        # с видимым предупреждением (fail closed).
        chart_runs = sel_runs.iloc[0:0].copy()
        comparability_note = (
            '<p class="warn">В сохранённых файлах отчёта нет данных о сопоставимости '
            "(колонка comparable_with_baseline в runs.csv и manifest.json): прогоны не "
            "считаются подтверждённо сопоставимыми, сравнительные графики и дельты не "
            "показаны. Пересоберите файлы отчёта из experiment.json и runs/, чтобы "
            "проверить идентификаторы прогонов.</p>"
        )
    # Ось K — все запланированные бюджеты (включая failed/skipped): значения
    # незапущенных конфигураций не дорисовываются, на их месте — разрыв линии.
    ks = sorted({float(k) for k in runs_df["requested_k"].dropna()})
    colors = {"global": "#1f77b4", "class_aware": "#ff7f0e", GRADIENT_X_INPUT: "#9467bd"}

    def series_of(metric: str) -> list[tuple[str, str, dict[float, float]]]:
        out = []
        for method in dict.fromkeys(chart_runs["method"]):
            sub = chart_runs[chart_runs["method"] == method]
            pts = {float(r.requested_k): float(getattr(r, metric)) for r in sub.itertuples()
                   if getattr(r, metric) is not None and not pd.isna(getattr(r, metric))}
            if pts:
                out.append((method, colors.get(method, "#333333"), pts))
        return out

    macro_chart = _svg_line_chart(
        "K → macro-F1 L3 (raw top-1, validation)", ks, series_of("l3_macro_f1"),
        baseline=("none", base_metrics["macro_f1"]) if base_metrics.get("macro_f1") is not None else None,
    )
    time_chart = _svg_bar_chart(
        "K → время обучения MLP, с", ks, series_of("train_seconds"), y_fmt="{:.1f}"
    )
    size_series = []
    for method in dict.fromkeys(chart_runs["method"]):
        sub = chart_runs[chart_runs["method"] == method]
        pts = {}
        for r in sub.itertuples():
            mb = r.model_size_bytes
            if mb is not None and not pd.isna(mb):
                pts[float(r.requested_k)] = mb / (1024 * 1024)
        if pts:
            size_series.append((method, colors.get(method, "#333333"), pts))
    base_mb = None
    if len(base) and base.iloc[0]["model_size_bytes"] is not None and not pd.isna(base.iloc[0]["model_size_bytes"]):
        base_mb = float(base.iloc[0]["model_size_bytes"]) / (1024 * 1024)
    size_chart = _svg_bar_chart(
        "K → размер модели, МБ", ks, size_series, y_fmt="{:.0f}"
    ) if size_series else '<p class="muted">Нет данных о размере модели.</p>'
    base_size_note = (
        f'<p class="muted">Размер модели baseline none: {base_mb:.1f} МБ.</p>' if base_mb is not None else ""
    )

    # Качество ↔ ресурсы
    qr = succ[["run_id", "method", "requested_k", "l3_macro_f1", "delta_macro_f1",
               "train_seconds", "val_inference_seconds", "model_size_bytes", "n_features"]].copy()
    qr["model_mb"] = qr["model_size_bytes"].map(lambda v: None if pd.isna(v) else v / (1024 * 1024))

    # Стоимость сценариев: все плановые прогоны (включая failed/skipped и не
    # сопоставимые с baseline) — бюджет сценария обязан быть виден неполным,
    # а не суммой оставшихся успешных запусков. Разделяемый transform полного
    # TF-IDF (run_id=shared_vectorization) добавляется каждому методу ровно
    # один раз; разделяемая подготовка G×I — только сценариям gradient_x_input,
    # в итоге метода — один раз (без двойного счёта); неизвестные времена —
    # null, а не ноль (см. _scenario_cost_rows).
    scenario_rows, scenario_totals = _scenario_cost_rows(stages_df, runs_df)
    qr_points: list[tuple[str, str, float, float]] = []
    for row in scenario_rows:
        total = row["scenario_total_seconds"]
        f1 = row["l3_macro_f1"]
        if total is not None and f1 is not None and not pd.isna(f1):
            qr_points.append(
                (str(row["run_id"]), colors.get(row["method"], "#333333"), float(total), float(f1))
            )
    if scenario_rows:
        scenario_tbl = _table_html(
            pd.DataFrame(scenario_rows),
            ["run_id", "method", "requested_k", "status", "own_seconds",
             "vectorization_seconds", "shared_seconds",
             "scenario_total_seconds", "l3_macro_f1"],
            formats={
                "own_seconds": lambda v: f"{v:.2f}",
                "vectorization_seconds": lambda v: f"{v:.2f}",
                "shared_seconds": lambda v: f"{v:.2f}",
                "scenario_total_seconds": lambda v: f"{v:.2f}",
                "l3_macro_f1": lambda v: f"{v:.4f}",
            },
            caption=None,
        )
    else:
        scenario_tbl = '<p class="muted">Стоимость сценария недоступна: нет плановых прогонов.</p>'
    if scenario_totals:
        scenario_total_tbl = _table_html(
            pd.DataFrame(scenario_totals),
            ["method", "n_budgets", "n_successful", "complete", "own_seconds",
             "vectorization_seconds", "shared_seconds", "scenario_total_seconds"],
            formats={
                "own_seconds": lambda v: f"{v:.2f}",
                "vectorization_seconds": lambda v: f"{v:.2f}",
                "shared_seconds": lambda v: f"{v:.2f}",
                "scenario_total_seconds": lambda v: f"{v:.2f}",
            },
            caption=None,
        )
    else:
        scenario_total_tbl = '<p class="muted">Итоги по сценариям недоступны: нет плановых прогонов.</p>'
    qr_chart = _svg_scatter_chart(
        "Качество ↔ полная стоимость прогона",
        qr_points,
        x_label="секунды (все разделяемые затраты учтены; итог по методу — в таблице ниже)",
        y_label="macro-F1 L3 (validation)",
    )
    scenario_caption = (
        f"Стоимость прогона: собственные успешные этапы (own_seconds) плюс разделяемый "
        f"transform полного TF-IDF train/validation (vectorization_seconds, "
        f"run_id={SHARED_VECTORIZATION_RUN_ID}), выполненный один раз для всего эксперимента "
        f"и добавленный каждому методу ровно один раз — прогонные этапы vectorize получают "
        f"готовые матрицы, поэтому порядок прогонов на стоимость не влияет. Для прогонов "
        f"gradient_x_input в scenario_total_seconds дополнительно учтена разделяемая подготовка "
        f"атрибуции (shared_seconds, run_id={SHARED_RUN_ID}: обучение attribution MLP и "
        "атрибуция), выполненная один раз: на none и Chi²-сценарии (global, class_aware) она "
        "не распределяется. Итог по сценарию — все плановые бюджеты метода: common-затраты "
        "G×I учтены ровно один раз; суммирование per-run строк завысило бы их в число раз, "
        "равное числу бюджетов. Неполный сценарий (failed/skipped бюджет или неизмеренный "
        "этап) показан явно: complete=false и итог null, а не сумма оставшихся бюджетов; при "
        "загрузке checkpoint'а извне стоимость его обучения неизвестна "
        "(attribution_model_train с временем null), поэтому полный итог G×I — null. "
        "Неизвестные времена — null, а не ноль. Таблицы покрывают все плановые прогоны; "
        "статус сопоставимости с baseline (нужен только для сравнения качества) — в runs.csv."
    )

    # Победители/проигравшие по дельтам per-class. Дельты сравнивают прогон с
    # baseline, поэтому без проверяемых данных о сопоставимости (S04 без
    # сохранённых флагов) блок подавляется, а не показывает непроверенные дельты.
    comparability_verified = (
        "comparable_with_baseline" in succ.columns
        or identity_schema == IDENTITY_SCHEMA_LEGACY
    )
    movers_blocks = []
    pc = per_class_df
    if comparability_verified and len(pc) and baseline_id in set(pc["run_id"]):
        for rid in [r for r in runs_df["run_id"] if r != baseline_id]:
            sub = pc[(pc["run_id"] == rid) & pc["delta_f1"].notna()]
            if not len(sub):
                continue
            top = sub.sort_values("delta_f1", ascending=False).head(5)
            bottom = sub.sort_values("delta_f1").head(5)
            rows = []
            for label, part in (("выигрыш", top), ("проигрыш", bottom)):
                for r in part.itertuples():
                    flag = " ⚠" if r.support <= noisy else ""
                    rows.append({
                        "направление": label, "L3": r.L3 + flag, "support": r.support,
                        "ΔF1": r.delta_f1, "Δrecall": r.delta_recall,
                        "F1 run": r.f1, "F1 baseline": r.baseline_f1,
                    })
            tdf = pd.DataFrame(rows)
            if len(tdf):
                movers_blocks.append(f"<h4>run {_esc(rid)}</h4>" + _table_html(
                    tdf, ["направление", "L3", "support", "ΔF1", "Δrecall", "F1 run", "F1 baseline"],
                    {"ΔF1": lambda v: f"{v:+.4f}", "Δrecall": lambda v: f"{v:+.4f}",
                     "F1 run": lambda v: f"{v:.4f}", "F1 baseline": lambda v: f"{v:.4f}"},
                ))
    movers_html = "".join(movers_blocks)
    if not movers_html:
        reason = (
            "нет сопоставимых с baseline прогонов"
            if comparability_verified
            else "данные о сопоставимости не сохранены в файлах отчёта"
        )
        movers_html = f'<p class="muted">Дельты per-class недоступны: {reason}.</p>'

    failed = runs_df[runs_df["status"] != "success"]
    failed_html = ""
    if len(failed):
        failed_html = "<h3>Неуспешные/пропущенные прогоны</h3>" + _table_html(
            failed[["run_id", "method", "requested_k", "status", "error"]],
            ["run_id", "method", "requested_k", "status", "error"],
        )

    # Секция Gradient × Input: разделяемые этапы (attribution MLP, атрибуция),
    # целевая величина, число атрибутированных примеров, correct/incorrect,
    # пустые строки, распределение вкладов, покрытие классов и стоимость.
    attribution = manifest.get("attribution") if isinstance(manifest, dict) else None
    gxi_html = ""
    if attribution:
        shared_stages = stages_df[stages_df["run_id"] == SHARED_RUN_ID]
        shared_tbl = _table_html(
            shared_stages,
            ["stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes"],
            formats={
                "seconds": lambda v: f"{v:.2f}",
                "peak_rss_bytes": lambda v: f"{v / (1024 * 1024):.0f} МБ",
                "peak_vram_bytes": lambda v: f"{v / (1024 * 1024):.0f} МБ",
            },
        ) if len(shared_stages) else '<p class="muted">Этапы атрибуции не зафиксированы.</p>'
        summary = attribution.get("summary") or {}
        contributions = summary.get("contributions") or {}
        correct = summary.get("correct") or {}
        incorrect = summary.get("incorrect") or {}
        margins = summary.get("margins") or {}
        sampling = attribution.get("sampling") or {}
        checkpoint = attribution.get("checkpoint") or {}

        def _c(key: str) -> Any:
            block = contributions.get(key) or {}
            if not block:
                return "—"
            if key == "negative":
                return f"{block.get('count')} (|sum|={_fmt(block.get('abs_sum'), 4)}, доля={_fmt(block.get('share'), 4)})"
            return f"{block.get('count')} (sum={_fmt(block.get('sum'), 4)}, доля={_fmt(block.get('share'), 4)})"

        rows = [
            ("Цель", attribution.get("target")),
            ("Формула", attribution.get("formula")),
            ("Режим", attribution.get("mode")),
            ("Сэмплинг train", f"{sampling.get('method')} (max_rows={sampling.get('max_rows')}, seed={sampling.get('seed')}), "
                               f"использовано {sampling.get('n_rows_used')} из {sampling.get('n_train_rows_total')}"),
            ("Checkpoint", f"{checkpoint.get('path')} (источник: {checkpoint.get('source')}; "
                           f"sha256 model.pt: {str((checkpoint.get('model.pt') or {}).get('sha256'))[:16]}…)"),
            ("Атрибутировано примеров", summary.get("n_attributed_rows")),
            ("Правильных предсказаний", f"{correct.get('count')} (доля {_fmt(correct.get('fraction'), 4)})"),
            ("Ошибочных предсказаний", f"{incorrect.get('count')} (доля {_fmt(incorrect.get('fraction'), 4)})"),
            ("Непустых TF-IDF-строк", summary.get("n_nonempty_rows")),
            ("Пустых TF-IDF-строк", summary.get("n_empty_rows")),
            ("Положительные вклады", _c("positive")),
            ("Отрицательные вклады", _c("negative")),
            ("Нулевые вклады", f"{(contributions.get('zero') or {}).get('count')} "
                               f"(доля {_fmt((contributions.get('zero') or {}).get('share'), 4)})"),
            ("Margin: mean/std", f"{_fmt(margins.get('mean'), 4)} / {_fmt(margins.get('std'), 4)}"),
            ("Margin: min/max", f"{_fmt(margins.get('min'), 4)} / {_fmt(margins.get('max'), 4)}"),
            ("Классов L3 (с положительными кандидатами)",
             f"{summary.get('n_classes')} ({summary.get('n_classes_with_positive')})"),
        ]
        summary_tbl = _table_html(
            pd.DataFrame(rows, columns=["показатель", "значение"]), ["показатель", "значение"]
        )
        train_info = attribution.get("train_info") or {}
        train_note = ""
        if train_info:
            train_note = (
                f"<p class=\"muted\">Attribution MLP: эпох выполнено {train_info.get('epochs_run')}, "
                f"лучшая {train_info.get('best_epoch')} "
                f"({train_info.get('early_stopping_metric')}={_fmt(train_info.get('best_score'), 4)}). "
                "Одноэпоховый smoke не доказывает полезность G×I; жёсткий порог качества не вводился.</p>"
            )
        error_note = ""
        if attribution.get("error"):
            error_note = f'<p class="warn">Атрибуция завершилась ошибкой: {_esc(attribution["error"])}</p>'
        per_class_counts = attribution.get("per_class_counts") or {}
        coverage_tbl = ""
        if per_class_counts:
            cov = pd.DataFrame(
                [{"L3": k, "n_attributed": int(v)} for k, v in sorted(
                    per_class_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))]
            )
            coverage_tbl = (
                "<h3>Покрытие классов: атрибутированные train-строки по L3</h3>"
                + _table_html(cov, ["L3", "n_attributed"])
            )
        gxi_html = f"""
<h2>Gradient × Input (атрибуция по L3-margin)</h2>
{error_note}
{summary_tbl}
{train_note}
{coverage_tbl}
<h3>Разделяемые этапы атрибуции (run_id={_esc(SHARED_RUN_ID)})</h3>
{shared_tbl}
<p class="caption">Число атрибутированных примеров — по всем train-строкам, включая пустые
TF-IDF-векторы (margin и классификация вычислены для каждой строки, пустые входят в N_c
знаменателя класса); непустые и пустые строки показаны отдельно. Затраты обучения attribution
MLP и самой атрибуции учтены один раз под run_id={_esc(SHARED_RUN_ID)} и относятся только на
сценарии gradient_x_input: ни на none, ни на Chi²-сценарии (global, class_aware) они не
распределяются. При загрузке checkpoint'а извне строка attribution_model_train имеет время
null: стоимость обучения в этом процессе неизвестна, и полный итог G×I честно не заявляется.
В per-run строках отчёта полная стоимость прогона G×I включает эту подготовку; в итоге по
сценарию («Качество ↔ ресурсы») она учтена ровно один раз для всех бюджетов вместе (без
двойного счёта). Общий transform полного TF-IDF train/validation учтён отдельно под
run_id={_esc(SHARED_VECTORIZATION_RUN_ID)} и добавлен каждому методу ровно один раз. Метод
замера: {RSS_SAMPLING_NOTE}</p>
"""

    inputs = exp.get("inputs") or {}
    splits = exp.get("splits") or {}
    lm = exp.get("label_maps") or {}
    proto = exp.get("metric_protocol") or {}
    env = exp.get("environment") or {}
    report_label = "S04" if "s04" in str(manifest.get("schema_version", "")).lower() else "S03"
    gxi_runs = runs_df[runs_df["method"] == GRADIENT_X_INPUT]
    gxi_note = ""
    if len(gxi_runs):
        gxi_note = (
            " · gradient_x_input: " + ", ".join(
                f"{r.run_id}={r.status}" for r in gxi_runs.itertuples()
            )
        )

    runs_tbl = _table_html(
        runs_df,
        ["run_id", "status", "method", "requested_k", "n_char", "n_word", "n_features",
         "l3_macro_f1", "l3_accuracy", "delta_macro_f1", "delta_accuracy",
         "train_seconds", "val_inference_seconds", "model_size_bytes", "error"],
        formats={
            "l3_macro_f1": lambda v: f"{v:.4f}", "l3_accuracy": lambda v: f"{v:.4f}",
            "delta_macro_f1": lambda v: f"{v:+.4f}", "delta_accuracy": lambda v: f"{v:+.4f}",
            "train_seconds": lambda v: f"{v:.1f}", "val_inference_seconds": lambda v: f"{v:.2f}",
            "model_size_bytes": _fmt_mb,
        },
    )
    runs_tbl = runs_tbl.replace("<td>—</td>", "<td class='muted'>—</td>")

    stages_tbl = _table_html(
        stages_df,
        ["run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes"],
        formats={
            "seconds": lambda v: f"{v:.2f}",
            "peak_rss_bytes": lambda v: f"{v / (1024 * 1024):.0f} МБ",
            "peak_vram_bytes": lambda v: f"{v / (1024 * 1024):.0f} МБ",
        },
    )

    qr_tbl = _table_html(
        qr, ["run_id", "method", "requested_k", "l3_macro_f1", "delta_macro_f1",
             "train_seconds", "val_inference_seconds", "model_mb", "n_features"],
        formats={
            "l3_macro_f1": lambda v: f"{v:.4f}",
            "delta_macro_f1": lambda v: f"{v:+.4f}",
            "train_seconds": lambda v: f"{v:.1f}",
            "val_inference_seconds": lambda v: f"{v:.2f}",
            "model_mb": lambda v: f"{v:.1f}",
        },
    )

    corpus = inputs.get("corpus") or {}
    labeled = inputs.get("labeled") or {}
    full_t = inputs.get("full_tfidf") or {}
    versions = env.get("versions") or {}
    version_rows = pd.DataFrame([{"компонент": k, "версия": v} for k, v in versions.items()])

    html_doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{report_label} — агрегированный отчёт по отбору признаков</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #222; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 28px; }} h3 {{ font-size: 15px; }}
table {{ border-collapse: collapse; margin: 10px 0; font-size: 12px; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
th {{ background: #f0f0f0; }}
.caption {{ color: #666; font-size: 12px; }}
.muted {{ color: #888; }}
.warn {{ background: #fff3cd; border: 1px solid #ffecb5; padding: 8px 12px; }}
svg {{ max-width: 100%; border: 1px solid #eee; margin: 8px 0; }}
.chart-title {{ font-size: 14px; font-weight: 600; }}
.grid {{ stroke: #e5e5e5; stroke-width: 1; }}
.axis {{ stroke: #999; stroke-width: 1; }}
.tick {{ font-size: 11px; fill: #555; }}
.baseline {{ stroke: #2ca02c; stroke-width: 1.5; stroke-dasharray: 6 4; }}
.baseline-label {{ fill: #2ca02c; }}
code {{ background: #f5f5f5; padding: 1px 4px; }}
</style>
</head>
<body>
<h1>{report_label} — supervised feature selection: агрегированный отчёт</h1>
<p class="muted">experiment_id: {_esc(exp.get('experiment_id'))} · создан {_esc(manifest.get('date'))} ·
git {_esc(str(exp.get('git_commit'))[:10])} (dirty={_esc(exp.get('git_dirty'))}) · устройство {_esc(env.get('device'))}{gxi_note}</p>

<div class="warn"><strong>Ограничение интерпретации.</strong> ODS/PoC не является оценкой промышленной
применимости: результаты диагностические, оптимальный метод/бюджет здесь не объявляется промышленным решением.</div>

<h2>Сводка по прогонам</h2>
{runs_tbl}
{failed_html}
{comparability_note}

<h2>Качество по бюджетам K</h2>
{macro_chart}
{time_chart}
{size_chart}
{base_size_note}
{gxi_html}

<h2>Качество ↔ ресурсы</h2>
{qr_chart}
{scenario_tbl}
<h3>Итоги по сценариям (все бюджеты метода)</h3>
{scenario_total_tbl}
<p class="caption">{_esc(scenario_caption)}</p>
{qr_tbl}

<h2>Стоимость этапов (selection / refit / training / inference)</h2>
{stages_tbl}
<p class="caption">Метод замера: {RSS_SAMPLING_NOTE} {VRAM_NULL_NOTE} Значения «—» (null) — измерение
отсутствует, а не равно нулю. Подробные примечания — в stages.csv.</p>

<h2>Per-class: сильнейшие сдвиги относительно baseline</h2>
<p class="muted">⚠ — категории с support ≤ {noisy} на validation: их метрики и дельты статистически
неустойчивы, низкий support не является свидетельством качества или его отсутствия.</p>
{movers_html}

<h2>Происхождение данных, инварианты сравнения, воспроизводимость</h2>
<ul>
<li>Метрика: <code>{_esc(proto.get('metric'))}</code>; macro-F1: {_esc(proto.get('macro_f1'))};
zero_division={_esc(proto.get('zero_division'))}; для support=0 метрики и дельты — null.</li>
<li>Инварианты: общие splits и label maps для всех прогонов (seed {_esc((cfg.get('seed')))});
selection только по размеченному train; метки test не участвуют в selection и сравнении;
MLP общий для всех прогонов (отличается только входная размерность).</li>
<li>Корпус: <code>{_esc(corpus.get('path'))}</code>, sha256 <code>{_esc(str(corpus.get('sha256'))[:16])}…</code>,
выбор строк: max_rows={_esc(((corpus.get('row_selection') or {}).get('max_rows')))},
strategy={_esc(((corpus.get('row_selection') or {}).get('strategy')))},
vectorizer_fit_rows={_esc(((corpus.get('row_selection') or {}).get('vectorizer_fit_rows')))},
seed={_esc(((corpus.get('row_selection') or {}).get('seed')))}, source_id={_esc(((corpus.get('row_selection') or {}).get('source_id')))}.</li>
<li>Размеченные данные: <code>{_esc(labeled.get('path'))}</code>, sha256 <code>{_esc(str(labeled.get('sha256'))[:16])}…</code>.</li>
<li>Полный TF-IDF: <code>{_esc(full_t.get('path'))}</code>, версия {_esc(full_t.get('artifact_version'))},
словарь char/word/total = {_esc(((full_t.get('vocab') or {}).get('n_char')))}/{_esc(((full_t.get('vocab') or {}).get('n_word')))}/{_esc(((full_t.get('vocab') or {}).get('n_total')))}.</li>
<li>Сплиты: train/val/test =
{_esc((splits.get('train') or {}).get('bytes'))}/{_esc((splits.get('val') or {}).get('bytes'))}/{_esc((splits.get('test') or {}).get('bytes'))} байт
(sha256 в manifest.json); label maps: {len(lm.get('l3_train_counts') or {})} L3 (хэш состава — в manifest.json).</li>
<li>Отчёт построен автоматически из runs.csv, per_class.parquet, stages.csv, manifest.json;
повторная генерация без обучения: <code>python scripts/run_{_esc(report_label.lower())}_experiment.py report --output {_esc(output_dir)}</code>.</li>
</ul>
{_table_html(version_rows, ['компонент', 'версия'])}
<p class="muted">Ограничения: {'; '.join(_esc(x) for x in (exp.get('limitations') or []))}</p>
</body>
</html>"""
    return html_doc


def _exp_from_manifest(manifest: dict) -> dict:
    """Восстанавливает experiment.json-подобный контекст из manifest.json
    (при регенерации отчёта из сохранённых файлов без runs/ и experiment.json)."""
    configs = manifest.get("configs") or {}
    return {
        "experiment_id": manifest.get("experiment_id"),
        "created_at": manifest.get("experiment_created_at"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "config": configs.get("experiment"),
        "environment": {
            "device": manifest.get("device"),
            "versions": manifest.get("versions") or {},
            "platform": manifest.get("platform"),
            "cpu_count": manifest.get("cpu_count"),
        },
        "inputs": manifest.get("inputs") or {},
        "splits": manifest.get("splits") or {},
        "label_maps": manifest.get("label_maps") or {},
        "metric_protocol": manifest.get("metric_protocol") or {},
        "attribution": manifest.get("attribution"),
        "limitations": manifest.get("resource_limitations") or [],
        "planned_runs": manifest.get("planned_runs") or [],
    }


def _load_report_files(report_dir: str):
    """Читает четыре файла отчёта (runs.csv, per_class.parquet, stages.csv,
    manifest.json). Возвращает (exp, runs_df, per_class_df, stages_df, manifest)
    или None, если файлов нет. Бросает ExperimentError при повреждении файлов."""
    paths = [os.path.join(report_dir, n) for n in
             ("runs.csv", "per_class.parquet", "stages.csv", "manifest.json")]
    if not all(os.path.isfile(p) for p in paths):
        return None
    try:
        runs_df = pd.read_csv(paths[0])
        per_class_df = pd.read_parquet(paths[1])
        stages_df = pd.read_csv(paths[2])
        with open(paths[3], encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise ExperimentError(
            f"Файлы отчёта в {report_dir!r} прочитать не удалось ({e}): восстановление невозможно, "
            "удалите повреждённые файлы и пересоберите отчёт из experiment.json/runs"
        ) from e
    if not isinstance(manifest, dict):
        raise ExperimentError(f"manifest.json в {report_dir!r}: ожидался объект")
    if "requested_k" in runs_df.columns:
        runs_df["requested_k"] = pd.to_numeric(runs_df["requested_k"], errors="coerce").astype("Int64")
    for col in ("n_features", "n_char", "n_word", "n_train", "n_val", "n_classes_eval",
                "seed", "model_size_bytes"):
        if col in runs_df.columns:
            runs_df[col] = pd.to_numeric(runs_df[col], errors="coerce").astype("Int64")
    # Сопоставимость с baseline — только сохранённые проверенные значения.
    # Колонка runs.csv: недопустимые значения трактуются как False (fail closed).
    if "comparable_with_baseline" in runs_df.columns:
        runs_df["comparable_with_baseline"] = runs_df["comparable_with_baseline"].map(
            lambda v: False if pd.isna(v) else str(v).strip().lower() == "true"
        )
    else:
        # Колонки нет (таблицы сохранены старым кодом): восстанавливаем флаги
        # из manifest.json, где они зафиксированы. Если их нет и там — данных
        # для проверки нет вовсе; решение о толковании принимает _html_report
        # по схеме отчёта, и отсутствующие данные верифицированными не считаются.
        flags = {
            p.get("run_id"): p.get("comparable_with_baseline")
            for p in (manifest.get("planned_runs") or [])
            if isinstance(p, dict) and p.get("run_id") is not None
            and "comparable_with_baseline" in p
        }
        if flags:
            runs_df["comparable_with_baseline"] = [
                bool(flags.get(rid)) for rid in runs_df["run_id"]
            ]
    return _exp_from_manifest(manifest), runs_df, per_class_df, stages_df, manifest


def build_report(output_dir: str) -> dict[str, str]:
    """Строит пять файлов агрегированного отчёта из сохранённых результатов.

    Не обучает модели и не пересчитывает метрики. Два источника (приоритет —
    первый):

    1. ``experiment.json`` + ``runs/<run_id>/run_result.json`` — свежие
       результаты прогонов; четыре файла отчёта (без HTML) перезаписываются,
       HTML строится из тех же данных.
    2. Только файлы отчёта (``report/runs.csv``, ``report/per_class.parquet``,
       ``report/stages.csv``, ``report/manifest.json``) — полная регенерация
       HTML без ``runs/`` и ``experiment.json``; источниковые файлы не
       изменяются.

    Повторный вызов перезаписывает ``report.html``. Неполный/поломанный прогон
    не ломает генерацию: неуспешные прогоны и этапы отражаются статусами.
    """
    report_dir = os.path.join(output_dir, REPORT_DIRNAME)
    exp_path = os.path.join(output_dir, "experiment.json")
    if os.path.isfile(exp_path):
        exp = _read_json(exp_path)
        planned = exp.get("planned_runs") or []
        # Семантика сравнения — по явной схеме версии эксперимента: s03-* —
        # legacy-отчёты S03 (дельты по val/labels), всё остальное — строгая
        # семантика S04 (все семь непустых идентификаторов, fail closed).
        identity_schema = _identity_schema_for_experiment(exp.get("schema_version"))
        l3_labels = (exp.get("label_maps") or {}).get("l3_labels")
        if not (isinstance(l3_labels, list) and all(isinstance(x, str) for x in l3_labels)):
            l3_labels = None
        results = {p["run_id"]: _load_run_result(output_dir, p["run_id"], l3_labels) for p in planned}

        baseline = results.get(BASELINE_RUN_ID)
        runs_df = _runs_frame(planned, results, baseline, identity_schema)
        per_class_df = _per_class_frame(planned, results, baseline, identity_schema)
        stages_df = _stages_frame(planned, results, output_dir)
        manifest = build_manifest(output_dir, exp, runs_df, identity_schema)

        os.makedirs(report_dir, exist_ok=True)
        runs_df.to_csv(os.path.join(report_dir, "runs.csv"), index=False)
        per_class_df.to_parquet(os.path.join(report_dir, "per_class.parquet"), index=False)
        stages_df.to_csv(os.path.join(report_dir, "stages.csv"), index=False)
        _write_json(manifest, os.path.join(report_dir, "manifest.json"))
    else:
        loaded = _load_report_files(report_dir)
        if loaded is None:
            raise ExperimentError(
                f"Ни experiment.json, ни файлы отчёта (runs.csv, per_class.parquet, stages.csv, "
                f"manifest.json) не найдены в {output_dir!r}: сначала выполните run или сохраните "
                "файлы отчёта"
            )
        exp, runs_df, per_class_df, stages_df, manifest = loaded

    report_html = os.path.join(report_dir, "report.html")
    doc = _html_report(output_dir, exp, runs_df, per_class_df, stages_df, manifest, report_dir)
    with open(report_html, "w", encoding="utf-8") as f:
        f.write(doc)
    log.info("Отчёт: %s (run %d: success %d)", report_dir, len(runs_df),
             int((runs_df["status"] == "success").sum()))
    return {
        "report_dir": report_dir,
        "report_html": report_html,
        "runs_csv": os.path.join(report_dir, "runs.csv"),
        "per_class_parquet": os.path.join(report_dir, "per_class.parquet"),
        "stages_csv": os.path.join(report_dir, "stages.csv"),
        "manifest_json": os.path.join(report_dir, "manifest.json"),
    }
