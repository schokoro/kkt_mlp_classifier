"""S06 (D06–D08): сравнительный контур Gate 2–3 — ранжирование, refit, 30 paired runs.

Реализует сравнение supervised feature selection на замороженном протоколе S06
поверх результатов Gate 0 (``outputs/S06/splits``, см. ``kkt_clf.s06_split``)
и Gate 1 (``outputs/S06/shared/training_protocol.json`` +
``outputs/S06/runs/baseline_full``, см. ``kkt_clf.s06_baseline``). Методология
отбора не меняется и целиком делегируется существующим контурам S03/S04:

- ``global`` и ``class_aware`` Chi² — :func:`kkt_clf.selection.select_features`
  (рейтинги полные, float64, на диске; положительные OvR-ассоциации и
  round-robin у class_aware — без изменений);
- ``gradient_x_input`` — :func:`kkt_clf.attribution.compute_margin_attributions`
  и :func:`kkt_clf.attribution.build_gxi_ratings` на опорном train-only
  checkpoint Gate 1 (``reference_seed``, margin истинной L3 к сильнейшему
  конкуренту, положительные вклады; отрицательные — только диагностика);
- fixed-vocabulary refit — :func:`kkt_clf.vectorizer.refit_selected_vectorizer`
  на **том же** фактическом снимке корпуса, что полный TF-IDF S02
  (проверка sha256 корпуса, инвариант IDF, L2-нормировка, round-trip).

Границы протокола:

- Рейтинги считаются **только по ``outer_train``**; метки comparison_val и
  sealed_test в отбор не попадают (fail closed в используемых контрактах).
  Рейтинг каждого метода вычисляется один раз и сохраняется с контрольными
  суммами в ``shared/rankings/`` / ``shared/attribution/``; бюджеты
  K = 50k/100k/200k (char_share=0.75) применяются из сохранённых рейтингов
  повторно, без пересчёта оценок. Выбранные словари фиксируются для всех seed,
  чтобы измерять вариабельность обучения MLP, а не смешивать её с
  вариабельностью рейтингов; ограничение дизайна зафиксировано в отчёте.
- Refit выполняется **один раз на метод×K** и переиспользуется тремя seed
  обучения; refit по seed не повторяется.
- Финальные MLP обучаются с нуля на ``outer_train`` ровно на замороженное в
  Gate 1 число эпох с секциями model/loss/train замороженного протокола
  (``restore_best=False``); конфигурация ``none`` — переиспользование трёх
  baseline checkpoint Gate 1 (повторное обучение не выполняется), их метрики
  пересчитываются тем же протоколом оценки и сверяются с сохранёнными Gate 1.
- Оценка — raw top-1 L3 без отказа на ``comparison_val`` по фиксированной
  L3-вселенной label maps; парные Δ считаются строго к ``none`` того же seed.
  ``sealed_test`` не читается и не оценивается нигде (никакого inference/метрик);
  используются только структурные сведения manifest, как в Gate 1.

Устойчивость к сбоям и отсутствие перезаписей: каждый запуск фиксирует
собственный результат (``runs/<slug>/run_result_seed_<s>.json``), предсказания
(``runs/predictions/<run_id>.parquet``) и строки этапов (``runs/stages.csv``,
только дополнение); уже записанные файлы не пересчитываются и не
перезаписываются. **Resume отключён (fail closed):** обнаружение ранее
сохранённых рейтингов, атрибуции, артефакта отбора, refit-результата или
run_result со статусом ``success`` — явный ``S06ComparisonError`` (provenance
не подтверждён; используйте новый ``output_root``). Существующие результаты
со статусами ``failed``/``pending``/``skipped`` не считаются успешными,
не перезаписываются и в ``n_done`` не входят. Fresh проход
rankings → refit → train в новом каталоге использует только ресурсы,
созданные этим проходом; удаление и перезапись чужих данных не выполняются.
Плановые 30 runs фиксируются заранее в ``runs/planned.json``; каждый этап до
любых записей требует неизменяемый план и проверенные input fingerprints.
Пропуски и сбои отражаются статусами ``pending/success/failed/skipped`` с
причинами, а не маскируются. Агрегированный отчёт Gate 4 строится отдельным контуром
(``kkt_clf.s06_report``) из сохранённых файлов без обучения и inference.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .artifact import environment_versions, load_artifact, save_artifact
from .attribution import (
    ATTRIBUTION_PROVENANCE_FILENAME,
    build_gxi_ratings,
    compute_margin_attributions,
    feature_contract_fingerprint,
    select_features_gradient_x_input,
    verify_attribution_checkpoint,
)
from .config import HEADS, LossConfig, ModelConfig, TrainConfig
from .experiment import _Measurement, _l3_val_metrics, _sha256_file, _sha256_json
from .labels import LEVELS, LabelMaps
from .model import SparseMLP
from .reject import RejectRule
from .s05_baseline import (
    load_part,
    verify_full_tfidf,
    _encode_labels,
    _transform,
)
from .s06_baseline import (
    S06Gate1Config,
    load_training_protocol,
    validate_protocol_against_inputs,
    verify_gate0_split,
)
from .selection import (
    RATINGS_FORMAT_VERSION,
    BranchSelection,
    SelectionArtifactError,
    SelectionConfig,
    SelectionError,
    SelectionResult,
    _rank_topk,
    _ratings_declared_names,
    branch_budgets,
    load_selection_artifact,
    roundrobin_select_from_rank_files,
    save_selection_artifact,
    select_features,
    validate_ratings_dir,
)
from .train import compute_class_weights, predict, set_seed, train_model
from .vectorizer import LoadedVectorizer, load_vectorizer_artifact, refit_selected_vectorizer

log = logging.getLogger(__name__)

S06_COMPARISON_SCHEMA = "s06-comparison@1"

METHOD_NONE = "none"
METHOD_GLOBAL = "global"
METHOD_CLASS_AWARE = "class_aware"
METHOD_GXI = "gradient_x_input"
SELECTION_METHODS = (METHOD_GLOBAL, METHOD_CLASS_AWARE, METHOD_GXI)

DEFAULT_BUDGETS = (50_000, 100_000, 200_000)
DEFAULT_SEEDS = (42, 43, 44)
# Заранее назначенный опорный seed Gate 1 (selected checkpoint).
GATE1_REFERENCE_SEED = 42
DEFAULT_CHAR_SHARE = 0.75

# Этапы контура и канонический порядок для дописывания skipped-строк.
STAGE_VECTORIZE_OUTER = "vectorize_outer_train"
STAGE_RANKING = "ranking"
STAGE_ATTRIBUTION = "attribution"
STAGE_SELECTION = "selection"
STAGE_REFIT = "selected_tfidf_refit"
STAGE_VECTORIZE_CONFIG = "vectorize_config"
STAGE_MLP_TRAIN = "mlp_train"
STAGE_VAL_INFERENCE = "comparison_val_inference"

STAGES_COLUMNS = [
    "run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
    "artefact_bytes", "measurement_note",
]

RUN_RESULT_SCHEMA = "s06-run-result@1"

PROTOCOL_NOTES = {
    "rankings_train_only": (
        "рейтинги всех методов вычислены только по outer_train полного TF-IDF; "
        "метки comparison_val/sealed_test не использовались (fail closed в "
        "контрактах S03/S04)"
    ),
    "gxi_reference": (
        "G×I-атрибуция выполнена на опорном checkpoint Gate 1 (reference_seed, "
        "train-only правило выбора): margin истинной L3 к сильнейшему конкуренту, "
        "положительные вклады; отрицательные — только в диагностике; ошибочно "
        "классифицированные train-примеры сохранены"
    ),
    "budget_reuse": (
        "полные рейтинги каждого метода вычислены один раз и сохранены с "
        "контрольными суммами; бюджеты 50k/100k/200k (char_share=0.75) применены "
        "из сохранённых рейтингов без пересчёта оценок; выбранные словари "
        "зафиксированы для всех seed (вариабельность рейтингов не смешивается "
        "с вариабельностью обучения MLP — ограничение дизайна)"
    ),
    "refit_once": (
        "refit TF-IDF с фиксированными словарями выполнен один раз на метод×K "
        "на том же снимке корпуса, что полный TF-IDF S02; между seed общий "
        "refit-артефакт, повторный fit не выполнялся"
    ),
    "frozen_protocol": (
        "финальные MLP обучены с нуля на outer_train ровно на замороженное в "
        "Gate 1 число эпох с секциями model/loss/train замороженного протокола "
        "(restore_best=False, веса финальной эпохи); comparison_val не "
        "использовался для обучения, эпох, checkpoint или ранжирования"
    ),
    "paired": (
        "парные Δ считаются строго к полному baseline (none) того же train seed; "
        "различия размеров входного слоя не предполагают идентичности начальных "
        "весов; три seed на одном split — не доверительный интервал"
    ),
    "sealed_test": (
        "sealed_test не использовался ни для inference, ни для метрик; обращений "
        "к его строкам в контуре нет (только структурные sha256/n_rows manifest)"
    ),
}


class S06ComparisonError(RuntimeError):
    """Ошибка сравнительного контура S06; существующие файлы не перезаписываются."""


@dataclass
class S06ComparisonConfig:
    """Конфигурация сравнительного контура Gate 2–3 S06.

    Все пути по умолчанию согласованы с контрактами Gate 0/1. Бюджеты и seed
    настраиваются (для изолированных smoke на фикстурах); плановый прогон S06
    использует budgets=(50_000, 100_000, 200_000), char_share=0.75,
    seeds=(42, 43, 44), reference_seed=42.
    """

    output_root: str = "outputs/S06"
    splits_manifest: str | None = None
    corpus_path: str = "data/corpus.pqt"
    full_tfidf_path: str = "outputs/S02/tfidf_1m"
    methods: tuple[str, ...] = SELECTION_METHODS
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    char_share: float = DEFAULT_CHAR_SHARE
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    reference_seed: int = 42
    device: str = "auto"
    features_chunk_rows: int = 50_000
    attribution_batch_size: int = 4096
    eval_batch_size: int = 8192
    # Пасsthrough настроек Gate 1 (должны совпадать с вызовом Gate 1:
    # validate_protocol_against_inputs воспроизводит замороженные секции из них).
    preset: str = "ods_light"
    model: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    sweep: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.splits_manifest is None:
            self.splits_manifest = os.path.join(self.output_root, "splits", "manifest.json")
        self.methods = tuple(self.methods)
        self.budgets = tuple(int(k) for k in self.budgets)
        self.seeds = tuple(int(s) for s in self.seeds)

    # --- Пути контракта -----------------------------------------------------
    @property
    def shared_dir(self) -> str:
        return os.path.join(self.output_root, "shared")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.output_root, "runs")

    @property
    def baseline_dir(self) -> str:
        return os.path.join(self.runs_dir, "baseline_full")

    @property
    def protocol_path(self) -> str:
        return os.path.join(self.shared_dir, "training_protocol.json")

    @property
    def tfidf_ref_path(self) -> str:
        return os.path.join(self.shared_dir, "full_tfidf_ref.json")

    @property
    def planned_path(self) -> str:
        return os.path.join(self.runs_dir, "planned.json")

    @property
    def stages_path(self) -> str:
        return os.path.join(self.runs_dir, "stages.csv")

    def rankings_dir(self, method: str) -> str:
        return os.path.join(self.shared_dir, "rankings", method)

    def attribution_dir(self) -> str:
        return os.path.join(self.shared_dir, "attribution")

    def selection_dir(self, slug: str) -> str:
        return os.path.join(self.shared_dir, "selection", slug)

    def config_dir(self, slug: str) -> str:
        return os.path.join(self.runs_dir, slug)

    def run_result_path(self, slug: str, seed: int) -> str:
        return os.path.join(self.config_dir(slug), f"run_result_seed_{seed}.json")

    def predictions_path(self, run_id: str) -> str:
        return os.path.join(self.runs_dir, "predictions", f"{run_id}.parquet")

    # --- Служебные ----------------------------------------------------------
    def slug(self, method: str, budget: int | None) -> str:
        return METHOD_NONE if method == METHOD_NONE else f"{method}_k{budget}"

    def run_id(self, slug: str, seed: int) -> str:
        return f"{slug}_seed{seed}"

    def planned_configs(self) -> list[dict[str, Any]]:
        """10 конфигураций: none + методы × бюджеты (порядок стабилен)."""
        configs = [{"slug": METHOD_NONE, "method": METHOD_NONE, "budget": None}]
        for method in self.methods:
            for budget in self.budgets:
                configs.append({"slug": self.slug(method, budget), "method": method, "budget": int(budget)})
        return configs

    def gate1_config(self) -> S06Gate1Config:
        """Согласованная обёртка Gate 1 для read-only проверок (код Gate 1 не меняется)."""
        return S06Gate1Config(
            output_root=self.output_root,
            splits_manifest=str(self.splits_manifest),
            full_tfidf_path=self.full_tfidf_path,
            corpus_path=self.corpus_path,
            preset=self.preset,
            model=dict(self.model),
            loss=dict(self.loss),
            train=dict(self.train),
            sweep=[dict(c) for c in self.sweep],
            seeds=list(self.seeds),
            reference_seed=self.reference_seed,
            features_chunk_rows=self.features_chunk_rows,
            device=self.device,
        )

    def validate(self) -> None:
        bad = [m for m in self.methods if m not in SELECTION_METHODS]
        if bad:
            raise S06ComparisonError(f"methods: неподдерживаемые методы {bad}, ожидаются {SELECTION_METHODS}")
        if len(set(self.methods)) != len(self.methods):
            raise S06ComparisonError(f"methods содержит дубликаты: {self.methods}")
        if not self.budgets or any(k <= 0 for k in self.budgets):
            raise S06ComparisonError(f"budgets: положительные целые, получено {self.budgets}")
        if len(set(self.budgets)) != len(self.budgets):
            raise S06ComparisonError(f"budgets содержат дубликаты: {self.budgets}")
        if not 0.0 < self.char_share < 1.0:
            raise S06ComparisonError(
                f"char_share должен быть строго внутри (0, 1) (обе ветки обязаны получить "
                f"ненулевой бюджет), получено {self.char_share}"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise S06ComparisonError(f"seeds должны быть различны: {self.seeds}")
        if self.reference_seed not in self.seeds:
            raise S06ComparisonError(f"reference_seed={self.reference_seed} должен входить в seeds={self.seeds}")


# ---------------------------------------------------------------------------
# Мелкие утилиты (без побочных эффектов)
# ---------------------------------------------------------------------------


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _dir_size(path: str) -> int | None:
    if not os.path.isdir(path):
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                return None
    return total


def _check_new(paths: Iterable[str]) -> None:
    """Отказ до любых записей, если какой-либо целевой путь уже существует."""
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise S06ComparisonError(
            "Существующие outputs не перезаписываются; удалите/переименуйте вручную: "
            + ", ".join(existing)
        )


def _atomic_write_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _append_stage_rows(path: str, rows: list[dict[str, Any]]) -> None:
    """Дописывает строки этапов в stages.csv (файл создаётся с заголовком)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame = pd.DataFrame(rows, columns=STAGES_COLUMNS)
    header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=header, index=False)


def _stage_recorded(cfg: "S06ComparisonConfig", run_id: str, stage: str) -> bool:
    """Есть ли уже строка этапа (любого статуса) в stages.csv — защита от
    повторной записи shared-строки этапа внутри одного прохода (несколько seed
    одной конфигурации делят общий vectorize-этап)."""
    if not os.path.exists(cfg.stages_path):
        return False
    try:
        frame = pd.read_csv(cfg.stages_path, usecols=["run_id", "stage"])
    except Exception:
        return False
    return bool(((frame["run_id"] == run_id) & (frame["stage"] == stage)).any())


def _git_info() -> tuple[str | None, bool | None]:
    import subprocess

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
            return out.stdout.strip() or None
        except Exception:
            return None

    commit = _run("rev-parse", "HEAD")
    dirty = None
    if commit:
        status = _run("status", "--porcelain")
        dirty = bool(status)
    return commit, dirty


def _full_tfidf_id(loaded: LoadedVectorizer) -> str | None:
    return (loaded.meta.get("files") or {}).get("features.joblib", {}).get("sha256")


# ---------------------------------------------------------------------------
# Проверка предусловий Gate 0/1 (read-only поверх их контрактов)
# ---------------------------------------------------------------------------


def _verify_gate1_baseline(
    cfg: "S06ComparisonConfig",
    manifest: dict[str, Any],
    verification: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Gate 1 (fail closed): три успешно обученных baseline против majority.

    Проверяет до любого ранжирования/refit/обучения:

    - по ``runs.csv`` Gate 1 каждый seed из **замороженного протокола**
      (исходный набор seed Gate 1; план S06 может использовать лишь
      проверенное подмножество) обучен и оценён со статусом ``success``;
    - каждый seed превосходит train-only majority (``majority_baseline.json``,
      macro-F1 на comparison_val): majority-контроль — условие остановки
      перед 9×3 selected runs;
    - baseline-прогон привязан к текущим входам: ``manifest.json`` baseline
      (sha256 сплитов, corpus, full_tfidf_id, label maps, fixed_epochs) и
      per-seed ``meta.json``/``label_maps.json`` checkpoint (seed, число эпох,
      restored_best=False, fingerprints сплитов и TF-IDF, label maps);
    - опорный checkpoint Gate 1 подтверждён round-trip
      (``selected_checkpoint.json``).

    Конфигурация связывается отдельно: ``validate_protocol_against_inputs``
    воспроизводит замороженные секции из текущего cfg до этой проверки.
    Все чтения — JSON/CSV, веса моделей не загружаются.
    """
    if not os.path.isdir(cfg.baseline_dir):
        raise S06ComparisonError(
            f"baseline-прогон Gate 1 не найден: {cfg.baseline_dir!r}. Парные сравнения "
            "требуют завершённого Gate 1 (D05/D06 value_dod)."
        )
    proto_cfg = protocol.get("config") or {}
    proto_seeds = proto_cfg.get("seeds")
    if not proto_seeds:
        raise S06ComparisonError(
            "замороженный протокол Gate 1 не содержит config.seeds: проверка Gate 1 невозможна"
        )
    proto_seeds = [int(s) for s in proto_seeds]
    # Gate 1 fail closed: замороженный протокол обязан содержать именно полный
    # набор seed Gate 1 (42/43/44). План S06 может быть подмножеством, но
    # протокол с урезанным набором seed (или с посторонними seed) не является
    # доказательством завершённого Gate 1 — отказ, а не валидация по подмножеству.
    if set(proto_seeds) != set(DEFAULT_SEEDS):
        raise S06ComparisonError(
            f"замороженный протокол Gate 1 содержит seeds {sorted(proto_seeds)}, "
            f"ожидался полный набор {sorted(DEFAULT_SEEDS)}: урезанный протокол "
            "не подтверждает Gate 1, отказ."
        )
    extra = [int(s) for s in cfg.seeds if int(s) not in proto_seeds]
    if extra:
        raise S06ComparisonError(
            f"план S06 использует seed {extra}, отсутствующие в замороженном протоколе "
            f"Gate 1 {proto_seeds}: gate валидируется по seed протокола"
        )

    runs_path = os.path.join(cfg.baseline_dir, "runs.csv")
    if not os.path.isfile(runs_path):
        raise S06ComparisonError(
            f"runs.csv baseline Gate 1 не найден: {runs_path!r}. Незавершённый baseline "
            "оставляет только каталог: gate не пройден, сравнительная сетка не запускается."
        )
    frame = pd.read_csv(runs_path)
    maj_path = os.path.join(cfg.baseline_dir, "majority_baseline.json")
    if not os.path.isfile(maj_path):
        raise S06ComparisonError(
            f"majority-контроль Gate 1 отсутствует: {maj_path!r}. Превосходство baseline "
            "над train-only majority не подтверждено — gate не пройден."
        )
    with open(maj_path, encoding="utf-8") as f:
        majority = json.load(f)
    maj_f1 = majority.get("l3_macro_f1")
    if maj_f1 is None:
        raise S06ComparisonError(
            f"{maj_path} не содержит l3_macro_f1: majority-контроль не подтверждён, отказ."
        )
    maj_f1 = float(maj_f1)

    # --- Привязка baseline-прогона к текущим split/corpus/TF-IDF/label maps.
    man_path = os.path.join(cfg.baseline_dir, "manifest.json")
    if not os.path.isfile(man_path):
        raise S06ComparisonError(
            f"manifest.json baseline Gate 1 отсутствует: {man_path!r}. Происхождение "
            "baseline не доказано — gate не пройден."
        )
    with open(man_path, encoding="utf-8") as f:
        bman = json.load(f)
    if int(bman.get("fixed_epochs") or -1) != int(protocol.get("fixed_epochs") or -2):
        raise S06ComparisonError(
            "fixed_epochs baseline-прогона не совпадает с замороженным протоколом Gate 1"
        )
    bman_config_id = bman.get("config_id")
    if not bman_config_id:
        raise S06ComparisonError(
            "manifest.json baseline Gate 1 не содержит config_id: привязка прогонов "
            "к конфигурации не доказана, отказ."
        )

    gate: dict[str, Any] = {"seeds": {}, "majority_l3_macro_f1": maj_f1}
    for seed in proto_seeds:
        rows = frame[frame["run_id"] == f"seed_{seed}"]
        if len(rows) != 1:
            raise S06ComparisonError(
                f"runs.csv Gate 1: нет уникальной строки seed_{seed} (найдено {len(rows)})"
            )
        row = rows.iloc[0].to_dict()
        if row.get("status") != "success":
            raise S06ComparisonError(
                f"Gate 1 baseline seed {seed}: status={row.get('status')!r} "
                f"({row.get('reason')}) — gate не пройден, сравнительная сетка не запускается"
            )
        # Статус success не отменяет сверки статистик с manifest: config_id и
        # full_tfidf_id каждого прогона обязаны совпадать с зафиксированным
        # manifest/current verification.
        if not row.get("config_id") or row.get("config_id") != bman_config_id:
            raise S06ComparisonError(
                f"Gate 1 baseline seed {seed}: config_id={row.get('config_id')!r} не совпадает "
                f"с manifest baseline ({bman_config_id!r}): подмена конфигурации недопустима"
            )
        row_tfidf = row.get("full_tfidf_id")
        if not row_tfidf or row_tfidf != verification.get("full_tfidf_id"):
            raise S06ComparisonError(
                f"Gate 1 baseline seed {seed}: full_tfidf_id прогона не совпадает с "
                "проверенным текущим full TF-IDF"
            )
        f1 = row.get("l3_macro_f1")
        try:
            f1_val = float(f1)
        except (TypeError, ValueError):
            f1_val = float("nan")
        if f1 is None or f1_val != f1_val or not f1_val > maj_f1:
            raise S06ComparisonError(
                f"Gate 1 baseline seed {seed}: macro-F1={f1} не превосходит train-only "
                f"majority macro-F1={maj_f1} (или метрика пуста/NaN) — gate не пройден"
            )
        ckpt_dir = row.get("checkpoint_path")
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            raise S06ComparisonError(
                f"Gate 1 baseline seed {seed}: checkpoint не найден: {ckpt_dir!r}"
            )
        gate["seeds"][seed] = row

    # --- Привязка к текущим split/corpus/TF-IDF/label maps (продолжение).
    inputs = bman.get("inputs") or {}
    # Ни один обязательный fingerprint не может быть пустым: missing с обеих
    # сторон — не совпадение provenance, а отсутствие доказательства.
    for name, part in manifest["parts"].items():
        cur_sha = part.get("sha256")
        base_sha = ((inputs.get("splits") or {}).get(name) or {}).get("sha256")
        if not cur_sha or not base_sha or cur_sha != base_sha:
            raise S06ComparisonError(
                f"baseline Gate 1 зафиксирован для другого сплита {name!r} (или fingerprint "
                "пуст с одной из сторон): подмена split между Gate 1 и сравнительным "
                "контуром недопустима"
            )
    cur_tfidf = verification.get("full_tfidf_id")
    base_tfidf = (inputs.get("full_tfidf") or {}).get("full_tfidf_id")
    if not cur_tfidf or not base_tfidf or cur_tfidf != base_tfidf:
        raise S06ComparisonError(
            "baseline Gate 1 относится к другому full TF-IDF, чем проверенный текущий "
            "артефакт (или full_tfidf_id пуст с одной из сторон)"
        )
    cur_corpus = (verification.get("corpus_fingerprint") or {}).get("sha256")
    base_corpus = (inputs.get("corpus") or {}).get("sha256")
    if not cur_corpus or not base_corpus or cur_corpus != base_corpus:
        raise S06ComparisonError(
            "baseline Gate 1 относится к другому корпусу, чем проверенный текущий "
            "fingerprint (или corpus fingerprint пуст с одной из сторон)"
        )
    cur_maps = manifest.get("label_maps") or {}
    base_maps = bman.get("label_maps") or {}
    if not cur_maps or not base_maps:
        raise S06ComparisonError(
            "label maps baseline Gate 1 не совпадают с текущими label maps сплита "
            "(или пусты с одной из сторон)"
        )
    if "l3_universe_sha256" in base_maps or "n_classes" in base_maps:
        # Контрактная схема baseline S05/S06: полный словарь maps свёрнут в
        # канонический sha256 + число классов по уровням.
        expected_sha = _sha256_json(cur_maps)
        recorded_sha = base_maps.get("l3_universe_sha256")
        n_classes = base_maps.get("n_classes") or {}
        expected_counts = {lvl: len(cur_maps.get(lvl) or []) for lvl in LEVELS}
        recorded_counts = {lvl: int(n_classes.get(lvl, -1)) for lvl in LEVELS}
        if not recorded_sha or recorded_sha != expected_sha or recorded_counts != expected_counts:
            raise S06ComparisonError(
                "label maps baseline Gate 1 (l3_universe_sha256/n_classes) не совпадают "
                "с текущими label maps сплита"
            )
    elif cur_maps != base_maps:
        raise S06ComparisonError(
            "label maps baseline Gate 1 не совпадают с текущими label maps сплита"
        )
    # --- Per-seed checkpoint meta (read-only JSON): seed/эпохи/fingerprints.
    for seed, row in gate["seeds"].items():
        ckpt_dir = row["checkpoint_path"]
        meta_path = os.path.join(ckpt_dir, "meta.json")
        maps_path = os.path.join(ckpt_dir, "label_maps.json")
        if not os.path.isfile(meta_path) or not os.path.isfile(maps_path):
            raise S06ComparisonError(
                f"checkpoint Gate 1 seed {seed}: meta.json/label_maps.json отсутствуют "
                f"в {ckpt_dir!r}: provenance недоказан, отказ."
            )
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("seed", -1)) != seed:
            raise S06ComparisonError(f"checkpoint meta seed {seed}: поле seed не совпадает")
        train_meta = meta.get("train") or {}
        if int(train_meta.get("epochs_completed", -1)) != int(protocol.get("fixed_epochs") or -2):
            raise S06ComparisonError(
                f"checkpoint Gate 1 seed {seed}: epochs_completed не равен fixed_epochs протокола"
            )
        if bool(train_meta.get("restored_best")):
            raise S06ComparisonError(
                f"checkpoint Gate 1 seed {seed}: restored_best=True, протоколом требуется False"
            )
        fp = meta.get("train_fingerprints") or {}
        fp_outer = fp.get("outer_train_sha256")
        fp_monitor = fp.get("train_monitor_sha256")
        if (not fp_outer or not fp_monitor
                or fp_outer != manifest["parts"]["outer_train"]["sha256"]
                or fp_monitor != manifest["parts"]["train_monitor"]["sha256"]):
            raise S06ComparisonError(
                f"checkpoint Gate 1 seed {seed}: fingerprints сплитов не совпадают с текущим "
                "manifest (или пусты)"
            )
        meta_tfidf = (meta.get("full_tfidf") or {}).get("full_tfidf_id")
        if not meta_tfidf or meta_tfidf != verification.get("full_tfidf_id"):
            raise S06ComparisonError(
                f"checkpoint Gate 1 seed {seed}: full_tfidf_id не совпадает с проверенным "
                "артефактом (или пуст)"
            )
        with open(maps_path, encoding="utf-8") as f:
            ckpt_maps = json.load(f)
        if ckpt_maps != (manifest.get("label_maps") or {}):
            raise S06ComparisonError(
                f"checkpoint Gate 1 seed {seed}: label maps не совпадают с текущим сплитом"
            )
    selected_path = os.path.join(cfg.baseline_dir, "selected_checkpoint.json")
    if not os.path.isfile(selected_path):
        raise S06ComparisonError(
            f"selected_checkpoint.json Gate 1 отсутствует: {selected_path!r}. Подтверждение "
            "опорного checkpoint (round-trip) обязательно для gate."
        )
    with open(selected_path, encoding="utf-8") as f:
        selected = json.load(f)
    # Опорный checkpoint — заранее назначенный seed 42 (reference_seed).
    if int(cfg.reference_seed) != GATE1_REFERENCE_SEED:
        raise S06ComparisonError(
            f"reference_seed={cfg.reference_seed} не равен заранее назначенному "
            f"seed {GATE1_REFERENCE_SEED} Gate 1: отказ."
        )
    expected_ref_run = f"seed_{GATE1_REFERENCE_SEED}"
    if selected.get("run_id") != expected_ref_run:
        raise S06ComparisonError(
            f"selected_checkpoint.json Gate 1: run_id={selected.get('run_id')!r}, ожидался "
            f"{expected_ref_run!r}: опорный checkpoint должен быть seed {GATE1_REFERENCE_SEED}."
        )
    ref_row = gate["seeds"].get(GATE1_REFERENCE_SEED)
    if ref_row is None:
        raise S06ComparisonError(
            f"Gate 1: runs.csv не содержит опорный прогон {expected_ref_run!r}: отказ."
        )
    ref_ckpt_dir = selected.get("checkpoint_dir")
    if not ref_ckpt_dir or os.path.abspath(str(ref_ckpt_dir)) != os.path.abspath(str(ref_row["checkpoint_path"])):
        raise S06ComparisonError(
            "selected_checkpoint.json Gate 1: checkpoint_dir не совпадает с checkpoint_path "
            f"опорного прогона в runs.csv ({ref_row['checkpoint_path']!r}): подмена checkpoint, отказ."
        )
    # Подтверждающие checksum: каждая контрольная сумма файла selected checkpoint
    # пересчитывается с диска; подмена весов/метаданных после фиксации — отказ.
    file_hashes = selected.get("checkpoint_files_sha256") or {}
    if not file_hashes:
        raise S06ComparisonError(
            "selected_checkpoint.json Gate 1 не содержит checkpoint_files_sha256: "
            "целостность опорного checkpoint не подтверждена, отказ."
        )
    for name, recorded in file_hashes.items():
        fpath = os.path.join(str(ref_ckpt_dir), name)
        if not recorded or not os.path.isfile(fpath) or _sha256_file(fpath) != recorded:
            raise S06ComparisonError(
                f"selected_checkpoint.json Gate 1: checksum файла {name!r} не совпадает с "
                "фактическим содержимым checkpoint: подмена файлов недопустима, отказ."
            )
    # Round-trip метрики против реальных значений: status success и флаг
    # metrics_match не отменяют сверку чисел; NaN/пустые метрики — отказ.
    roundtrip = selected.get("roundtrip") or {}
    if roundtrip.get("status") != "success":
        raise S06ComparisonError(
            "selected_checkpoint.json Gate 1: round-trip проверка не пройдена — gate не пройден"
        )
    if roundtrip.get("metrics_match") is not True:
        raise S06ComparisonError(
            "selected_checkpoint.json Gate 1: round-trip metrics_match не подтверждён "
            "(флаг не выставлен в true) — gate не пройден"
        )
    for metric in ("l3_macro_f1", "l3_accuracy"):
        saved = roundtrip.get(f"{metric}_saved")
        recomputed = roundtrip.get(f"{metric}_recomputed")
        try:
            saved_val, recomputed_val = float(saved), float(recomputed)
        except (TypeError, ValueError):
            saved_val = recomputed_val = float("nan")
        if (saved is None or recomputed is None
                or saved_val != saved_val or recomputed_val != recomputed_val
                or saved_val != recomputed_val):
            raise S06ComparisonError(
                f"selected_checkpoint.json Gate 1: round-trip метрика {metric} несогласована "
                f"(saved={saved!r}, recomputed={recomputed!r}) — gate не пройден"
            )
        run_metric = ref_row.get(metric)
        try:
            run_val = float(run_metric)
        except (TypeError, ValueError):
            run_val = float("nan")
        if run_metric is None or run_val != run_val or run_val != saved_val:
            raise S06ComparisonError(
                f"selected_checkpoint.json Gate 1: метрика {metric} опорного прогона "
                f"runs.csv ({run_metric!r}) не совпадает с зафиксированным round-trip "
                f"({saved!r}): подмена метрик недопустима, отказ."
            )
    return gate


def verify_preconditions(cfg: S06ComparisonConfig) -> dict[str, Any]:
    """Fail closed до любого ранжирования/refit/обучения.

    Проверяет: сплит Gate 0 (через ``verify_gate0_split``), full TF-IDF против
    корпуса (``verify_full_tfidf``) и согласованность с сохранённой ссылкой
    ``shared/full_tfidf_ref.json``, замороженный протокол Gate 1 и его
    совместимость с фактическими входами (``validate_protocol_against_inputs``),
    и сам gate Gate 1 (три успешных baseline против majority с привязкой к
    текущим split/corpus/TF-IDF/label maps/конфигурации,
    см. :func:`_verify_gate1_baseline`). К ``sealed_test``
    обращений за данными нет и не будет.
    """
    cfg.validate()
    g1 = cfg.gate1_config()
    manifest = verify_gate0_split(g1)  # S06Gate1Error при несоответствии
    verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
    if verification.get("decision") != "ok":
        raise S06ComparisonError(f"проверка full TF-IDF не пройдена: {verification.get('decision')}")
    # Согласованность с сохранённой ссылкой Gate 1 (оба известных формата).
    if os.path.isfile(cfg.tfidf_ref_path):
        with open(cfg.tfidf_ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        ref_id = ref.get("full_tfidf_id")
        if ref_id is None and isinstance(ref.get("files"), dict):
            ref_id = (ref["files"].get("features.joblib") or {}).get("sha256")
        if ref_id and ref_id != verification.get("full_tfidf_id"):
            raise S06ComparisonError(
                "full TF-IDF не совпадает с сохранённой ссылкой "
                f"{cfg.tfidf_ref_path}: подмена артефакта между фазами недопустима"
            )
    protocol = load_training_protocol(g1)
    # Gate 1 валидируется строго по seeds замороженного протокола (полный набор
    # baseline 42/43/44), а не по seeds плана S06: план вправе задавать лишь
    # проверенное подмножество (например [42] для smoke). Плановые seeds при
    # этом обязаны быть непустым подмножеством seeds протокола — иначе отказ.
    proto_seeds = [int(s) for s in (protocol.get("config") or {}).get("seeds") or []]
    if not proto_seeds:
        raise S06ComparisonError(
            "замороженный протокол Gate 1 не содержит config.seeds: проверка Gate 1 невозможна"
        )
    plan_seeds = [int(s) for s in cfg.seeds]
    if not plan_seeds or any(s not in proto_seeds for s in plan_seeds):
        raise S06ComparisonError(
            f"seeds плана S06 {plan_seeds} должны быть непустым подмножеством seeds "
            f"замороженного протокола Gate 1 {proto_seeds}"
        )
    g1_proto = dataclasses.replace(g1, seeds=list(proto_seeds))
    validate_protocol_against_inputs(g1_proto, protocol, manifest, verification)
    gate1 = _verify_gate1_baseline(cfg, manifest, verification, protocol)
    log.info(
        "Предусловия S06 проверены: split min_count=%s, full_tfidf_id=%s…, протокол "
        "заморожен (%d эпох), Gate 1 подтверждён (seed %s, majority macro-F1=%.4f)",
        manifest["policy"].get("min_count"), str(verification.get("full_tfidf_id"))[:12],
        int(protocol.get("fixed_epochs") or 0),
        sorted(gate1["seeds"]), gate1["majority_l3_macro_f1"],
    )
    return {"manifest": manifest, "verification": verification, "protocol": protocol, "gate1": gate1}


def _resolved_train_sections(protocol: dict[str, Any]) -> tuple[ModelConfig, LossConfig, TrainConfig, int]:
    """Замороженные секции обучения из протокола Gate 1 (источник истины)."""
    sections = protocol.get("resolved_sections") or {}
    try:
        model_cfg = ModelConfig(**sections["model"])
        loss_cfg = LossConfig(**sections["loss"])
        train_cfg = TrainConfig(**sections["train"])
    except (KeyError, TypeError) as e:
        raise S06ComparisonError(f"протокол Gate 1 не содержит корректных resolved_sections: {e}")
    # Fail closed: расписание LR, зависящее от числа шагов (cosine/onecycle),
    # не подтверждено идентичностью кривых selection/refit протокола S06;
    # разрешён только постоянный LR 'none' (замороженный протокол Gate 1
    # использует именно его).
    if str(train_cfg.scheduler) != "none":
        raise S06ComparisonError(
            f"протокол Gate 1 содержит scheduler={train_cfg.scheduler!r}: для S06 "
            "разрешён только постоянный LR 'none'; расписания с кривой от числа "
            "шагов не проходят независимую проверку идентичности selection/refit."
        )
    fixed_epochs = int(protocol.get("fixed_epochs") or 0)
    if fixed_epochs < 1:
        raise S06ComparisonError(f"протокол Gate 1: fixed_epochs некорректен: {fixed_epochs!r}")
    return model_cfg, loss_cfg, train_cfg, fixed_epochs


# ---------------------------------------------------------------------------
# План (30 runs) — фиксация до оценки
# ---------------------------------------------------------------------------


def write_plan(cfg: S06ComparisonConfig) -> dict[str, Any]:
    """Фиксирует список плановых запусков (10 конфигураций × seed) до оценки.

    Файл пишется один раз; повторный вызов — сверка идентичности плана
    (методы/бюджеты/seed/protocol hash) без перезаписи.
    """
    cfg.validate()
    runs = [
        {
            "run_id": cfg.run_id(c["slug"], seed),
            "config_slug": c["slug"], "method": c["method"], "budget": c["budget"],
            "seed": int(seed), "status": "pending", "reason": None,
        }
        for c in cfg.planned_configs()
        for seed in cfg.seeds
    ]
    if not os.path.isfile(cfg.protocol_path):
        raise S06ComparisonError(
            f"training_protocol.json не найден: {cfg.protocol_path!r}. План фиксируется "
            "только при существующем замороженном протоколе Gate 1."
        )
    protocol_sha = _sha256_file(cfg.protocol_path)
    if not protocol_sha:
        raise S06ComparisonError(
            f"training_protocol.json пуст: {cfg.protocol_path!r}. План не фиксируется."
        )
    plan = {
        "schema_version": S06_COMPARISON_SCHEMA,
        "role": "planned_runs",
        "generated_at": _now(),
        "config": {
            "methods": list(cfg.methods), "budgets": list(cfg.budgets),
            "char_share": float(cfg.char_share), "seeds": list(cfg.seeds),
            "reference_seed": int(cfg.reference_seed),
            "full_tfidf_path": os.path.abspath(cfg.full_tfidf_path),
            "corpus_path": os.path.abspath(cfg.corpus_path),
            "splits_manifest": os.path.abspath(str(cfg.splits_manifest)),
        },
        "protocol_sha256": protocol_sha,
        "n_planned": len(runs),
        "runs": runs,
        "notes": [
            "10 конфигураций = none (полный baseline Gate 1) + методы × бюджеты; "
            "3 общих train seed; парные Δ строго к none того же seed.",
            "Статусы обновляются результатами прогонов (runs/<slug>/run_result_seed_<s>.json); "
            "pending — запуск не выполнялся; существующие результаты не перезаписываются.",
        ],
    }
    if os.path.exists(cfg.planned_path):
        with open(cfg.planned_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("protocol_sha256") != protocol_sha or existing.get("config") != plan["config"]:
            raise S06ComparisonError(
                f"{cfg.planned_path} уже существует с другим планом (config/protocol_sha256 "
                "не совпадают): перезапись плана недопустима; используйте новый output_root."
            )
        return existing
    os.makedirs(os.path.dirname(cfg.planned_path), exist_ok=True)
    _atomic_write_json(plan, cfg.planned_path)
    log.info("План зафиксирован: %d runs → %s", len(runs), cfg.planned_path)
    return plan


_PLAN_REQUIRED_FINGERPRINTS = (
    "methods", "budgets", "char_share", "seeds", "reference_seed",
    "full_tfidf_path", "corpus_path", "splits_manifest",
)


def _require_plan(cfg: S06ComparisonConfig) -> dict[str, Any]:
    """Требует неизменяемый план до любых записей этапа.

    План загружается через :func:`write_plan`: та же функция сверяет
    конфигурацию (методы/бюджеты/char_share/seed/пути) и ``protocol_sha256`` с
    текущим cfg и замороженным протоколом и отказывает при любом расхождении.
    Дополнительно fail closed: отсутствие обязательного fingerprint в плане,
    отсутствие ``protocol_sha256`` или отсутствие в плане любого run, который
    текущая конфигурация собирается выполнять, — отказ до любых записей.
    """
    plan = write_plan(cfg)
    section = plan.get("config") or {}
    missing = [k for k in _PLAN_REQUIRED_FINGERPRINTS if section.get(k) is None]
    if missing:
        raise S06ComparisonError(
            f"План {cfg.planned_path} не содержит обязательные fingerprint {missing}: отказ."
        )
    if not plan.get("protocol_sha256"):
        raise S06ComparisonError(
            f"План {cfg.planned_path} не содержит protocol_sha256: происхождение протокола "
            "не доказано, отказ до любых записей."
        )
    by_id: dict[str, dict[str, Any]] = {}
    for entry in plan.get("runs") or []:
        run_id = entry.get("run_id")
        if not run_id:
            raise S06ComparisonError(
                f"План {cfg.planned_path} содержит run без run_id: отказ."
            )
        by_id[run_id] = entry
    for c in cfg.planned_configs():
        for seed in cfg.seeds:
            run_id = cfg.run_id(c["slug"], seed)
            entry = by_id.get(run_id)
            if entry is None:
                raise S06ComparisonError(
                    f"План {cfg.planned_path} не содержит run {run_id!r}, который требует "
                    "текущая конфигурация: отказ до любых записей."
                )
            if (entry.get("config_slug") != c["slug"] or entry.get("method") != c["method"]
                    or entry.get("budget") != c["budget"] or entry.get("seed") is None
                    or int(entry["seed"]) != int(seed)):
                raise S06ComparisonError(
                    f"План: run {run_id!r} зафиксирован с другими параметрами "
                    f"({entry.get('config_slug')}/{entry.get('method')}/{entry.get('budget')}/"
                    f"{entry.get('seed')}), ожидалось {c['slug']}/{c['method']}/{c['budget']}/{seed}: отказ."
                )
    return plan


def _strict_branch_budgets(k: int, char_share: float, n_char: int, n_word: int) -> tuple[int, int]:
    """Строгое деление бюджета для S06 (общий ``branch_budgets`` S03/S04 не меняется).

    Общий контракт S03 намеренно допускает усечение бюджета по факту доступных
    признаков. S06 этого не допускает: любое расхождение запрошенного деления
    (``int(k*char_share)`` / остаток) с фактическим — отказ, а не молчаливое
    сокращение выборки.
    """
    k_char_raw = int(k * char_share)
    k_word_raw = k - k_char_raw
    k_char, k_word = branch_budgets(k, char_share, n_char, n_word)
    if k_char != k_char_raw or k_word != k_word_raw:
        raise S06ComparisonError(
            f"бюджет k={k} (char_share={char_share}) был бы молча усечён общим делением "
            f"S03: запрошено char={k_char_raw}/word={k_word_raw}, доступно в словаре "
            f"char={n_char}/word={n_word} (фактически char={k_char}/word={k_word}). "
            "S06 не допускает неявного сокращения бюджета: уменьшите K или "
            "скорректируйте char_share."
        )
    return k_char, k_word


# ---------------------------------------------------------------------------
# Gate 2a: рейтинги (Chi² — по outer_train; G×I — опорный checkpoint Gate 1)
# ---------------------------------------------------------------------------


def _transform_outer_train(
    cfg: S06ComparisonConfig,
    loaded: LoadedVectorizer,
    manifest: dict[str, Any],
    device: torch.device,
) -> tuple[sp.csr_matrix, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Общий transform outer_train полным TF-IDF (единый замер этапа)."""
    train_df = load_part(manifest, "outer_train")
    y_train = _encode_labels(LabelMaps.from_dict(manifest["label_maps"]), train_df, "outer_train")
    with _Measurement(device, f"transform outer_train полным TF-IDF, chunk_rows={cfg.features_chunk_rows}") as m:
        X_train = _transform(loaded, train_df["name"].tolist(), cfg.features_chunk_rows)
    stage = m.stage_row("shared_vectorization", STAGE_VECTORIZE_OUTER, "success")
    stage["artefact_bytes"] = None
    stage["measurement_note"] += (
        "; матрица общая для ранжирования всех методов и атрибуции G×I; повторный "
        "transform внутри фазы rankings не выполняется"
    )
    if not _stage_recorded(cfg, "shared_vectorization", STAGE_VECTORIZE_OUTER):
        _append_stage_rows(cfg.stages_path, [stage])
    log.info("outer_train: %d строк (пустых %d), признаков %d",
             X_train.shape[0], int((np.diff(X_train.indptr) == 0).sum()), X_train.shape[1])
    return X_train, y_train, train_df, stage


def _verify_ratings_meta(ratings_dir: str, method: str) -> dict[str, Any]:
    """Сверяет контрольные суммы объявленных файлов рейтингов с meta.json."""
    meta_path = os.path.join(ratings_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != RATINGS_FORMAT_VERSION:
        raise SelectionArtifactError(
            f"Рейтинги {ratings_dir}: несовместимый формат {manifest.get('format')!r}"
        )
    if manifest.get("method") != method:
        raise SelectionArtifactError(
            f"Рейтинги {ratings_dir}: метод {manifest.get('method')!r}, ожидается {method!r}"
        )
    checksums = manifest.get("files")
    if not isinstance(checksums, dict) or not checksums:
        raise SelectionArtifactError(f"Рейтинги {ratings_dir}: нет корректной секции контрольных сумм")
    for entry in (manifest.get("branches") or {}).values():
        for name in _ratings_declared_names(entry).values():
            path = os.path.join(ratings_dir, name)
            info = checksums.get(name)
            if not isinstance(info, dict) or not os.path.isfile(path):
                raise SelectionArtifactError(f"Рейтинги {ratings_dir}: нет файла/суммы {name!r}")
            if _sha256_file(path) != info.get("sha256") or os.path.getsize(path) != info.get("bytes"):
                raise SelectionArtifactError(
                    f"Рейтинги {ratings_dir}: контрольная сумма/размер {name!r} не совпадают"
                )
    return manifest


def _persist_ratings(
    cfg: S06ComparisonConfig,
    method: str,
    result: SelectionResult,
    *,
    outer_train_sha256: str,
    n_rows: int,
    label_map_sha256: str,
    full_tfidf_id: str | None,
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Переносит полные рейтинги из временного каталога в ``shared/rankings``."""
    dest = cfg.rankings_dir(method)
    if os.path.exists(dest):
        raise S06ComparisonError(f"Каталог рейтингов уже существует и не перезаписывается: {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # select_features возвращает ratings_dir, указывающий на каталог с meta.json
    # и .npy-файлами рейтингов; переносим его целиком в shared/rankings/<method>.
    shutil.move(result.ratings_dir, dest)
    result.ratings_dir = dest
    manifest = _verify_ratings_meta(dest, method)
    files = {
        name: {"sha256": _sha256_file(os.path.join(result.ratings_dir, name)),
               "bytes": os.path.getsize(os.path.join(result.ratings_dir, name))}
        for name in sorted(os.listdir(result.ratings_dir)) if os.path.isfile(os.path.join(result.ratings_dir, name))
    }
    meta = {
        "schema_version": S06_COMPARISON_SCHEMA,
        "role": "rankings",
        "method": method,
        "created_at": _now(),
        "ratings_dir": os.path.abspath(result.ratings_dir),
        "ratings_manifest": manifest,
        "files": files,
        "computed_on": {
            "split": "outer_train",
            "outer_train_sha256": outer_train_sha256,
            "n_rows": int(n_rows),
            "label_map_sha256": label_map_sha256,
            "full_tfidf_id": full_tfidf_id,
        },
        "budgets_note": PROTOCOL_NOTES["budget_reuse"],
        "train_only_note": PROTOCOL_NOTES["rankings_train_only"],
    }
    if extra:
        meta.update(extra)
    _atomic_write_json(meta, os.path.join(dest, "rankings_meta.json"))
    log.info("Рейтинги %s сохранены: %s (%d файлов)", method, dest, len(files))
    return result.ratings_dir, meta


def _load_rankings_meta(cfg: S06ComparisonConfig, method: str) -> tuple[str, dict[str, Any]] | None:
    path = os.path.join(cfg.rankings_dir(method), "rankings_meta.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    ratings_dir = meta["ratings_dir"]
    _verify_ratings_meta(ratings_dir, method)
    for name, info in (meta.get("files") or {}).items():
        p = os.path.join(ratings_dir, name)
        if not os.path.isfile(p) or _sha256_file(p) != info.get("sha256"):
            raise SelectionArtifactError(
                f"Сохранённые рейтинги {method} повреждены (файл {name!r}): "
                "пересчёт без удаления зафиксированных результатов не выполняется"
            )
    return ratings_dir, meta


def _selection_from_saved_ratings(
    method: str,
    ratings_dir: str,
    k: int,
    char_share: float,
    n_char: int,
    n_total: int,
    l3_labels: list[str] | None = None,
) -> SelectionResult:
    """Повторное применение бюджета к сохранённым полным рейтингам Chi².

    Воспроизводит поведение :func:`kkt_clf.selection.select_features` для
    заданного K без пересчёта оценок: global — стабильный top-K глобального
    рейтинга; class_aware — тот же равноправный round-robin из тех же
    потоковых файлов с тем же детерминированным добором. Индексы результата —
    полно-пространные (word со смещением ``n_char``).
    """
    manifest = _verify_ratings_meta(ratings_dir, method)
    k_char, k_word = _strict_branch_budgets(k, char_share, n_char, n_total - n_char)
    report: dict[str, Any] = {
        "method": method,
        "k_requested": int(k),
        "char_share": float(char_share),
        "seed": 42,
        "ratings_dir": os.path.abspath(ratings_dir),
        "tie_policy": "stable sort by (-score, feature_index); no RNG",
        "budgets": {"k_char": int(k_char), "k_word": int(k_word)},
        "n_features_total": int(n_total),
        "reapplied_from_saved_ratings": True,
    }
    branches: dict[str, BranchSelection] = {}
    if method == METHOD_GLOBAL:
        for branch, k_b, width in (("char", k_char, n_char), ("word", k_word, n_total - n_char)):
            entry = (manifest.get("branches") or {}).get(branch) or {}
            name = entry.get("global_scores")
            if not name:
                raise SelectionArtifactError(f"Манифест рейтингов global: ветка {branch!r} не объявляет global_scores")
            scores = np.fromfile(os.path.join(ratings_dir, name), dtype=np.float64)
            if len(scores) != width:
                raise SelectionArtifactError(
                    f"Глобальный рейтинг ветки {branch!r}: длина {len(scores)} != ширине {width}"
                )
            top = _rank_topk(scores, k_b)
            branches[branch] = BranchSelection(
                indices=top, scores=scores[top].astype(np.float64),
                origin=["global_fallback"] * len(top),
            )
            report[branch] = {"n_selected": len(top), "n_fallback": 0}
    elif method == METHOD_CLASS_AWARE:
        arrays = validate_ratings_dir(
            ratings_dir, manifest, method=METHOD_CLASS_AWARE,
            widths={"char": int(n_char), "word": int(n_total - n_char)},
        )
        for branch, k_b, offset in (("char", k_char, 0), ("word", k_word, n_char)):
            width = n_char if branch == "char" else n_total - n_char
            entry = (manifest.get("branches") or {}).get(branch) or {}
            sel, rep = roundrobin_select_from_rank_files(
                os.path.join(ratings_dir, entry["rank_indices"]),
                os.path.join(ratings_dir, entry["rank_scores"]),
                arrays[branch]["counts"], arrays[branch]["classes"],
                k_b, width, arrays[branch]["global_scores"],
                positive_only_fallback=False,
            )
            branches[branch] = sel
            report[branch] = {"n_selected": len(sel.indices), **rep}
    else:
        raise SelectionError(f"_selection_from_saved_ratings: метод {method!r} не поддерживается")
    word = branches["word"]
    word = BranchSelection(
        indices=word.indices + n_char if len(word.indices) else word.indices,
        scores=word.scores,
        origin=word.origin,
    )
    classes_labels = None
    if l3_labels is not None and method == METHOD_CLASS_AWARE:
        # Ровно те классы, что присутствовали в train при построении рейтингов.
        present = validate_ratings_dir(
            ratings_dir, manifest, method=METHOD_CLASS_AWARE,
            widths={"char": int(n_char), "word": int(n_total - n_char)},
        )["char"]["classes"]
        classes_labels = [str(l3_labels[int(i)]) for i in present]
    return SelectionResult(
        method=method,
        k_requested=int(k),
        char_share=float(char_share),
        seed=42,
        char=branches["char"],
        word=word,
        classes=classes_labels,
        report=report,
        ratings_dir=ratings_dir,
        ratings_manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Опорный checkpoint Gate 1 для G×I (read-only проверки, fail closed)
# ---------------------------------------------------------------------------


def _verify_reference_checkpoint(
    cfg: S06ComparisonConfig,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    loaded: LoadedVectorizer,
    protocol: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Проверяет опорный checkpoint Gate 1 до Gradient × Input-атрибуции.

    Слои проверок (все fail closed):

    1. Загрузка артефакта checkpoint ``reference_seed`` и сверка полного
       контракта: seed, размерность (= полный TF-IDF), label maps, число эпох
       (замороженное fixed_epochs), restored_best=False, sha256 full TF-IDF,
       sha256 train-сплитов, проверяемый отпечаток контракта признаков
       (словари/IDF/параметры transform).
    2. При наличии ``attribution_provenance.json`` в checkpoint — дополнительно
       полная проверка S04 (:func:`verify_attribution_checkpoint`, включая
       sha256 весов model.pt).
    3. При наличии ``selected_checkpoint.json`` Gate 1 — сверка статуса
       round-trip и файловых sha256: отсутствие записанных hash или
       несовпадение хотя бы одного файла (в первую очередь ``model.pt``) —
       отказ; изменённые веса для G×I не используются, даже если остальной
       provenance отсутствует. Hash ``model.pt`` обязан вычисляться.
    """
    seed = int(cfg.reference_seed)
    ckpt_dir = os.path.join(cfg.baseline_dir, "checkpoints", f"seed_{seed}")
    if not os.path.isdir(ckpt_dir):
        raise S06ComparisonError(
            f"Опорный checkpoint Gate 1 не найден: {ckpt_dir!r}. G×I-атрибуция "
            "требует завершённого refit reference_seed (D06 value_dod)."
        )
    art = load_artifact(ckpt_dir, device)
    maps = LabelMaps.from_dict(manifest["label_maps"])
    _, _, _, fixed_epochs = _resolved_train_sections(protocol)
    art_meta = art.meta or {}
    art_train = art_meta.get("train") or {}
    checks: dict[str, Any] = {
        "checkpoint_dir": os.path.abspath(ckpt_dir),
        "seed_matches": bool(art_meta.get("seed") == seed),
        "n_features_matches_full_tfidf": bool(art.model.n_features == loaded.builder.n_features),
        "label_maps_match": bool(art.maps.to_dict() == maps.to_dict()),
        "fixed_epochs_matches": bool(int(art_train.get("epochs_completed", -1)) == fixed_epochs),
        "restored_best_is_false": bool(art_train.get("restored_best")) is False,
        "full_tfidf_id_matches": bool(
            (art_meta.get("full_tfidf") or {}).get("full_tfidf_id")
            == (verification.get("full_tfidf_id") or _full_tfidf_id(loaded))
        ),
        "train_fingerprints_match": bool(
            (art_meta.get("train_fingerprints") or {}).get("outer_train_sha256")
            == manifest["parts"]["outer_train"]["sha256"]
            and (art_meta.get("train_fingerprints") or {}).get("train_monitor_sha256")
            == manifest["parts"]["train_monitor"]["sha256"]
        ),
        "feature_contract_matches": bool(
            feature_contract_fingerprint(art.features) == feature_contract_fingerprint(loaded.builder)
        ),
    }
    failed = {k: v for k, v in checks.items() if isinstance(v, bool) and not v}
    if failed:
        raise S06ComparisonError(
            f"Опорный checkpoint {ckpt_dir!r} не совместим с входами S06: {failed}"
        )
    prov_path = os.path.join(ckpt_dir, ATTRIBUTION_PROVENANCE_FILENAME)
    if os.path.isfile(prov_path):
        # Полная проверка S04: fingerprint контракта признаков, sha256 весов,
        # label maps, train split. data_cfg у checkpoint контура S05 отсутствует
        # (сплиты приходят из manifest) — проверка data_cfg вакуальна, остальные
        # связки обязательны.
        verify_attribution_checkpoint(
            ckpt_dir, art.model, art.config,
            n_features_full=int(loaded.builder.n_features),
            label_map_sha256=_sha256_json(maps.to_dict()),
            data_cfg=None,
            seed=seed,
            full_tfidf_fp=feature_contract_fingerprint(loaded.builder),
            train_split_sha256=manifest["parts"]["outer_train"]["sha256"],
        )
        checks["attribution_provenance_s04_check"] = True
    else:
        checks["attribution_provenance_s04_check"] = None  # Gate 1 не писал файл — слой 1 обязателен
    selected_path = os.path.join(cfg.baseline_dir, "selected_checkpoint.json")
    if os.path.isfile(selected_path):
        with open(selected_path, encoding="utf-8") as f:
            selected = json.load(f)
        checks["gate1_selected_roundtrip_ok"] = bool((selected.get("roundtrip") or {}).get("status") == "success")
        if checks["gate1_selected_roundtrip_ok"] is not True:
            raise S06ComparisonError(
                f"selected_checkpoint.json Gate 1: round-trip проверка не пройдена "
                f"(status={(selected.get('roundtrip') or {}).get('status')!r}) — опорный "
                "checkpoint для G×I не подтверждён"
            )
        files_sha = selected.get("checkpoint_files_sha256") or {}
        actual = {
            name: _sha256_file(os.path.join(ckpt_dir, name))
            for name in sorted(os.listdir(ckpt_dir)) if os.path.isfile(os.path.join(ckpt_dir, name))
        }
        if not files_sha:
            raise S06ComparisonError(
                "selected_checkpoint.json Gate 1 не содержит checkpoint_files_sha256: "
                "hash опорного checkpoint не зафиксирован и не может быть сверен — отказ."
            )
        mismatched = sorted(
            {name for name in set(actual) | set(files_sha) if actual.get(name) != files_sha.get(name)}
        )
        if mismatched:
            raise S06ComparisonError(
                f"Хэши файлов опорного checkpoint {ckpt_dir!r} не совпадают с "
                f"selected_checkpoint.json Gate 1: {mismatched}. Подмена весов/файлов "
                "между Gate 1 и атрибуцией недопустима."
            )
        checks["gate1_selected_files_match"] = True
    weights_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.isfile(weights_path):
        raise S06ComparisonError(f"model.pt опорного checkpoint отсутствует: {ckpt_dir!r}")
    checks["model_weights_sha256"] = _sha256_file(weights_path)
    if not checks["model_weights_sha256"]:
        raise S06ComparisonError(f"model.pt опорного checkpoint не читается: {ckpt_dir!r}")
    log.info("Опорный checkpoint проверен: %s (seed=%d)", ckpt_dir, seed)
    return {"artifact": art, "checks": checks}


# ---------------------------------------------------------------------------
# Gate 2a целиком: Chi²-рейтинги, G×I-атрибуция, 9 артефактов отбора
# ---------------------------------------------------------------------------


def _selection_provenance(
    cfg: S06ComparisonConfig,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    slug: str,
    method: str,
    budget: int | None,
    attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prov: dict[str, Any] = {
        "vectorizer": {
            "path": os.path.abspath(cfg.full_tfidf_path),
            "source_sha256": (verification.get("corpus_fingerprint") or {}).get("sha256"),
            "full_tfidf_id": verification.get("full_tfidf_id"),
        },
        "labeled": {
            "split": "outer_train",
            "splits_manifest": os.path.abspath(str(cfg.splits_manifest)),
            "outer_train_sha256": manifest["parts"]["outer_train"]["sha256"],
        },
        "protocol": {
            "path": os.path.abspath(cfg.protocol_path),
            "sha256": _sha256_file(cfg.protocol_path),
        },
        "experiment": {
            "sprint": "S06",
            "config_slug": slug,
            "method": method,
            "budget": budget,
            "char_share": float(cfg.char_share),
            "selection_seed": 42,
            "note": PROTOCOL_NOTES["rankings_train_only"],
        },
    }
    if attribution is not None:
        prov["attribution"] = attribution
    return prov


def _save_selection_artifact(
    cfg: S06ComparisonConfig,
    slug: str,
    result: SelectionResult,
    provenance: dict[str, Any],
) -> str:
    """Сохраняет артефакт отбора; resume отключён: существующий каталог — отказ."""
    out_dir = cfg.selection_dir(slug)
    if os.path.isdir(out_dir):
        raise SelectionArtifactError(
            f"resume отключён: артефакт отбора уже существует ({out_dir}): provenance не "
            "подтверждён; используйте новый output_root, перезапись недопустима"
        )
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    save_selection_artifact(out_dir, result, provenance=provenance, overwrite=False, ratings_copy=True)
    return os.path.abspath(out_dir)


def run_rankings(cfg: S06ComparisonConfig, device: torch.device) -> dict[str, Any]:
    """Этап Gate 2a: рейтинги трёх методов (outer_train-only) + 9 отборов.

    Resume отключён (fail closed): обнаружение ранее сохранённых рейтингов,
    атрибуции или артефактов отбора — явный ``S06ComparisonError``/
    ``SelectionArtifactError`` до любых записей; fresh проход в новом каталоге
    создаёт все артефакты заново и ничего не перезаписывает.
    """
    cfg.validate()
    inputs = verify_preconditions(cfg)
    manifest, verification, protocol = inputs["manifest"], inputs["verification"], inputs["protocol"]
    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
    maps = LabelMaps.from_dict(manifest["label_maps"])
    l3_labels = list(maps.l3)
    n_char = int(loaded.builder.n_char_features)
    n_total = int(loaded.builder.n_features)
    # Строгие бюджеты char/word (fail fast, до transform и любых записей):
    # общая функция S03 допускает усечение, S06 — нет.
    for budget in cfg.budgets:
        _strict_branch_budgets(int(budget), cfg.char_share, n_char, n_total - n_char)
    # Неизменяемый план обязан быть зафиксирован до любых записей этапа.
    _require_plan(cfg)
    label_map_sha = _sha256_json(maps.to_dict())
    outer_sha = manifest["parts"]["outer_train"]["sha256"]
    full_id = verification.get("full_tfidf_id")

    X_outer, y_outer, _train_df, _stage = _transform_outer_train(cfg, loaded, manifest, device)
    if X_outer.shape[1] != n_total:
        raise S06ComparisonError(f"размерность матрицы outer_train {X_outer.shape[1]} != словарю полного TF-IDF {n_total}")
    y_l3 = y_outer[2]
    if np.any(y_l3 < 0):
        raise S06ComparisonError("В outer_train есть строки вне label maps: ранжирование невозможно")

    index: dict[str, Any] = {
        "schema_version": S06_COMPARISON_SCHEMA,
        "role": "rankings_index",
        "generated_at": _now(),
        "methods": {},
        "selections": {},
        "notes": list(PROTOCOL_NOTES.values()),
    }
    k_max = max(cfg.budgets)
    sel_provenance_extra: dict[str, Any] = {}

    # --- Chi²-методы: рейтинги один раз, бюджеты из сохранённых рейтингов.
    for method in (m for m in cfg.methods if m in (METHOD_GLOBAL, METHOD_CLASS_AWARE)):
        saved = _load_rankings_meta(cfg, method)
        stage_note = (
            f"supervised Chi²-отбор (method={method}) только по размеченному outer_train "
            f"полного TF-IDF; метки comparison_val/sealed_test не использовались"
        )
        if saved is not None:
            raise S06ComparisonError(
                f"resume отключён: сохранённые рейтинги {method} уже существуют "
                f"({cfg.rankings_dir(method)}): provenance не подтверждён; используйте новый "
                "output_root, перезапись и переиспользование недопустимы"
            )
        with _Measurement(device, stage_note) as m:
            result_max = select_features(
                X_outer, y_l3,
                SelectionConfig(method=method, k=int(k_max), char_share=cfg.char_share, seed=42),
                n_char, l3_labels=l3_labels,
            )
            ratings_dir, _ratings_meta = _persist_ratings(
                cfg, method, result_max,
                outer_train_sha256=outer_sha, n_rows=int(X_outer.shape[0]),
                label_map_sha256=label_map_sha, full_tfidf_id=full_id,
            )
        stage = m.stage_row(f"shared_ranking_{method}", STAGE_RANKING, "success")
        stage["artefact_bytes"] = _dir_size(cfg.rankings_dir(method))
        _append_stage_rows(cfg.stages_path, [stage])
        index["methods"][method] = {
            "ratings_meta": os.path.abspath(os.path.join(cfg.rankings_dir(method), "rankings_meta.json")),
            "ratings_dir": ratings_dir,
            "sha256": _sha256_file(os.path.join(cfg.rankings_dir(method), "rankings_meta.json")),
            "stage_seconds": stage.get("seconds"),
        }
        for budget in cfg.budgets:
            slug = cfg.slug(method, budget)
            art_path = cfg.selection_dir(slug)
            if os.path.isdir(art_path):
                raise SelectionArtifactError(
                    f"resume отключён: артефакт отбора уже существует ({art_path}): provenance "
                    "не подтверждён; используйте новый output_root, перезапись недопустима"
                )
            result = _selection_from_saved_ratings(
                method, ratings_dir, int(budget), cfg.char_share, n_char, n_total, l3_labels=l3_labels
            )
            _save_selection_artifact(
                cfg, slug, result,
                _selection_provenance(cfg, manifest, verification, slug, method, int(budget)),
            )
            index["selections"][slug] = {"artifact": os.path.abspath(art_path)}

    # --- Gradient × Input: атрибуция на опорном checkpoint Gate 1 (один раз).
    if METHOD_GXI in cfg.methods:
        att_dir = cfg.attribution_dir()
        report_path = os.path.join(att_dir, "attribution_report.json")
        ratings_dir_gxi = os.path.join(att_dir, "ratings")
        ref = _verify_reference_checkpoint(cfg, manifest, verification, loaded, protocol, device)
        if os.path.isfile(report_path):
            raise S06ComparisonError(
                f"resume отключён: сохранённая G×I-атрибуция уже существует ({att_dir}): "
                "provenance не подтверждён; используйте новый output_root, перезапись недопустима"
            )
        if os.path.isdir(att_dir) and os.listdir(att_dir):
            raise S06ComparisonError(
                f"Каталог атрибуции {att_dir} существует, но attribution_report.json отсутствует: "
                "состояние частично; resume отключён, удаление за пользователем, перезапись недопустима."
            )
        os.makedirs(att_dir, exist_ok=True)
        with _Measurement(
            device,
            "Gradient × Input-атрибуция по margin (истинная L3 vs сильнейший конкурент) на "
            f"outer_train, опорный checkpoint Gate 1 seed={cfg.reference_seed}; "
            "вклады только по ненулевым CSR-значениям, плотные N×F не создаются",
        ) as m:
            att = compute_margin_attributions(
                ref["artifact"].model, X_outer, y_l3, device, batch_size=cfg.attribution_batch_size
            )
            manifest_gxi, att_report = build_gxi_ratings(att, n_char, ratings_dir_gxi, l3_labels=l3_labels)
        att_report["checkpoint"] = {
            "dir": ref["checks"]["checkpoint_dir"],
            "model_weights_sha256": ref["checks"]["model_weights_sha256"],
            "checks": {k: v for k, v in ref["checks"].items() if k != "model_weights_sha256"},
            "selection_rule": {"kind": "reference_seed", "value": int(cfg.reference_seed)},
        }
        att_report["computed_on"] = {
            "split": "outer_train", "outer_train_sha256": outer_sha,
            "n_rows": int(X_outer.shape[0]), "label_map_sha256": label_map_sha,
            "full_tfidf_id": full_id,
        }
        att_report["protocol_notes"] = {
            "train_only": PROTOCOL_NOTES["rankings_train_only"],
            "reference": PROTOCOL_NOTES["gxi_reference"],
        }
        _atomic_write_json(att_report, report_path)
        stage = m.stage_row("shared_attribution", STAGE_ATTRIBUTION, "success")
        stage["artefact_bytes"] = _dir_size(att_dir)
        _append_stage_rows(cfg.stages_path, [stage])
        index["methods"][METHOD_GXI] = {
            "attribution_report": os.path.abspath(report_path),
            "ratings_dir": os.path.abspath(ratings_dir_gxi),
            "sha256": _sha256_file(report_path),
            "stage_seconds": stage.get("seconds"),
        }
        sel_provenance_extra = {
            "checkpoint_dir": ref["checks"]["checkpoint_dir"],
            "model_weights_sha256": ref["checks"]["model_weights_sha256"],
            "selection_rule": {"kind": "reference_seed", "value": int(cfg.reference_seed)},
        }
        for budget in cfg.budgets:
            slug = cfg.slug(METHOD_GXI, budget)
            art_path = cfg.selection_dir(slug)
            if os.path.isdir(art_path):
                raise SelectionArtifactError(
                    f"resume отключён: артефакт отбора уже существует ({art_path}): provenance "
                    "не подтверждён; используйте новый output_root, перезапись недопустима"
                )
            stage_sel = {
                "run_id": f"shared_selection_{slug}", "stage": STAGE_SELECTION, "status": "success",
                "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None, "artefact_bytes": None,
                "measurement_note": (
                    "class-aware Top-K round-robin из сохранённых G×I-рейтингов "
                    "(shared, train-only); метки val/test не используются; выбор фиксируется для всех seed"
                ),
            }
            result = select_features_gradient_x_input(
                ratings_dir_gxi,
                SelectionConfig(method=METHOD_GXI, k=int(budget), char_share=cfg.char_share, seed=42),
                n_char, n_total, l3_labels=l3_labels,
            )
            _save_selection_artifact(
                cfg, slug, result,
                _selection_provenance(
                    cfg, manifest, verification, slug, METHOD_GXI, int(budget),
                    attribution=sel_provenance_extra,
                ),
            )
            _append_stage_rows(cfg.stages_path, [stage_sel])
            index["selections"][slug] = {"artifact": os.path.abspath(art_path)}

    # Фиксируем индекс рейтингов/отборов (только новый файл; resume отключён).
    index_path = os.path.join(cfg.shared_dir, "rankings_index.json")
    index["generated_at"] = _now()
    if os.path.exists(index_path):
        raise S06ComparisonError(
            f"resume отключён: {index_path} уже существует: provenance не подтверждён; "
            "используйте новый output_root, перезапись недопустима"
        )
    _atomic_write_json(index, index_path)
    log.info("Этап rankings завершён: методы %s, отборов %d", list(index["methods"]), len(index["selections"]))
    return index


# ---------------------------------------------------------------------------
# Gate 2b: fixed-vocabulary refit (один на метод×K)
# ---------------------------------------------------------------------------


def run_refit_selected(cfg: S06ComparisonConfig) -> dict[str, Any]:
    """Refit выбранных словарей на том же снимке корпуса, что полный TF-IDF.

    Делегирует проверки и fit :func:`kkt_clf.vectorizer.refit_selected_vectorizer`
    (sha256 корпуса, воспроизведение подвыборки, инвариант IDF, L2-нормировка).
    Каждый refit-артефакт строится один раз и делится тремя seed. Resume
    отключён (fail closed): существующий refit-результат или частичный каталог
    — явный ``S06ComparisonError`` до любых записей.
    """
    cfg.validate()
    inputs = verify_preconditions(cfg)
    manifest, verification = inputs["manifest"], inputs["verification"]
    # Неизменяемый план обязан быть зафиксирован до любых записей этапа.
    _require_plan(cfg)
    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
    results: dict[str, Any] = {}
    for c in cfg.planned_configs():
        if c["method"] == METHOD_NONE:
            continue
        slug = c["slug"]
        out_dir = os.path.join(cfg.config_dir(slug), "tfidf_selected")
        result_path = os.path.join(cfg.config_dir(slug), "refit_result.json")
        if os.path.isfile(result_path):
            raise S06ComparisonError(
                f"resume отключён: refit-результат уже существует ({result_path}): provenance не "
                "подтверждён; используйте новый output_root, перезапись и переиспользование недопустимы"
            )
        if os.path.isdir(out_dir):
            raise S06ComparisonError(
                f"Каталог {out_dir} существует без refit_result.json: состояние частично, "
                "перезапись недопустима; resume отключён, удаление за пользователем"
            )
        selection, _sel_meta = load_selection_artifact(cfg.selection_dir(slug))
        with _Measurement(
            torch.device("cpu"),
            f"refit TF-IDF с фиксированными словарями ({slug}) на том же снимке корпуса, "
            "что полный TF-IDF S02; новый IDF и L2-нормировка в каждой ветке",
        ) as m:
            refit = refit_selected_vectorizer(
                cfg.corpus_path, loaded.builder, loaded.meta, selection,
                out_dir=out_dir, overwrite=False,
            )
            art = load_vectorizer_artifact(out_dir)  # round-trip load check
        checks = dict(refit.checks)
        checks["roundtrip_load"] = True
        result = {
            "schema_version": S06_COMPARISON_SCHEMA,
            "slug": slug, "method": c["method"], "budget": c["budget"],
            "status": "success", "created_at": _now(),
            "artifact_dir": os.path.abspath(out_dir),
            "vocab": {
                "n_char": int(art.builder.n_char_features),
                "n_word": int(art.builder.n_word_features),
                "n_total": int(art.builder.n_features),
            },
            "checks": checks,
            "files": {
                name: {"sha256": _sha256_file(os.path.join(out_dir, name)),
                       "bytes": os.path.getsize(os.path.join(out_dir, name))}
                for name in sorted(os.listdir(out_dir)) if os.path.isfile(os.path.join(out_dir, name))
            },
            "provenance": {
                "corpus_sha256": (verification.get("corpus_fingerprint") or {}).get("sha256"),
                "full_tfidf_id": verification.get("full_tfidf_id"),
                "selection_artifact": os.path.abspath(cfg.selection_dir(slug)),
                "selection_payload_sha256": (
                    (load_selection_artifact(cfg.selection_dir(slug))[1].get("files") or {})
                    .get("selection.joblib", {}).get("sha256")
                ),
                "outer_train_sha256": manifest["parts"]["outer_train"]["sha256"],
                "note": PROTOCOL_NOTES["refit_once"],
            },
            "stage_seconds": m.seconds,
        }
        _atomic_write_json(result, result_path)
        stage = {
            "run_id": f"shared_refit_{slug}", "stage": STAGE_REFIT, "status": "success",
            "seconds": m.seconds, "peak_rss_bytes": m.peak_rss_bytes,
            "peak_vram_bytes": None, "artefact_bytes": _dir_size(out_dir),
            "measurement_note": "refit_selected_vectorizer: sha256 корпуса, та же подвыборка строк, "
                                "инвариант IDF, L2-нормировка, save→load round-trip",
        }
        _append_stage_rows(cfg.stages_path, [stage])
        results[slug] = result
        log.info("Refit %s завершён: char %d, word %d", slug, result["vocab"]["n_char"], result["vocab"]["n_word"])
    return results


# ---------------------------------------------------------------------------
# Gate 3: 30 paired runs (none — reuse baseline Gate 1; selected — обучение)
# ---------------------------------------------------------------------------


def _read_gate1_baseline_runs(cfg: S06ComparisonConfig) -> pd.DataFrame:
    path = os.path.join(cfg.baseline_dir, "runs.csv")
    if not os.path.isfile(path):
        raise S06ComparisonError(
            f"runs.csv baseline Gate 1 не найден: {path!r}. Парные сравнения невозможны без Gate 1."
        )
    return pd.read_csv(path)


def _load_run_result(cfg: S06ComparisonConfig, slug: str, seed: int) -> dict[str, Any] | None:
    path = cfg.run_result_path(slug, seed)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_predictions(
    cfg: S06ComparisonConfig,
    run_id: str,
    row_ids: np.ndarray,
    y_true_l3: np.ndarray,
    y_pred_l3: np.ndarray,
    l3_labels: list[str],
) -> str:
    path = cfg.predictions_path(run_id)
    if os.path.exists(path):
        raise S06ComparisonError(f"Предсказания уже существуют и не перезаписываются: {path}")
    frame = pd.DataFrame({
        "row_id": np.asarray(row_ids, dtype=np.int64),
        "run_id": run_id,
        "true_L3": [l3_labels[int(t)] for t in y_true_l3],
        "predicted_L3": [l3_labels[int(p)] for p in y_pred_l3],
    })
    frame["correct"] = frame["true_L3"] == frame["predicted_L3"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path, index=False)
    return os.path.abspath(path)


def _eval_run(
    cfg: S06ComparisonConfig,
    model: torch.nn.Module,
    X_val: sp.csr_matrix,
    y_val_l3: np.ndarray,
    val_row_ids: np.ndarray,
    l3_labels: list[str],
    run_id: str,
    stage_rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Raw top-1 на comparison_val + сохранение предсказаний (единый протокол оценки)."""
    m = _Measurement(
        device, f"{run_id}: predict raw top-1 на comparison_val (n={X_val.shape[0]}), без отказа"
    )
    try:
        with m:
            preds, _ = predict(model, X_val, device, cfg.eval_batch_size, top_k=1)
            macro, acc, per_class = _l3_val_metrics(y_val_l3, preds.l3_top1, l3_labels)
    except Exception as e:
        row = m.stage_row(run_id, STAGE_VAL_INFERENCE, "failed")
        row["artefact_bytes"] = None
        row["measurement_note"] += f" Этап прерван: {type(e).__name__}: {e}"
        stage_rows.append(row)
        raise
    stage = m.stage_row(run_id, STAGE_VAL_INFERENCE, "success")
    stage["artefact_bytes"] = None
    stage["measurement_note"] += (
        "; фиксированная L3-вселенная label maps; support=0 → null; пустые векторы признаков "
        "считаются ошибками (модель вызывается, bias-путь), их число фиксируется отдельно"
    )
    stage_rows.append(stage)
    pred_path = _write_predictions(cfg, run_id, val_row_ids, y_val_l3, preds.l3_top1, l3_labels)
    return {
        "l3_macro_f1": macro, "l3_accuracy": acc, "per_class": per_class,
        "n_empty_features_val": int((np.diff(X_val.indptr) == 0).sum()),
        "inference_seconds": m.seconds,
        "inference_peak_rss_bytes": m.peak_rss_bytes,
        "inference_peak_vram_bytes": m.peak_vram_bytes,
        "predictions_path": pred_path,
    }


def _verify_none_checkpoint(
    cfg: S06ComparisonConfig,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    loaded: LoadedVectorizer,
    ckpt_dir: str,
    seed: int,
    device: torch.device,
) -> Any:
    """Проверка reuse baseline-checkpoint Gate 1 перед оценкой (fail closed)."""
    art = load_artifact(ckpt_dir, device)
    maps = LabelMaps.from_dict(manifest["label_maps"])
    meta = art.meta or {}
    ok = (
        meta.get("seed") == seed
        and art.model.n_features == loaded.builder.n_features
        and art.maps.to_dict() == maps.to_dict()
        and (meta.get("full_tfidf") or {}).get("full_tfidf_id") == verification.get("full_tfidf_id")
        and (meta.get("train_fingerprints") or {}).get("outer_train_sha256")
        == manifest["parts"]["outer_train"]["sha256"]
        and feature_contract_fingerprint(art.features) == feature_contract_fingerprint(loaded.builder)
    )
    if not ok:
        raise S06ComparisonError(
            f"Baseline-checkpoint {ckpt_dir} не прошёл проверку provenance для reuse (seed={seed})"
        )
    return art


def _finalize_run_result(
    cfg: S06ComparisonConfig,
    *,
    slug: str,
    method: str,
    budget: int | None,
    seed: int,
    status: str,
    reason: str | None,
    n_features: int | None,
    n_char: int | None,
    n_word: int | None,
    train_info: dict[str, Any] | None,
    eval_info: dict[str, Any] | None,
    checkpoint_dir: str | None,
    tfidf_artifact: str | None,
    tfidf_sha256: str | None,
    selection_artifact: str | None,
    train_stage: dict[str, Any] | None,
    baseline_check: dict[str, Any] | None,
    device: torch.device,
) -> dict[str, Any]:
    pred_path = (eval_info or {}).get("predictions_path")
    predictions_sha256: str | None = None
    if status == "success":
        # Fail closed: успешный run обязан иметь созданный файл предсказаний
        # (каноничный runs/predictions/<run_id>.parquet) и фиксирует его sha256
        # — протокол отчёта kkt_clf.s06_report сверяет hash с файлом, run_id и
        # validation split; отсутствие файла/hash — отказ, а не success.
        if not pred_path or not os.path.isfile(pred_path):
            raise S06ComparisonError(
                f"run {cfg.run_id(slug, seed)}: status=success без созданного файла предсказаний "
                f"({pred_path!r}): результат не фиксируется"
            )
        predictions_sha256 = _sha256_file(pred_path)
    result = {
        "schema_version": RUN_RESULT_SCHEMA,
        "run_id": cfg.run_id(slug, seed),
        "config_slug": slug, "method": method, "budget": budget, "seed": int(seed),
        "status": status, "reason": reason,
        "n_features": n_features, "n_char": n_char, "n_word": n_word,
        "fixed_l3_universe": None,
        "train": train_info,
        "comparison_val_eval": eval_info,
        "baseline_metric_check": baseline_check,
        "artifacts": {
            "checkpoint_dir": checkpoint_dir,
            "tfidf_artifact": tfidf_artifact,
            "selection_artifact": selection_artifact,
        },
        "hashes": {
            "model_pt_sha256": (
                _sha256_file(os.path.join(checkpoint_dir, "model.pt")) if checkpoint_dir else None
            ),
            "tfidf_features_sha256": tfidf_sha256,
            "predictions_sha256": predictions_sha256,
        },
        "device": str(device),
        "created_at": _now(),
        "versions": environment_versions(),
    }
    if eval_info is not None:
        result["fixed_l3_universe"] = len(eval_info.get("per_class") or [])
    path = cfg.run_result_path(slug, seed)
    if os.path.exists(path):
        raise S06ComparisonError(f"Результат run уже существует и не перезаписывается: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write_json(result, path)
    return result


def run_train(cfg: S06ComparisonConfig, device: torch.device) -> dict[str, Any]:
    """Этап Gate 3: обучение/переиспользование 30 финальных MLP и оценка.

    Порядок гарантирует границы: предусловия Gate 0/1 (fail closed) → загрузка
    протокола и плана → для ``none`` reuse checkpoint Gate 1 (без обучения),
    для selected-конфигураций обучение с нуля по замороженному протоколу →
    единая оценка raw top-1 на comparison_val каждого run с сохранением
    предсказаний. Resume отключён (fail closed): существующий run_result со
    статусом ``success`` — явный ``S06ComparisonError`` до любых записей;
    статусы ``failed``/``pending``/``skipped`` не считаются успешными, не
    перезаписываются и в ``n_done`` не входят. Успешный результат фиксирует
    ``hashes.predictions_sha256`` реального созданного parquet предсказаний.
    """
    cfg.validate()
    inputs = verify_preconditions(cfg)
    manifest, verification, protocol = inputs["manifest"], inputs["verification"], inputs["protocol"]
    # Неизменяемый план обязан быть зафиксирован до любых записей этапа;
    # write_plan сверяет конфигурацию и hash протокола, отсутствующий
    # fingerprint — отказ.
    _require_plan(cfg)
    model_cfg, loss_cfg, train_cfg, fixed_epochs = _resolved_train_sections(protocol)
    loaded_full = load_vectorizer_artifact(cfg.full_tfidf_path)
    maps = LabelMaps.from_dict(manifest["label_maps"])
    l3_labels = list(maps.l3)
    n_classes = tuple(maps.n_classes(lvl) for lvl in LEVELS)

    train_df = load_part(manifest, "outer_train")
    mon_df = load_part(manifest, "train_monitor")
    val_df = load_part(manifest, "comparison_val")
    y_outer = _encode_labels(maps, train_df, "outer_train")
    y_mon = _encode_labels(maps, mon_df, "train_monitor")
    y_val = _encode_labels(maps, val_df, "comparison_val")
    val_row_ids = val_df["row_id"].to_numpy(dtype=np.int64)
    class_weights = {
        h: compute_class_weights(y_outer[i], n_classes[i], loss_cfg.class_weight_mode[h],
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
    fixed_train_cfg = dataclasses.replace(train_cfg, epochs=fixed_epochs, patience=fixed_epochs)
    gate1_runs = _read_gate1_baseline_runs(cfg)
    protocol_sha = _sha256_file(cfg.protocol_path)

    summary: dict[str, Any] = {"trained": [], "reused": [], "failed": [], "skipped": []}

    for c in cfg.planned_configs():
        slug, method, budget = c["slug"], c["method"], c["budget"]
        stage_rows: list[dict[str, Any]] = []
        ckpt_root = os.path.join(cfg.config_dir(slug), "checkpoints")
        is_none = method == METHOD_NONE

        # --- векторизатор конфигурации и матрицы
        if is_none:
            builder = loaded_full.builder
            tfidf_artifact = os.path.abspath(cfg.full_tfidf_path)
            tfidf_sha = _full_tfidf_id(loaded_full)
            selection_artifact = None
            with _Measurement(
                device, f"transform comparison_val полным TF-IDF (конфиг none), chunk_rows={cfg.features_chunk_rows}"
            ) as m:
                X_val = _transform(loaded_full, val_df["name"].tolist(), cfg.features_chunk_rows)
            stage = m.stage_row("shared_vectorize_none", STAGE_VECTORIZE_CONFIG, "success")
            stage["artefact_bytes"] = None
            stage["measurement_note"] += "; матрицы train/monitor не строятся: обучение none не выполняется (reuse Gate 1)"
            if not _stage_recorded(cfg, "shared_vectorize_none", STAGE_VECTORIZE_CONFIG):
                _append_stage_rows(cfg.stages_path, [stage])
        else:
            refit_result_path = os.path.join(cfg.config_dir(slug), "refit_result.json")
            if not os.path.isfile(refit_result_path):
                raise S06ComparisonError(
                    f"Нет refit-результата для {slug}: сначала выполните этап refit (D07)."
                )
            sel_artifact = load_vectorizer_artifact(os.path.join(cfg.config_dir(slug), "tfidf_selected"))
            builder = sel_artifact.builder
            tfidf_artifact = os.path.abspath(os.path.join(cfg.config_dir(slug), "tfidf_selected"))
            tfidf_sha = (sel_artifact.meta.get("files") or {}).get("features.joblib", {}).get("sha256")
            selection_artifact = os.path.abspath(cfg.selection_dir(slug))
            with _Measurement(
                device,
                f"transform outer_train/train_monitor/comparison_val выбранным TF-IDF ({slug}), "
                f"chunk_rows={cfg.features_chunk_rows}; матрицы общие для всех seed конфигурации",
            ) as m:
                X_outer = builder.transform(train_df["name"].tolist(), chunk_rows=cfg.features_chunk_rows)
                X_mon = builder.transform(mon_df["name"].tolist(), chunk_rows=cfg.features_chunk_rows)
                X_val = builder.transform(val_df["name"].tolist(), chunk_rows=cfg.features_chunk_rows)
            stage = m.stage_row(f"shared_vectorize_{slug}", STAGE_VECTORIZE_CONFIG, "success")
            stage["artefact_bytes"] = None
            if not _stage_recorded(cfg, f"shared_vectorize_{slug}", STAGE_VECTORIZE_CONFIG):
                _append_stage_rows(cfg.stages_path, [stage])

        for seed in cfg.seeds:
            run_id = cfg.run_id(slug, seed)
            existing = _load_run_result(cfg, slug, seed)
            if existing is not None:
                if existing.get("status") == "success":
                    # Resume отключён: переиспользование успешного результата
                    # без полной проверяемой provenance запрещено — явный
                    # отказ до любых записей; новый прогон — новый каталог.
                    raise S06ComparisonError(
                        f"resume отключён: результат {run_id} уже существует (status=success) в "
                        f"{cfg.run_result_path(slug, seed)}: переиспользование не подтверждено; "
                        "используйте новый output_root, перезапись недопустима"
                    )
                # failed/pending/skipped-результат не считается завершённым,
                # не перезаписывается и в n_done не входит: запуск пропускается.
                log.warning(
                    "%s: существующий результат status=%s не засчитывается как done "
                    "и не перезаписывается; для перезапуска используйте новый каталог",
                    run_id, existing.get("status"),
                )
                summary["skipped"].append(run_id)
                continue
            try:
                if is_none:
                    # --- reuse baseline Gate 1: обучение не выполняется
                    gate1_row = gate1_runs[gate1_runs["run_id"] == f"seed_{seed}"]
                    if len(gate1_row) != 1:
                        raise S06ComparisonError(f"runs.csv Gate 1: нет уникальной строки seed_{seed}")
                    gate1_row = gate1_row.iloc[0].to_dict()
                    if gate1_row.get("status") != "success":
                        raise S06ComparisonError(
                            f"Gate 1 baseline seed {seed}: status={gate1_row.get('status')!r} "
                            f"({gate1_row.get('reason')}) — парный none-run недоступен"
                        )
                    ckpt_dir = gate1_row.get("checkpoint_path")
                    if not ckpt_dir or not os.path.isdir(ckpt_dir):
                        raise S06ComparisonError(f"checkpoint Gate 1 для seed {seed} не найден: {ckpt_dir!r}")
                    art = _verify_none_checkpoint(cfg, manifest, verification, loaded_full, ckpt_dir, seed, device)
                    _append_stage_rows(cfg.stages_path, [{
                        "run_id": run_id, "stage": STAGE_MLP_TRAIN, "status": "reused",
                        "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
                        "artefact_bytes": _dir_size(ckpt_dir),
                        "measurement_note": "обучение не выполнялось: checkpoint полного baseline "
                                            "переиспользован из Gate 1 (runs/baseline_full); затраты обучения — "
                                            "в stages.csv Gate 1, здесь только оценка",
                    }])
                    eval_info = _eval_run(
                        cfg, art.model, X_val, y_val[2], val_row_ids, l3_labels, run_id, stage_rows, device
                    )
                    saved_f1, saved_acc = gate1_row.get("l3_macro_f1"), gate1_row.get("l3_accuracy")
                    metrics_match = bool(
                        np.isclose(eval_info["l3_macro_f1"], float(saved_f1), rtol=1e-6, atol=1e-8)
                        and np.isclose(eval_info["l3_accuracy"], float(saved_acc), rtol=1e-6, atol=1e-8)
                    )
                    baseline_check = {
                        "gate1_run_id": f"seed_{seed}",
                        "gate1_l3_macro_f1": float(saved_f1), "gate1_l3_accuracy": float(saved_acc),
                        "recomputed_l3_macro_f1": eval_info["l3_macro_f1"],
                        "recomputed_l3_accuracy": eval_info["l3_accuracy"],
                        "metrics_match": metrics_match,
                    }
                    if not metrics_match:
                        raise S06ComparisonError(
                            f"Пересчитанные метрики none seed {seed} не совпали с Gate 1: {baseline_check}"
                        )
                    if stage_rows:
                        _append_stage_rows(cfg.stages_path, stage_rows)
                        stage_rows = []
                    result = _finalize_run_result(
                        cfg, slug=slug, method=method, budget=budget, seed=seed,
                        status="success", reason=None,
                        n_features=int(builder.n_features), n_char=int(builder.n_char_features),
                        n_word=int(builder.n_word_features),
                        train_info={
                            "reused_from_gate1": True, "fixed_epochs": int(fixed_epochs),
                            "checkpoint_dir_gate1": os.path.abspath(ckpt_dir),
                        },
                        eval_info=eval_info, checkpoint_dir=os.path.abspath(ckpt_dir),
                        tfidf_artifact=tfidf_artifact, tfidf_sha256=tfidf_sha,
                        selection_artifact=selection_artifact, train_stage=None,
                        baseline_check=baseline_check, device=device,
                    )
                else:
                    # --- обучение с нуля по замороженному протоколу
                    set_seed(seed)
                    model = SparseMLP(builder.n_features, n_classes, model_cfg)
                    with _Measurement(
                        device,
                        f"{run_id}: обучение SparseMLP с нуля на outer_train (n={X_outer.shape[0]}), "
                        f"признаков {builder.n_features}, ровно {fixed_epochs} эпох "
                        f"(scheduler={fixed_train_cfg.scheduler}, restore_best=False); протокол Gate 1",
                    ) as m:
                        history, best, lr_trace = train_model(
                            model, X_outer, y_outer, X_mon, y_mon,
                            loss_cfg, fixed_train_cfg, class_weights, device, seed,
                            restore_best=False,
                        )
                    stage = m.stage_row(run_id, STAGE_MLP_TRAIN, "success")
                    stage["artefact_bytes"] = None
                    _append_stage_rows(cfg.stages_path, [stage] + stage_rows)
                    stage_rows = []
                    ckpt_dir = os.path.join(ckpt_root, f"seed_{seed}")
                    save_artifact(
                        ckpt_dir, model, model_cfg, builder, maps,
                        RejectRule(0.5, False, False),
                        {"note": "S06-эксперимент: сравнение по raw top-1 на comparison_val; τ не применяется",
                         "tau_mode": "fixed"},
                        cw_json,
                        {
                            "experiment_run": run_id,
                            "role": "s06_selected_run",
                            "config": {
                                "protocol_sha256": protocol_sha,
                                "model": dataclasses.asdict(model_cfg),
                                "loss": dataclasses.asdict(loss_cfg),
                                "train": dataclasses.asdict(fixed_train_cfg),
                            },
                        },
                        {
                            "seed": int(seed),
                            "split_seed": int(manifest["policy"]["split_seed"]),
                            "n_features": int(builder.n_features),
                            "device": str(device),
                            "protocol": S06_COMPARISON_SCHEMA,
                            "config_slug": slug,
                            "method": method,
                            "budget": budget,
                            "train": {
                                "epochs_completed": len(history),
                                "fixed_epochs": int(fixed_epochs),
                                "scheduler": str(fixed_train_cfg.scheduler),
                                "best_epoch_monitor": int(best["epoch"]),
                                "early_stopping_metric": best["metric"],
                                "best_score_monitor": float(best["score"]),
                                "restored_best": bool(best.get("restored_best")),
                                "n_train_rows_used": int(best.get("n_train_rows_used", 0)),
                                "final_monitor_l3_macro_f1": (history[-1]["val_l3_macro_f1"] if history else None),
                                "final_monitor_l3_accuracy": (history[-1]["val_l3_accuracy"] if history else None),
                            },
                            "feature_stats": {
                                "outer_train": _matrix_stats(X_outer),
                                "train_monitor": _matrix_stats(X_mon),
                                "comparison_val": _matrix_stats(X_val),
                            },
                            "train_fingerprints": {
                                "outer_train_sha256": manifest["parts"]["outer_train"]["sha256"],
                                "train_monitor_sha256": manifest["parts"]["train_monitor"]["sha256"],
                            },
                            "tfidf": {
                                "artifact_dir": tfidf_artifact,
                                "features_sha256": tfidf_sha,
                                "derived_from_full_tfidf_id": verification.get("full_tfidf_id"),
                            },
                            "selection": {"artifact": selection_artifact},
                        },
                    )
                    eval_info = _eval_run(
                        cfg, model, X_val, y_val[2], val_row_ids, l3_labels, run_id, stage_rows, device
                    )
                    _append_stage_rows(cfg.stages_path, stage_rows)
                    stage_rows = []
                    result = _finalize_run_result(
                        cfg, slug=slug, method=method, budget=budget, seed=seed,
                        status="success", reason=None,
                        n_features=int(builder.n_features), n_char=int(builder.n_char_features),
                        n_word=int(builder.n_word_features),
                        train_info={
                            "epochs_completed": len(history),
                            "fixed_epochs": int(fixed_epochs),
                            "scheduler": str(fixed_train_cfg.scheduler),
                            "best_epoch_monitor": int(best["epoch"]),
                            "early_stopping_metric": best["metric"],
                            "best_score_monitor": float(best["score"]),
                            "restored_best": bool(best.get("restored_best")),
                            "n_train_rows_used": int(best.get("n_train_rows_used", 0)),
                            "final_monitor_l3_macro_f1": (history[-1]["val_l3_macro_f1"] if history else None),
                            "final_monitor_l3_accuracy": (history[-1]["val_l3_accuracy"] if history else None),
                            "train_seconds": m.seconds,
                            "peak_rss_bytes": m.peak_rss_bytes,
                            "peak_vram_bytes": m.peak_vram_bytes,
                            "lr_trace_tail": lr_trace[-3:] if lr_trace else [],
                        },
                        eval_info=eval_info, checkpoint_dir=os.path.abspath(ckpt_dir),
                        tfidf_artifact=tfidf_artifact, tfidf_sha256=tfidf_sha,
                        selection_artifact=selection_artifact, train_stage=None,
                        baseline_check=None, device=device,
                    )
                summary["trained"].append(run_id)
                log.info("%s: success (macro-F1=%.4f, acc=%.4f)", run_id,
                         result["comparison_val_eval"]["l3_macro_f1"],
                         result["comparison_val_eval"]["l3_accuracy"])
            except Exception as e:  # noqa: BLE001 — фиксируем failed run, остальные продолжаем
                log.exception("%s: failed", run_id)
                if stage_rows:
                    _append_stage_rows(cfg.stages_path, stage_rows)
                    stage_rows = []
                try:
                    _finalize_run_result(
                        cfg, slug=slug, method=method, budget=budget, seed=seed,
                        status="failed", reason=f"{type(e).__name__}: {e}"[:800],
                        n_features=None, n_char=None, n_word=None,
                        train_info=None, eval_info=None, checkpoint_dir=None,
                        tfidf_artifact=None, tfidf_sha256=None, selection_artifact=None,
                        train_stage=None, baseline_check=None, device=device,
                    )
                except Exception as inner:  # noqa: BLE001 — результат failed уже существует: не трогаем
                    log.warning("%s: не удалось записать failed-результат: %s", run_id, inner)
                summary["failed"].append(run_id)
    done = len(summary["trained"]) + len(summary["reused"])
    log.info("Этап train завершён: %d/%d runs успешно, failed=%d, reused=%d",
             done, len(cfg.planned_configs()) * len(cfg.seeds), len(summary["failed"]), len(summary["reused"]))
    summary["n_planned"] = len(cfg.planned_configs()) * len(cfg.seeds)
    summary["n_done"] = done
    return summary


def _matrix_stats(X: sp.csr_matrix) -> dict[str, Any]:
    from .features import matrix_stats

    return dict(matrix_stats(X))


def run_comparison_all(cfg: S06ComparisonConfig, device: torch.device) -> dict[str, Any]:
    """Полный контур Gate 2–3: строгая сверка плана → rankings → refit → train."""
    cfg.validate()
    # write_plan фиксирует план при первом вызове и сверяет config/protocol
    # hash при повторном: несовпадение — S06ComparisonError до любых записей.
    write_plan(cfg)
    index = run_rankings(cfg, device)
    refits = run_refit_selected(cfg)
    summary = run_train(cfg, device)
    return {"rankings": list(index["methods"]), "refits": list(refits), "train": summary}


__all__ = [
    "S06_COMPARISON_SCHEMA",
    "S06ComparisonConfig",
    "S06ComparisonError",
    "METHOD_NONE",
    "METHOD_GLOBAL",
    "METHOD_CLASS_AWARE",
    "METHOD_GXI",
    "SELECTION_METHODS",
    "DEFAULT_BUDGETS",
    "DEFAULT_SEEDS",
    "DEFAULT_CHAR_SHARE",
    "STAGES_COLUMNS",
    "write_plan",
    "verify_preconditions",
    "run_rankings",
    "run_refit_selected",
    "run_train",
    "run_comparison_all",
]
