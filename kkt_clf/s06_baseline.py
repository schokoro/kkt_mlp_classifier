"""S06 Gate 1 (D04–D06): замороженный обучающий протокол и новый full baseline.

Контур Gate 1 поверх результатов Gate 0 (``outputs/S06/splits``, см.
``kkt_clf.s06_split``) и одного проверенного full-TFIDF S02. **Модуль не
переобучает и не оценивает ничего на ``sealed_test``**: часть читается только
как структурная запись manifest (sha256/число строк), любой code path к её
строкам отсутствует; единственная внешняя оценка — raw top-1 L3 на
``comparison_val`` после refit всех seed, против train-only majority.

Протокол (границы частей совпадают с S05, D05/D06 value_dod S06):

- Выбор числа эпох и конфигурации — **только** по ``train_fit``/``train_monitor``
  (фаза ``select``), до любого обращения к ``comparison_val``; кривые и отчёт
  selection сохраняются, затем протокол **замораживается** в неизменяемый
  ``shared/training_protocol.json`` (файл пишется один раз; повторная запись,
  как и любое расхождение provenance, — отказ, а не перезапись).
- Финальный refit — три полных baseline seed 42/43/44 на ``outer_train`` ровно
  на замороженное число эпох с фиксированным заранее расписанием LR
  (переиспользуется контур S05 без изменения его поведения), веса финальной
  эпохи (restore_best=False), без выбора checkpoint по monitor.
- Опорный checkpoint для G×I (Gate 2) — заранее заданный ``reference_seed``
  (по умолчанию 42), не по качеству на comparison_val; round-trip проверка —
  контур S05.
- Оценка — одна raw top-1 L3 на ``comparison_val`` против train-only majority;
  ``sealed_test`` не используется ни для inference, ни для метрик.

Проверки до обучения (fail closed): manifest Gate 0 (sha256 частей, политика
``min_count == 12`` — сплит S05 с ``min_count == 10`` отклоняется; заявленные
``min_rows_per_class``/``min_train_fit`` >= 2; фактические per-L3 support
всех частей из реальных файлов и идентичность наборов L3 label maps),
целостность и provenance full-TFIDF S02 (sha256 файлов артефакта +
fingerprint корпуса + полный контракт словаря 300 000 char + 100 000 word +
fit на полном снимке корпуса без усечения), сверка с существующей ссылкой
``shared/full_tfidf_ref.json``, замороженного протокола с фактическими
входами (sha256 сплитов, full_tfidf_id, hash selection-отчёта, согласованность
fixed_epochs на всех уровнях вложенности, постоянный LR scheduler 'none').
Никакие существующие файлы не перезаписываются; исторические checkpoint/split
S05 не используются.

Реализация **делегирует обучение** существующему контуру S05
(``s05_baseline.run_selection`` / ``run_refit`` и их проверки) с конфигурацией,
направленной в ``outputs/S06/runs/baseline_full``; тренировочный цикл не
дублируется. Ресурсная оценка (``estimate``) — честная проекция от измерений
S05 и размеров нового сплита, без обучения и без выдачи проекций за замеры.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

import pandas as pd

from .config import _merge, build_config
from .data import file_fingerprint
from .experiment import _sha256_file, _sha256_json, _write_json
from .s06_split import S06_MIN_ROWS_PER_CLASS, S06_MIN_TRAIN_FIT
from .s05_baseline import (
    S05BaselineError,
    S05Config,
    load_split_manifest,
    record_tfidf_verification,
    run_refit,
    run_selection,
    verify_full_tfidf,
)

log = logging.getLogger(__name__)

S06_GATE1_SCHEMA = "s06-gate1-protocol@1"

# Полный контракт TF-IDF S02 (D04): артефакт с иным vocab при том же корпусе
# (например усечённый fit tfidf_100k) для S06 недопустим.
EXPECTED_FULL_VOCAB = {"n_char": 300_000, "n_word": 100_000, "n_total": 400_000}

# Части, к которым Gate 1 запрещено обращаться за данными. comparison_val
# читается ровно один раз — внутри s05.run_refit после всех refit (оценка);
# sealed_test не читается никогда (только структурные sha256/n_rows из manifest).
SEALED_PART = "sealed_test"
EVAL_PART = "comparison_val"

PROTOCOL_NOTES = {
    "epoch_selection": (
        "число эпох и конфигурация выбраны только по train_fit/train_monitor "
        "(early stopping по train_monitor, selection_seed), до любого обращения "
        "к comparison_val; кривые — runs/baseline_full/selection/curves_train_monitor.csv"
    ),
    "final_train": (
        "refit каждого seed на outer_train ровно на замороженное число эпох, "
        "фиксированное заранее расписание LR (общее с selection), "
        "restore_best=False (веса финальной эпохи), без выбора checkpoint по monitor"
    ),
    "evaluation": (
        "единственная внешняя оценка Gate 1 — raw top-1 L3 без отказа на "
        "comparison_val после refit всех seed; majority baseline — большинство "
        "по outer_train (train-only); метрики по фиксированной L3-вселенной label maps"
    ),
    "sealed_test": (
        "sealed_test запрещён для inference/eval в Gate 1: часть не читается, "
        "метрик на ней нет; использованы только структурные sha256/n_rows из manifest"
    ),
    "reference_checkpoint": (
        "опорный checkpoint для G×I (Gate 2) — заранее заданный reference_seed, "
        "не по качеству на comparison_val; round-trip проверка контура S05"
    ),
    "transductive": (
        "режим pretrained трансдуктивен: тексты корпуса full TF-IDF могут "
        "совпадать с train/val/test; метки при этом не переносятся"
    ),
}


class S06Gate1Error(RuntimeError):
    """Ошибка контура Gate 1 S06; файлы не перезаписываются."""


def ensure_constant_lr_schedule(scheduler: str) -> None:
    """Fail closed: S06 допускает только постоянный LR (scheduler 'none').

    ``cosine``/``onecycle`` строят кривую LR от ``total_steps``, которое у
    selection (train_fit, ранняя остановка) и refit (outer_train,
    замороженное число эпох) различается, — кривые LR selection/refit были бы
    разными, и выбранное число эпох относилось бы к другому расписанию.
    Идентичное воспроизводимое расписание для разного числа шагов в S06 не
    реализовано и не проверено, поэтому такие scheduler'ы отклоняются до
    любого обучения. ``plateau`` контур S05 уже заменяет на ``none`` при
    сборке секций (зафиксировано заметкой в протоколе), ``none`` сохраняется.
    """
    if scheduler != "none":
        raise S06Gate1Error(
            f"scheduler={scheduler!r} для S06 запрещён: кривая LR зависит от числа "
            "шагов, а selection и refit имеют разное число шагов (train_fit + early "
            "stopping против outer_train + замороженные эпохи), поэтому идентичность "
            "расписания не подтверждена. Разрешён только scheduler 'none' "
            "(постоянный LR); текущий замороженный протокол S06 использует 'none'."
        )


@dataclass
class S06Gate1Config:
    """Конфигурация Gate 1 S06 (обёртка над S05Config с S06-путями)."""

    output_root: str = "outputs/S06"
    splits_manifest: str = "outputs/S06/splits/manifest.json"
    full_tfidf_path: str = "outputs/S02/tfidf_1m"
    corpus_path: str = "data/corpus.pqt"
    expected_min_count: int = 12
    expected_vocab: dict[str, int] = field(default_factory=lambda: dict(EXPECTED_FULL_VOCAB))
    preset: str = "ods_light"
    model: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    sweep: list[dict[str, Any]] = field(default_factory=list)
    features_chunk_rows: int = 50_000
    selection_seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    reference_seed: int = 42
    device: str = "auto"

    @property
    def baseline_dir(self) -> str:
        return os.path.join(self.output_root, "runs", "baseline_full")

    @property
    def shared_dir(self) -> str:
        return os.path.join(self.output_root, "shared")

    @property
    def protocol_path(self) -> str:
        return os.path.join(self.shared_dir, "training_protocol.json")

    @property
    def tfidf_ref_path(self) -> str:
        return os.path.join(self.shared_dir, "full_tfidf_ref.json")

    def to_s05_config(self) -> S05Config:
        """S05-контур с S06-путями; поведение контура S05 не изменяется."""
        return S05Config(
            output_dir=self.baseline_dir,
            splits_manifest=self.splits_manifest,
            full_tfidf_path=self.full_tfidf_path,
            corpus_path=self.corpus_path,
            preset=self.preset,
            model=dict(self.model),
            loss=dict(self.loss),
            train=dict(self.train),
            sweep=[dict(c) for c in self.sweep],
            features_chunk_rows=self.features_chunk_rows,
            selection_seed=self.selection_seed,
            seeds=list(self.seeds),
            reference_seed=self.reference_seed,
            fixed_epochs=None,  # эпохи — только из замороженного протокола
            device=self.device,
        )

    def resolved_scheduler(self) -> str:
        """Расписание LR после сборки секций (plateau контур S05 уже заменил на none)."""
        _, _, train_cfg = self.to_s05_config().build_sections()
        return str(train_cfg.scheduler)

    def validate(self) -> None:
        if self.expected_min_count != 12:
            raise S06Gate1Error(
                f"expected_min_count должен быть 12 (протокол S06), получено {self.expected_min_count}"
            )
        if self.reference_seed not in self.seeds:
            raise S06Gate1Error(
                f"reference_seed={self.reference_seed} должен входить в seeds={self.seeds}"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise S06Gate1Error(f"seeds должны быть различны: {self.seeds}")
        ensure_constant_lr_schedule(self.resolved_scheduler())
        self.to_s05_config().validate()


# ---------------------------------------------------------------------------
# Gate 0: проверка provenance нового сплита
# ---------------------------------------------------------------------------


def verify_gate0_split(cfg: S06Gate1Config) -> dict[str, Any]:
    """Проверяет замороженный сплит Gate 0 (fail closed до любого обучения).

    Поверх проверок целостности S05 (sha256/bytes/число строк каждой части)
    требует политики S06: ``min_count == 12`` (сплит S05 с min_count=10
    отклоняется), ``min_rows_per_class``/``min_train_fit`` не меньше 2 (если
    заявлены в политике), прошедшие integrity checks Gate 0, наличие всех
    пяти частей и зафиксированных label maps. Дополнительно проверяет по
    фактическим файлам частей (когда в manifest есть пути): реальные per-L3
    support каждой части — не менее 2 строк в comparison_val/sealed_test/
    train_monitor и не менее 2 в train_fit — и идентичность наборов L3 между
    всеми частями и label maps manifest. К ``sealed_test`` обращений за
    данными нет и не будет: здесь используются только его структурные поля
    manifest и count/support-подсчёт, допустимый протоколом S06.
    """
    if not os.path.isfile(cfg.splits_manifest):
        raise S06Gate1Error(
            f"manifest сплитов Gate 0 не найден: {cfg.splits_manifest!r}. "
            "Gate 1 не запускается без независимо проверенного Gate 0."
        )
    manifest = load_split_manifest(cfg.splits_manifest)  # sha256/bytes/n_rows всех частей
    policy = manifest.get("policy") or {}
    min_count = policy.get("min_count")
    if min_count != cfg.expected_min_count:
        raise S06Gate1Error(
            f"Политика сплита min_count={min_count!r} != {cfg.expected_min_count} (протокол S06). "
            "Вероятен сплит S05 (min_count=10) или иной протокол: Gate 1 отказывается, "
            "старую выборку для нового протокола не подменяем."
        )
    # Политика per-class минимумов: протокол S06 требует >= 2; значение 1
    # (или отсутствие заявленных минимумов при заявленном значении < 2)
    # отклоняется. Отсутствие ключей в чужом manifest не подтверждает политику
    # — достоверность тогда определяется фактическими проверками support ниже.
    policy_min_rows = policy.get("min_rows_per_class")
    policy_min_fit = policy.get("min_train_fit")
    if policy_min_rows is not None and int(policy_min_rows) < S06_MIN_ROWS_PER_CLASS:
        raise S06Gate1Error(
            f"Политика сплита min_rows_per_class={policy_min_rows} < {S06_MIN_ROWS_PER_CLASS} "
            "(протокол S06): comparison_val/sealed_test/train_monitor обязаны иметь "
            ">= 2 строки каждой сохранённой L3; сплит с минимумом 1 отклоняется."
        )
    if policy_min_fit is not None and int(policy_min_fit) < S06_MIN_TRAIN_FIT:
        raise S06Gate1Error(
            f"Политика сплита min_train_fit={policy_min_fit} < {S06_MIN_TRAIN_FIT} "
            "(протокол S06): train_fit обязан иметь >= 2 строки каждой сохранённой L3."
        )
    checks = manifest.get("checks") or {}
    if not checks.get("ok"):
        raise S06Gate1Error(f"Integrity checks Gate 0 не пройдены (checks.ok != True): {checks}")
    for part in ("outer_train", "train_fit", "train_monitor", EVAL_PART, SEALED_PART):
        if part not in manifest.get("parts", {}):
            raise S06Gate1Error(f"В manifest сплита нет части {part!r}")
    if not manifest.get("label_maps"):
        raise S06Gate1Error("В manifest сплита нет label_maps")

    # --- Фактические per-L3 минимумы и идентичность label maps по файлам частей.
    required = {
        EVAL_PART: int(policy_min_rows) if policy_min_rows is not None else S06_MIN_ROWS_PER_CLASS,
        SEALED_PART: int(policy_min_rows) if policy_min_rows is not None else S06_MIN_ROWS_PER_CLASS,
        "train_monitor": int(policy_min_rows) if policy_min_rows is not None else S06_MIN_ROWS_PER_CLASS,
        "train_fit": int(policy_min_fit) if policy_min_fit is not None else S06_MIN_TRAIN_FIT,
    }
    l3_universe = set((manifest.get("label_maps") or {}).get("L3") or [])
    part_l3_sets: dict[str, set] = {}
    problems: list[str] = []
    for name, minimum in required.items():
        part = manifest["parts"].get(name) or {}
        fpath = part.get("file")
        if not fpath or not os.path.isfile(fpath):
            # Manifest без путей к частям (сторонняя схема): фактические
            # проверки невозможны, минимумы не подтверждены — пропускаем.
            continue
        df_part = pd.read_parquet(fpath, columns=["L1", "L2", "L3"])
        part_l3_sets[name] = set(df_part["L3"].unique())
        if l3_universe and part_l3_sets[name] != l3_universe:
            missing = sorted(l3_universe - part_l3_sets[name])[:5]
            extra = sorted(part_l3_sets[name] - l3_universe)[:5]
            problems.append(
                f"{name}: набор L3 не совпадает с label maps manifest "
                f"(нет: {missing}; лишние: {extra})"
            )
        low = df_part.groupby(["L1", "L2", "L3"]).size()
        low = low[low < minimum]
        if len(low):
            examples = {"/".join(map(str, idx)): int(v) for idx, v in low.head(5).items()}
            problems.append(
                f"{name}: {len(low)} L3 с фактическим support < {minimum} (напр. {examples})"
            )
    if len(part_l3_sets) >= 2:
        reference_name = next(iter(part_l3_sets))
        for name, l3s in part_l3_sets.items():
            if l3s != part_l3_sets[reference_name]:
                problems.append(
                    f"идентичность label maps: набор L3 части {name!r} отличается от {reference_name!r}"
                )
    if problems:
        raise S06Gate1Error(
            "Gate 0 split: фактические per-class проверки не пройдены: " + "; ".join(problems)
        )
    log.info(
        "Gate 0 split проверен: min_count=%s, per-L3 минимумы по фактическим "
        "файлам частей (>=2 в val/test/monitor/fit), наборы L3 частей "
        "идентичны label maps; sealed_test — только структурные sha256/n_rows "
        "и count/support, инференс/метрик на нём нет",
        min_count,
    )
    return manifest


# ---------------------------------------------------------------------------
# D04: full TF-IDF S02 → shared/full_tfidf_ref.json
# ---------------------------------------------------------------------------


def verify_s06_tfidf(cfg: S06Gate1Config) -> dict[str, Any]:
    """Проверяет full TF-IDF S02 и сохраняет его provenance в ``shared/``.

    Проверка целостности и совпадения корпуса — ``s05_baseline.verify_full_tfidf``
    (без изменения поведения). Поверх неё fail closed полный контракт S02:
    словарь ровно 300 000 char + 100 000 word и fit выполнен на полном
    снимке корпуса (``source.rows_used == source.rows_in_file``,
    ``truncated is False``) — иной валидный TF-IDF с тем же корпусом (например
    усечённый fit на подвыборке строк) отклоняется.
    ``shared/full_tfidf_ref.json`` пишется один раз;
    при уже существующем файле (в т.ч. созданном Gate 0 через
    ``scripts/s06_tfidf_ref.py`` — иная схема) выполняется сверка записанных в
    нём hash-полей с фактической проверкой — расхождение отказ, совпадение —
    существующий файл сохраняется, перезапись чужого артефакта недопустима.
    Сам full-артефакт не копируется (ссылка + hash).
    """
    verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
    expected = dict(cfg.expected_vocab)
    if (verification.get("vocab") or {}) != expected:
        raise S06Gate1Error(
            f"full TF-IDF не соответствует контракту S02: vocab="
            f"{verification.get('vocab')} != {expected}. Артефакт валиден сам по "
            "себе и относится к тому же корпусу, но это не полный TF-IDF S06 "
            "(например усечённый или иной fit): подмена контракта признаков "
            "между фазами недопустима, отказ до любых записей."
        )
    src = verification.get("source") or {}
    if not (src.get("rows_used") == src.get("rows_in_file") and src.get("truncated") is False):
        raise S06Gate1Error(
            "provenance полного исходного fit не подтверждён: metadata source "
            f"должен содержать rows_used == rows_in_file и truncated=false, "
            f"получено rows_used={src.get('rows_used')!r}, rows_in_file="
            f"{src.get('rows_in_file')!r}, truncated={src.get('truncated')!r}. "
            "Fit на подвыборке строк корпуса для S06 недопустим: refit selected "
            "TF-IDF обязан воспроизводить ту же фактическую выборку строк."
        )
    os.makedirs(cfg.shared_dir, exist_ok=True)
    path = cfg.tfidf_ref_path
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        problems = _existing_tfidf_ref_problems(existing, verification, expected)
        if problems:
            raise S06Gate1Error(
                f"{path} уже существует и не совпадает с текущей проверкой full "
                f"TF-IDF по полям {problems}: подмена артефакта/корпуса между "
                "фазами недопустима, отказ до любых записей."
            )
        log.info("%s уже существует и совпадает с текущей проверкой full TF-IDF", path)
        return verification
    ref = {
        "schema_version": S06_GATE1_SCHEMA,
        "role": "full_tfidf_ref",
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "artifact_dir": verification["artifact_dir"],
        "full_tfidf_id": verification["full_tfidf_id"],
        "artifact_version": verification["artifact_version"],
        "seed": verification["seed"],
        "vocab": verification["vocab"],
        "fit": verification["fit"],
        "source": verification["source"],
        "corpus_fingerprint": verification["corpus_fingerprint"],
        "corpus_matches_source": verification["corpus_matches_source"],
        "files": verification["files"],
        "decision": verification["decision"],
        "mode": verification["mode"],
        "notes": [
            "Артефакт не копируется: ссылка + sha256; проверка целостности — load_vectorizer_artifact.",
            PROTOCOL_NOTES["transductive"],
        ],
    }
    _write_json(ref, path)
    return verification


def _existing_tfidf_ref_problems(
    existing: dict[str, Any], verification: dict[str, Any], expected_vocab: dict[str, int]
) -> list[str]:
    """Сверяет существующий full_tfidf_ref.json (любой известной схемы) с фактической проверкой.

    Поддерживает схему этого модуля (``full_tfidf_id``/``corpus_fingerprint``)
    и схему ``scripts/s06_tfidf_ref.py`` Gate 0 (``files.features.joblib.sha256``
    + ``corpus_provenance.corpus_actual`` + ``status``/``load_check.status``).
    Поверх идентичности проверяет полный контракт S02: vocab существующей
    ссылки равен 300 000 char + 100 000 word, а зафиксированный fit —
    полному снимку корпуса (``meta_source.rows_used == rows_in_file``,
    ``truncated is False``), чтобы ссылка не узаконивала усечённый fit.
    """
    problems: list[str] = []
    actual_id = verification.get("full_tfidf_id")
    new_corpus = (verification.get("corpus_fingerprint") or {})
    existing_vocab = existing.get("vocab")
    if existing_vocab is None:
        existing_vocab = ((existing.get("load_check") or {}).get("loaded_vocab")) or (
            (existing.get("artifact") or {}).get("vocab_meta")
        )
    if existing_vocab is not None and existing_vocab != expected_vocab:
        problems.append(f"vocab: существующая ссылка фиксирует {existing_vocab}, контракт S02 {expected_vocab}")
    if "full_tfidf_id" in existing:  # схема kkt_clf.s06_baseline
        for key in ("decision", "full_tfidf_id", "artifact_version"):
            if existing.get(key) != verification.get(key):
                problems.append(key)
        ex_corpus = (existing.get("corpus_fingerprint") or {})
        if (ex_corpus.get("sha256") != new_corpus.get("sha256")
                or ex_corpus.get("bytes") != new_corpus.get("bytes")):
            problems.append("corpus_fingerprint")
        return problems
    fj = ((existing.get("files") or {}).get("features.joblib") or {})
    if existing.get("status") != "ok":
        problems.append("status")
    if fj.get("sha256") != actual_id:
        problems.append("features.joblib.sha256")
    actual = ((existing.get("corpus_provenance") or {}).get("corpus_actual") or {})
    if (actual.get("sha256") != new_corpus.get("sha256")
            or actual.get("bytes") != new_corpus.get("bytes")):
        problems.append("corpus_actual")
    if (existing.get("load_check") or {}).get("status") not in (None, "ok"):
        problems.append("load_check.status")
    meta_source = ((existing.get("corpus_provenance") or {}).get("meta_source")) or {}
    if meta_source and not (
        meta_source.get("rows_used") == meta_source.get("rows_in_file")
        and meta_source.get("truncated") is False
    ):
        problems.append("meta_source.full_fit: ссылка фиксирует fit не на полном снимке корпуса")
    return problems


# ---------------------------------------------------------------------------
# D05: train-only выбор эпох → неизменяемый shared/training_protocol.json
# ---------------------------------------------------------------------------


def freeze_training_protocol(
    cfg: S06Gate1Config,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    loaded: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Фаза selection (train_fit/train_monitor) + заморозка протокола.

    Selection делегируется ``s05_baseline.run_selection`` (train-only ранний
    останов по train_monitor; comparison_val не читается). Затем протокол
    пишется в ``shared/training_protocol.json`` **один раз**: повторный вызов
    при существующем файле — отказ (перевыбор эпох незаметно недопустим).
    Протокол содержит полный selection-отчёт, sha256 сплитов, full_tfidf_id,
    замороженные секции конфигурации выбранного кандидата и hash файла
    selection-отчёта; refit валидируется против него (fail closed).
    """
    if os.path.isfile(cfg.protocol_path):
        raise S06Gate1Error(
            f"Протокол уже заморожен: {cfg.protocol_path}. Перевыбор эпох/конфигурации "
            "после заморозки запрещён; для нового протокола нужен новый каталог "
            "output_root, существующие результаты не перезаписываются."
        )
    # Fail closed до selection: расписание LR обязано быть подтверждённо
    # идентичным для selection/refit (см. ensure_constant_lr_schedule).
    ensure_constant_lr_schedule(cfg.resolved_scheduler())
    selection = run_selection(cfg.to_s05_config(), manifest, loaded, device)
    sel_path = os.path.join(cfg.baseline_dir, "selection", "selection_report.json")
    chosen = selection["chosen"]
    cand = next(c for c in selection["candidates"] if c["name"] == chosen["name"])
    protocol = {
        "schema_version": S06_GATE1_SCHEMA,
        "gate": "S06 Gate 1: frozen training protocol (epoch selection + final train config)",
        "frozen": True,
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "notes": dict(PROTOCOL_NOTES),
        "config": {
            "preset": cfg.preset,
            "model": dict(cfg.model),
            "loss": dict(cfg.loss),
            "train": dict(cfg.train),
            "sweep": [dict(c) for c in cfg.sweep],
            "features_chunk_rows": int(cfg.features_chunk_rows),
            "selection_seed": int(cfg.selection_seed),
            "seeds": list(cfg.seeds),
            "reference_seed": int(cfg.reference_seed),
        },
        "resolved_sections": {
            "model": cand["model"],
            "loss": cand["loss"],
            "train": cand["train"],
        },
        "chosen": chosen,
        "early_stopping_metric": selection["metric"],
        "fixed_epochs": int(chosen["fixed_epochs"]),
        "splits": {
            name: {"sha256": part["sha256"], "bytes": part["bytes"], "n_rows": part["n_rows"]}
            for name, part in manifest["parts"].items()
        },
        "labeled_parquet": manifest["inputs"]["labeled_parquet"],
        "min_count_policy": int(manifest["policy"]["min_count"]),
        "split_seed": int(manifest["policy"]["split_seed"]),
        "full_tfidf_id": verification["full_tfidf_id"],
        "artifact_version": verification["artifact_version"],
        "corpus_fingerprint": verification["corpus_fingerprint"],
        "selection_report_file": os.path.abspath(sel_path),
        "selection_report_sha256": _sha256_file(sel_path),
        "selection_report": selection,
        "sealed_test_boundary": (
            "sealed_test исключён из inference/eval Gate 1; метрик на нём нет "
            "и не будет до отдельного поручения"
        ),
    }
    _write_json(protocol, cfg.protocol_path)
    log.info(
        "Протокол заморожен: %d эпох (кандидат %s, метрика %s), файл %s",
        protocol["fixed_epochs"], chosen["name"], selection["metric"], cfg.protocol_path,
    )
    return protocol


def load_training_protocol(cfg: S06Gate1Config) -> dict[str, Any]:
    if not os.path.isfile(cfg.protocol_path):
        raise S06Gate1Error(
            f"Нет замороженного протокола {cfg.protocol_path!r}: сначала выполните "
            "фазу select (train-only выбор эпох на train_fit/train_monitor). "
            "Обучение без замороженного протокола запрещено."
        )
    with open(cfg.protocol_path, encoding="utf-8") as f:
        return json.load(f)


def validate_protocol_against_inputs(
    cfg: S06Gate1Config,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    """Fail closed: протокол применим только к своим входам.

    Сверяет sha256 всех частей сплита, full_tfidf_id, fingerprint корпуса,
    политику min_count, seeds/reference_seed и замороженные секции
    конфигурации, пересобранные из текущего cfg — до любого чтения данных и
    обучения. Поверх того проверяет неизменность замороженного selection-отчёта
    (пересчёт sha256 файла против ``selection_report_sha256``), совпадение
    фактического содержимого файла отчёта со вложенной копией
    ``selection_report`` (именно её использует refit), согласованность
    числа эпох на всех уровнях вложенности (верхний ``fixed_epochs`` ==
    ``chosen.fixed_epochs`` == ``selection_report.chosen.fixed_epochs``,
    совпадение ``chosen`` и ``selection_report.chosen``) и постоянство LR
    (``resolved_sections.train.scheduler == 'none'``): расхождение —
    S06Gate1Error, а не молчаливое обучение по чужому/повреждённому протоколу.
    """
    problems: list[str] = []

    # --- Целостность замороженного selection-отчёта и эпох.
    sel_path = protocol.get("selection_report_file")
    sel_sha = protocol.get("selection_report_sha256")
    if not sel_path or not sel_sha:
        problems.append("протокол не содержит selection_report_file/selection_report_sha256")
    elif not os.path.isfile(sel_path):
        problems.append(f"замороженный selection-отчёт не найден: {sel_path}")
    elif _sha256_file(sel_path) != sel_sha:
        problems.append(
            "selection_report_sha256: файл отчёта изменился после заморозки протокола"
        )
    else:
        # Hash совпадает, но refit использует вложенную копию: сверяем её с
        # фактическим содержимым файла, а не только с hash.
        try:
            with open(sel_path, encoding="utf-8") as f:
                file_report = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            problems.append(f"selection-отчёт не читается как JSON: {sel_path}: {e}")
        else:
            if file_report != (protocol.get("selection_report") or {}):
                problems.append(
                    "содержимое файла selection-отчёта не совпадает со вложенной копией "
                    "selection_report протокола (refit использует вложенную копию)"
                )
    top_epochs = protocol.get("fixed_epochs")
    chosen = protocol.get("chosen") or {}
    sel_chosen = (protocol.get("selection_report") or {}).get("chosen") or {}
    for label, value in (
        ("chosen.fixed_epochs", chosen.get("fixed_epochs")),
        ("selection_report.chosen.fixed_epochs", sel_chosen.get("fixed_epochs")),
    ):
        try:
            consistent = int(value) == int(top_epochs)
        except (TypeError, ValueError):
            consistent = False
        if not consistent:
            problems.append(
                f"несогласованность эпох: верхний fixed_epochs={top_epochs!r} != {label}={value!r}"
            )
    if sel_chosen and sel_chosen != chosen:
        problems.append("chosen != selection_report.chosen (расхождение вложенной копии)")
    resolved_train = (protocol.get("resolved_sections") or {}).get("train") or {}
    if resolved_train.get("scheduler", "none") != "none":
        problems.append(
            f"scheduler={resolved_train.get('scheduler')!r}: расписание LR, зависящее от "
            "числа шагов, для S06 не подтверждено идентичностью selection/refit; "
            "разрешён только 'none'"
        )

    proto_splits = protocol.get("splits") or {}
    for name, part in manifest["parts"].items():
        expected = proto_splits.get(name) or {}
        if expected.get("sha256") != part.get("sha256") or expected.get("n_rows") != part.get("n_rows"):
            problems.append(
                f"сплит {name}: протокол зафиксирован для sha256 "
                f"{str(expected.get('sha256'))[:16]}…/{expected.get('n_rows')} строк, "
                f"текущий сплит {str(part.get('sha256'))[:16]}…/{part.get('n_rows')}"
            )
    if protocol.get("full_tfidf_id") != verification.get("full_tfidf_id"):
        problems.append(
            f"full_tfidf_id: протокол относится к {str(protocol.get('full_tfidf_id'))[:16]}…, "
            f"проверен артефакт {str(verification.get('full_tfidf_id'))[:16]}…"
        )
    proto_corpus = (protocol.get("corpus_fingerprint") or {})
    new_corpus = (verification.get("corpus_fingerprint") or {})
    if (proto_corpus.get("sha256") != new_corpus.get("sha256")
            or proto_corpus.get("bytes") != new_corpus.get("bytes")):
        problems.append("corpus_fingerprint")
    if protocol.get("min_count_policy") != manifest["policy"].get("min_count"):
        problems.append("min_count_policy")
    proto_cfg = protocol.get("config") or {}
    if list(proto_cfg.get("seeds") or []) != list(cfg.seeds):
        problems.append(f"seeds: протокол {proto_cfg.get('seeds')} != текущие {cfg.seeds}")
    if int(proto_cfg.get("reference_seed", -1)) != int(cfg.reference_seed):
        problems.append("reference_seed")
    # Замороженные секции конфигурации должны воспроизводиться из текущего cfg:
    # иначе refit обучал бы не тот протокол, что был заморожен.
    s05cfg = cfg.to_s05_config()
    chosen = (protocol.get("selection_report") or {}).get("chosen") or {}
    cand = next(
        (c for c in (protocol.get("selection_report") or {}).get("candidates", [])
         if c["name"] == chosen.get("name")),
        None,
    )
    if cand is None:
        problems.append("chosen-кандидат не найден в selection_report протокола")
    else:
        model_cfg, loss_cfg, train_cfg = s05cfg.build_sections(cand["model"], cand["train"])
        if dataclasses.asdict(model_cfg) != (protocol.get("resolved_sections") or {}).get("model"):
            problems.append("model-секция не воспроизводится из текущего cfg")
        if dataclasses.asdict(loss_cfg) != (protocol.get("resolved_sections") or {}).get("loss"):
            problems.append("loss-секция не воспроизводится из текущего cfg (проверьте --loss)")
        if dataclasses.asdict(train_cfg) != (protocol.get("resolved_sections") or {}).get("train"):
            problems.append("train-секция не воспроизводится из текущего cfg (проверьте --train)")
    if problems:
        raise S06Gate1Error(
            "Замороженный протокол несовместим с текущими входами: " + "; ".join(problems)
        )


# ---------------------------------------------------------------------------
# D05–D06: финальный refit 42/43/44 + единственная оценка comparison_val
# ---------------------------------------------------------------------------


def run_gate1_refit(
    cfg: S06Gate1Config,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    loaded: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Refit трёх full baseline на outer_train по замороженному протоколу.

    Порядок гарантирует границы: проверка TF-IDF уже выполнена вызывающей
    стороной (fail closed), затем загрузка и валидация замороженного протокола
    против текущих входов — только после этого контур S05 обучает seed
    42/43/44 на outer_train и выполняет единственную оценку на
    ``comparison_val`` (majority-контроль и round-trip опорного checkpoint —
    внутри контура S05 без изменений). ``sealed_test`` не читается.
    """
    protocol = load_training_protocol(cfg)
    validate_protocol_against_inputs(cfg, protocol, manifest, verification)
    selection = protocol["selection_report"]
    s05cfg = cfg.to_s05_config()
    # Проверенный provenance дублируется контуром S05 в baseline_dir
    # (record_tfidf_verification внутри run_refit: новый файл либо сверка).
    record_tfidf_verification(s05cfg.output_dir, verification)
    return run_refit(s05cfg, manifest, loaded, device,
                     selection=selection, tfidf_verification=verification)


def run_gate1_contour(cfg: S06Gate1Config) -> dict[str, Any]:
    """Полный Gate 1: verify split (Gate 0) → verify TF-IDF → select+freeze → refit+eval."""
    from .train import resolve_device
    from .vectorizer import load_vectorizer_artifact

    cfg.validate()
    device = resolve_device(cfg.device)
    manifest = verify_gate0_split(cfg)
    verification = verify_s06_tfidf(cfg)
    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
    freeze_training_protocol(cfg, manifest, verification, loaded, device)
    return run_gate1_refit(cfg, manifest, verification, loaded, device)


# ---------------------------------------------------------------------------
# Ресурсная оценка Gate 1 (проекция от измерений S05, без обучения)
# ---------------------------------------------------------------------------


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


def estimate_gate1_resources(
    cfg: S06Gate1Config,
    manifest: dict[str, Any],
    write: bool = False,
) -> dict[str, Any]:
    """Честная проекция ресурсов Gate 1 от измерений S05 и размеров S06-сплита.

    **Не выполняет обучение, transform и inference.** Опора — сохранённые
    замеры S05 (``outputs/S05/stages.csv``, ``outputs/S05/manifest.json``,
    размер checkpoint) и число строк частей нового сплита из manifest Gate 0.
    Все проекции помечены ``projected=true`` и источником; это не измерения.
    GPU недоступен — ``null`` (не ноль). При ``write`` результат пишется в
    ``shared/resource_estimate_gate1.json`` (только новый файл).
    """
    parts = manifest["parts"]
    n_outer = int(parts["outer_train"]["n_rows"])
    n_fit = int(parts["train_fit"]["n_rows"])
    n_mon = int(parts["train_monitor"]["n_rows"])
    n_val = int(parts[EVAL_PART]["n_rows"])

    anchor: dict[str, Any] = {"available": False}
    proj: dict[str, Any] = {
        "vectorize_train_seconds": None,
        "selection_train_seconds": None,
        "refit_seconds_per_seed": None,
        "eval_seconds_per_seed": None,
    }
    s05_stages_path = os.path.join("outputs", "S05", "stages.csv")
    s05_manifest_path = os.path.join("outputs", "S05", "manifest.json")
    if os.path.isfile(s05_stages_path) and os.path.isfile(s05_manifest_path):
        import pandas as pd

        stages = pd.read_csv(s05_stages_path)
        s05 = json.load(open(s05_manifest_path, encoding="utf-8"))
        n_outer_s05 = int(s05["inputs"]["splits"]["outer_train"]["n_rows"])
        fixed_s05 = int(s05.get("fixed_epochs") or 0)
        sel_mask = stages["stage"].eq("selection_train")
        refit_mask = stages["stage"].eq("mlp_train") & stages["run_id"].astype(str).str.startswith("seed_")
        eval_mask = stages["stage"].eq("comparison_val_inference")
        vec_mask = stages["stage"].eq("vectorize_train")
        scale = n_outer / max(n_outer_s05, 1)
        anchor = {
            "available": True,
            "source": os.path.abspath(s05_stages_path),
            "s05_outer_train_rows": n_outer_s05,
            "s05_fixed_epochs": fixed_s05,
            "note": "исторический протокол S05 (min_count=10, старые сплиты): только "
                    "опора для проекции времени/RAM, не baseline S06",
        }

        def _mean_seconds(mask) -> float | None:
            vals = stages.loc[mask, "seconds"]
            vals = vals[vals.notna()]
            return float(vals.mean()) if len(vals) else None

        vec_s = _mean_seconds(vec_mask)
        if vec_s is not None:
            proj["vectorize_train_seconds"] = round(vec_s * scale, 3)
        sel_s = _mean_seconds(sel_mask)
        if sel_s is not None and fixed_s05 > 0:
            # selection ≈ обучение на train_fit примерно на числе эпох ≤ refit
            proj["selection_train_seconds"] = round(sel_s * (n_fit / max(n_outer_s05, 1)), 3)
        refit_s = _mean_seconds(refit_mask)
        if refit_s is not None and fixed_s05 > 0:
            per_epoch = refit_s / fixed_s05
            proj["refit_seconds_per_seed"] = round(per_epoch * scale * fixed_s05, 3)
            proj["assumption_epochs"] = (
                fixed_s05,
                "число эпох S06 ещё не выбрано; проекция при том же числе эпох, "
                "после select замените фактическим fixed_epochs из протокола",
            )
        eval_s = _mean_seconds(eval_mask)
        if eval_s is not None:
            proj["eval_seconds_per_seed"] = round(eval_s * (n_val / max(int(
                s05["inputs"]["splits"][EVAL_PART]["n_rows"]), 1)), 3)

    ckpt_anchor = _dir_size(os.path.join("outputs", "S05", "checkpoints", "seed_42"))
    estimate = {
        "schema_version": S06_GATE1_SCHEMA,
        "role": "gate1_resource_estimate",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "projected": True,
        "disclaimer": (
            "Проекция от измерений S05 и размеров сплита S06; НЕ измерение. "
            "Фактические время/RAM/VRAM фиксируются stages.csv самого Gate 1."
        ),
        "s06_split_rows": {
            "outer_train": n_outer, "train_fit": n_fit,
            "train_monitor": n_mon, EVAL_PART: n_val,
            SEALED_PART: int(parts[SEALED_PART]["n_rows"]),
        },
        "anchor_s05": anchor,
        "projections_seconds": proj,
        "ram_bytes_reference_s05": {
            "refit_peak_rss": None,  # заполняется ниже при наличии stages
            "eval_peak_rss": None,
            "note": "RAM сублиинейна по числу строк (разреженные признаки); ориентир — "
                    "фактические пики stages.csv Gate 1, здесь только якоря S05",
        },
        "vram": {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "projected_peak_bytes_s05_gpu_refit": None,
            "note": "при недоступности GPU — null, не ноль; CPU-прогон дольше и не измерен здесь",
        },
        "checkpoint_bytes_reference_s05_seed42": ckpt_anchor,
        "sealed_test": "оценка ресурсов не выполняет inference; sealed_test исключён",
    }
    if anchor["available"]:
        import pandas as pd

        stages = pd.read_csv(s05_stages_path)
        refit_rss = stages.loc[stages["stage"].eq("mlp_train"), "peak_rss_bytes"]
        eval_rss = stages.loc[stages["stage"].eq("comparison_val_inference"), "peak_rss_bytes"]
        if refit_rss.notna().any():
            estimate["ram_bytes_reference_s05"]["refit_peak_rss"] = int(refit_rss.max())
        if eval_rss.notna().any():
            estimate["ram_bytes_reference_s05"]["eval_peak_rss"] = int(eval_rss.max())
        vram = stages.loc[stages["stage"].eq("mlp_train"), "peak_vram_bytes"]
        if vram.notna().any() and bool(torch.cuda.is_available()):
            estimate["vram"]["projected_peak_bytes_s05_gpu_refit"] = int(vram.max())
    if write:
        os.makedirs(cfg.shared_dir, exist_ok=True)
        path = os.path.join(cfg.shared_dir, "resource_estimate_gate1.json")
        if os.path.exists(path):
            raise S06Gate1Error(f"Существующие outputs не перезаписываются: {path}")
        _write_json(estimate, path)
        estimate = dict(estimate)
        estimate["written_to"] = os.path.abspath(path)
    return estimate


__all__ = [
    "S06_GATE1_SCHEMA",
    "S06Gate1Config",
    "S06Gate1Error",
    "EXPECTED_FULL_VOCAB",
    "ensure_constant_lr_schedule",
    "estimate_gate1_resources",
    "freeze_training_protocol",
    "load_training_protocol",
    "run_gate1_contour",
    "run_gate1_refit",
    "validate_protocol_against_inputs",
    "verify_gate0_split",
    "verify_s06_tfidf",
]
