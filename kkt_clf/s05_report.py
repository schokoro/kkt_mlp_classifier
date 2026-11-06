"""S05 (D09): агрегированный отчёт из сохранённых машинных результатов обучения.

Строит ``report/report.html``, ``report/runs.csv``, ``report/per_class.parquet``,
``report/stages.csv``, ``report/history.csv``, ``report/manifest.json`` **без
обучения и без пересчёта метрик** — исключительно по файлам, записанным
обучающим контуром ``kkt_clf.s05_baseline`` (D04–D08) и аудитом (D02/D03).

Гарантии контракта D09 и исправлений независимого ревью 1 (пункты 3, 5, 6, 7, 10):

- числа берутся из файлов как есть; отсутствующие значения остаются ``null``
  (в HTML — «—»), ничего не дорисовывается нулями — в том числе ресурсы
  stages.csv: группа/сумма без единого измерения показывают «—», не 0;
- validation маркируется строго: в seed/config-агрегаты и сравнение с majority
  входят только прогоны с ``metric_evaluated_on == "comparison_val"``;
  успешные прогоны с метриками ``sealed_test`` отклоняются (S05ReportError до
  любых записей) — test-метрик в отчёте S05 нет;
- запланированные seed из ``manifest.seeds.planned`` без строки в ``runs.csv``
  добавляются как ``skipped``; failed/skipped видны и в таблице, и в gate;
- gate отражает подтверждённость требований: ``not_ready`` при нарушении или
  неподтверждении обязательных требований (нет кривых обучения, baseline не
  лучше majority, ненулевые пересечения сплита, проваленные проверки
  TF-IDF/checkpoint), ``conditional`` при частичных пробелах в доказательствах,
  ``ready`` только когда все проверяемые требования подтверждены файлами;
- кривые ``monitor`` в history.csv — это внутренний ``train_monitor`` (часть
  outer_train, попавшая в обучение), а не отложенная выборка: HTML явно
  предупреждает, что кривые и train↔monitor gap характеризуют оптимизацию, а
  не generalization; независимая оценка — метрики comparison_val в runs.csv;
- HTML пересобирается из шести файлов ``report/``: содержимое опциональных
  внешних источников (majority, checkpoint, TF-IDF verification, selection,
  аудит сплита/разметки) сохраняется в ``manifest.json`` (``embedded_sources``)
  и при повторной сборке из ``report/`` читается из встроенной копии — gate и
  HTML не зависят от исходного ``output_dir``;
- ``manifest.json`` и ``per_class.parquet`` в ``output_dir`` опциональны: при
  их отсутствии (контур прерван до записи) отчёт строится диагностический, а
  gate фиксирует неподтверждённые требования. Обязательны и проверяются на
  читаемость: ``runs.csv``, ``history.csv``, ``stages.csv``; битый обязательный
  файл — ``S05ReportError`` до любых записей, частичного вывода нет;
- файл сплит-аудита: в исходном ``split_audit.csv`` колонка
  ``n_row_ids_unique`` была ошибочно записана постоянной ``1``; используется
  ``split_audit_corrected.csv``, ошибка явно указывается в отчёте.

Исправления независимого ревью 2 (пункты 3, 4, 5, 8):

- gate корректно трактует ``restored_best=False`` как единственно допустимое
  значение для финального refit (True — нарушение D08, отсутствие поля —
  неподтверждённость); булевы поля provenance больше не сводятся к ``all(...)``
  по всем полям, включая ``restored_best``;
- gate требует фактического подтверждения, а не наличия пустых структур:
  непустые проверки provenance checkpoint, кандидаты и выбранные эпохи
  selection, ненулевой аудит сплита, полные кривые train/monitor каждого
  успешного прогона и настоящие кривые ``selection:*`` в history.csv, а также
  идентификатор ``full_tfidf_id`` и в tfidf_verification.json, и в каждом
  успешном прогоне; ``ready`` по пустым структурам невозможен;
- кривые ``selection:*`` (train-only selection, train_fit ↔ train_monitor)
  входят в history.csv: отображаются в HTML и участвуют в train↔monitor gap
  (строки ``phase=selection``); текст HTML больше не утверждает, что эти кривые
  не входят в шесть файлов отчёта;
- файл поддержки классов по сплиту: в исходном ``audit/class_support_split.csv``
  прежняя версия freeze_split теряла ``support_all`` редких классов
  (``kept_after_min_count=False`` → NaN, колонка float); используется
  ``audit/class_support_split_corrected.csv`` при наличии, ошибка явно
  указана в отчёте.

Исправления закрытия gap (промпт 27):

- максимальный train↔monitor gap по loss считается по всем попарно
  сопоставленным эпохам настоящего train-only selection (run_id
  ``selection:*``, train_fit ↔ train_monitor) с указанием run_id/эпохи;
  финальный refit, где monitor входит в outer train, в максимум не входит —
  для refit сохраняется отдельно значение последней сопоставленной эпохи;
  gate использует именно этот максимум selection; эпохи с отсутствующей/NaN
  loss не образуют пар и не дают нулевой gap;
- сохранённые предсказания моделей в output_dir отсутствуют: таблица
  L3-путаниц (confusion) не строится, обучение/inference не запускаются —
  ограничение указано в HTML явно.

Ничего не перезаписывается: если любой целевой файл в ``report_dir`` уже
существует — отказ до любых записей.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from .experiment import (
    _esc,
    _fmt,
    _fmt_mb,
    _read_json,
    _sha256_file,
    _svg_bar_chart,
    _svg_line_chart,
    _table_html,
    _write_json,
)

log = __import__("logging").getLogger(__name__)

S05_REPORT_SCHEMA_VERSION = "s05-report@1"
REPORT_DIRNAME = "report"
REPORT_FILES = ("report.html", "runs.csv", "per_class.parquet",
                "stages.csv", "history.csv", "manifest.json")

COMPARISON_VAL = "comparison_val"
SEALED_TEST = "sealed_test"
METRIC_QUALITY_COLUMNS = ("l3_macro_f1", "l3_accuracy")

# Опциональные внешние источники, содержимое которых сохраняется в manifest.json
# отчёта, чтобы report.html полностью пересобирался из шести файлов report/.
EMBEDDED_JSON_SOURCES = (
    "majority_baseline.json",
    "selected_checkpoint.json",
    "tfidf_verification.json",
    "selection/selection_report.json",
    "audit/data_audit.json",
)
EMBEDDED_CSV_SOURCES = (
    "audit/split_audit_corrected.csv",
    "audit/split_audit.csv",
    "audit/class_support_split_corrected.csv",
    "audit/class_support_split.csv",
    "audit/class_support.csv",
)

SPLIT_AUDIT_ORIGINAL_ERROR = (
    "Исходный audit/split_audit.csv содержал ошибку в колонке n_row_ids_unique: "
    "вместо фактического числа уникальных row_id во всех частях было записано "
    "постоянное значение 1 (остальные колонки корректны). В отчёте используется "
    "исправленный audit/split_audit_corrected.csv; sha256/bytes частей сплита "
    "в обоих файлах совпадают."
)

CLASS_SUPPORT_SPLIT_ORIGINAL_ERROR = (
    "Исходный audit/class_support_split.csv был создан прежней версией "
    "freeze_split: support_all для редких классов (kept_after_min_count=False) "
    "терялся (NaN), колонка записывалась как float. В отчёте используется "
    "audit/class_support_split_corrected.csv; остальные колонки совпадают."
)

SPLIT_AUDIT_OVERLAP_COLUMNS = (
    "char_keys_shared_with_other_parts",
    "word_keys_shared_with_other_parts",
)

RUNS_REQUIRED = [
    "run_id", "config_id", "train_seed", "split_seed", "status", "reason",
    "n_train", "n_monitor", "n_val", "n_classes", "full_tfidf_id",
    "epochs_completed", "best_epoch", "l3_macro_f1", "l3_accuracy",
    "metric_evaluated_on", "train_seconds", "peak_rss_bytes", "checkpoint_path",
]
STAGES_REQUIRED = [
    "run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
    "measurement_note",
]
HISTORY_REQUIRED = [
    "run_id", "epoch", "split", "loss", "l3_macro_f1", "l3_accuracy", "lr",
]
PER_CLASS_REQUIRED = ["run_id", "L3", "support", "precision", "recall", "f1"]


class S05ReportError(RuntimeError):
    """Ошибка построения отчёта S05; файлы не записываются."""


# ---------------------------------------------------------------------------
# Загрузка машинных файлов
# ---------------------------------------------------------------------------


def _required_frame(path: str, columns: list[str], kind: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise S05ReportError(
            f"{kind}: обязательный машинный файл не найден: {path!r}. "
            "Сначала выполните обучающий контур S05 (scripts/run_s05_baseline.py "
            "refit); отчёт не подменяет несуществующие данные."
        )
    try:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 — битая обязательная таблица = чистый отказ
        raise S05ReportError(
            f"{kind}: обязательный файл {path!r} не читается ({type(e).__name__}: {e}); "
            "частичный отчёт не записывается — восстановите файл обучающего контура."
        ) from e
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise S05ReportError(f"{kind}: в {path!r} нет обязательных колонок {missing}")
    return df


def _embed_csv(df: pd.DataFrame) -> dict[str, Any]:
    """DataFrame → JSON-совместимый payload (NaN/NaT → null)."""
    return json.loads(df.to_json(orient="split"))


def _read_embedded_csv(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(payload["data"], columns=payload["columns"])


def _load_optional_json(
    output_dir: str, name: str, embedded: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Опциональный JSON: внешний файл предпочтительнее встроенной копии."""
    path = os.path.join(output_dir, name)
    if os.path.isfile(path):
        try:
            return _read_json(path), "file"
        except Exception as e:  # noqa: BLE001 — битый опциональный файл не роняет отчёт
            log.warning("Не удалось прочитать %s: %s", path, e)
            return None, "unreadable"
    if name in embedded:
        return embedded[name], "embedded"
    return None, "missing"


def _load_optional_csv(
    output_dir: str, name: str, embedded: dict[str, Any],
) -> tuple[pd.DataFrame | None, str]:
    path = os.path.join(output_dir, name)
    if os.path.isfile(path):
        try:
            return pd.read_csv(path), "file"
        except Exception as e:  # noqa: BLE001
            log.warning("Не удалось прочитать %s: %s", path, e)
            return None, "unreadable"
    if name in embedded:
        return _read_embedded_csv(embedded[name]), "embedded"
    return None, "missing"


def _load_split_audit(
    output_dir: str, embedded: dict[str, Any],
) -> tuple[pd.DataFrame | None, str | None]:
    """Исправленный аудит сплита предпочтительнее (файл, затем встроенная
    копия); ошибка исходного файла фиксируется для показа в отчёте."""
    corrected, corrected_origin = _load_optional_csv(
        output_dir, "audit/split_audit_corrected.csv", embedded)
    if corrected is not None:
        label = "split_audit_corrected.csv"
        if corrected_origin == "embedded":
            label += " (встроенная копия из manifest.json отчёта)"
        return corrected, label
    original, original_origin = _load_optional_csv(
        output_dir, "audit/split_audit.csv", embedded)
    if original is not None:
        label = "split_audit.csv (с ошибкой в n_row_ids_unique, см. примечание)"
        if original_origin == "embedded":
            label += "; встроенная копия из manifest.json отчёта"
        return original, label
    return None, None


def _load_class_support_split(
    output_dir: str, embedded: dict[str, Any],
) -> tuple[pd.DataFrame | None, str | None]:
    """Исправленная поддержка классов по сплиту предпочтительнее (файл, затем
    встроенная копия); ошибка исходного файла фиксируется для показа в отчёте."""
    corrected, corrected_origin = _load_optional_csv(
        output_dir, "audit/class_support_split_corrected.csv", embedded)
    if corrected is not None:
        label = "class_support_split_corrected.csv"
        if corrected_origin == "embedded":
            label += " (встроенная копия из manifest.json отчёта)"
        return corrected, label
    original, original_origin = _load_optional_csv(
        output_dir, "audit/class_support_split.csv", embedded)
    if original is not None:
        label = "class_support_split.csv (с ошибкой в support_all редких классов, см. примечание)"
        if original_origin == "embedded":
            label += "; встроенная копия из manifest.json отчёта"
        return original, label
    return None, None


def _load_training_manifest(
    output_dir: str,
) -> tuple[dict[str, Any] | None, str]:
    """Манифест обучающего контура; опционален (диагностический отчёт возможен).

    При пересборке из ``report/`` ``manifest.json`` — это манифест отчёта:
    обучающий контекст разворачивается из его поля ``training_manifest``."""
    path = os.path.join(output_dir, "manifest.json")
    if not os.path.isfile(path):
        return None, "missing"
    try:
        raw = _read_json(path)
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось прочитать %s: %s", path, e)
        return None, "unreadable"
    if not isinstance(raw, dict):
        return None, "unreadable"
    nested = raw.get("training_manifest")
    if isinstance(nested, dict):
        return nested, "report_manifest"
    return raw, "file"


def _check_new(paths: list[str]) -> None:
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise S05ReportError(
            "Существующие файлы отчёта не перезаписываются; отказ до любых записей: "
            f"{existing}. Удалите/перенесите их или укажите другой --report-dir."
        )


# ---------------------------------------------------------------------------
# Агрегаты: seed-статистика, train–monitor gap, majority
# ---------------------------------------------------------------------------


def _mean_std_range(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "range": None, "n": 0}
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
        "range": float(arr.max() - arr.min()), "n": int(len(arr)),
    }


def _with_skipped_runs(runs_df: pd.DataFrame, training_manifest: dict | None) -> pd.DataFrame:
    """Запланированные seed без строки в runs.csv → status=skipped (null-метрики)."""
    planned = ((training_manifest or {}).get("seeds") or {}).get("planned") or []
    present = set(runs_df["run_id"].astype(str))
    rows = []
    split_seed = None
    if training_manifest:
        split_seed = ((training_manifest.get("config") or {}).get("selection_seed"))
    for seed in planned:
        run_id = f"seed_{seed}"
        if run_id not in present:
            rows.append({
                "run_id": run_id, "config_id": None, "train_seed": int(seed),
                "split_seed": split_seed, "status": "skipped",
                "reason": "запланирован в manifest.seeds.planned, но обучающий "
                          "контур не записал результат (прогон не выполнен или "
                          "контур прерван до его записи)",
                "n_train": None, "n_monitor": None, "n_val": None,
                "n_classes": None, "full_tfidf_id": None,
                "epochs_completed": None, "best_epoch": None,
                "l3_macro_f1": None, "l3_accuracy": None,
                "metric_evaluated_on": COMPARISON_VAL,
                "train_seconds": None, "peak_rss_bytes": None, "checkpoint_path": None,
            })
    if not rows:
        return runs_df
    skipped_df = pd.DataFrame(rows).reindex(columns=runs_df.columns)
    for col in runs_df.columns:
        if not skipped_df[col].notna().any():
            if pd.api.types.is_numeric_dtype(runs_df[col].dtype):
                skipped_df[col] = np.nan
            else:
                skipped_df[col] = None
        elif pd.api.types.is_numeric_dtype(runs_df[col].dtype) and \
                pd.api.types.is_numeric_dtype(skipped_df[col].dropna().dtype):
            skipped_df[col] = skipped_df[col].astype(runs_df[col].dtype)
    return pd.concat([runs_df, skipped_df], ignore_index=True)


def _comparison_val_success(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Успешные прогоны с метриками строго на comparison_val."""
    ok = runs_df[runs_df["status"] == "success"]
    return ok[ok["metric_evaluated_on"].astype(str) == COMPARISON_VAL]


def _seed_stats(runs_df: pd.DataFrame) -> list[dict[str, Any]]:
    """mean/std/range внутри групп одинакового config_id (успешные runs,
    метрики строго на comparison_val — без объединения разных выборок)."""
    out = []
    ok = _comparison_val_success(runs_df)
    ok = ok[ok["config_id"].notna()]
    for config_id, grp in ok.groupby(ok["config_id"].astype(str)):
        out.append({
            "config_id": config_id,
            "n_success": int(len(grp)),
            "l3_macro_f1": _mean_std_range(grp["l3_macro_f1"].tolist()),
            "l3_accuracy": _mean_std_range(grp["l3_accuracy"].tolist()),
            "evaluated_on": COMPARISON_VAL,
        })
    return out


def _train_monitor_gap(history_df: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
    """Train↔monitor gap по loss: значение последней эпохи и настоящий максимум.

    «monitor» здесь — внутренний train_monitor (часть outer_train), а не
    отложенная выборка; gap характеризует оптимизацию, не generalization.
    Строки ``phase=selection`` — фаза train-only selection (train_fit ↔
    train_monitor, run_id ``selection:*``) — настоящая независимая проверка:
    для неё считается максимум monitor−train по всем попарно сопоставленным
    эпохам (обе loss присутствуют и не NaN) с указанием эпохи. Финальный
    refit в максимум не входит: там monitor — часть outer train, поэтому для
    refit сохраняется только значение последней сопоставленной эпохи."""
    run_ids = set(runs_df["run_id"].astype(str))
    rows = []
    for run_id, grp in history_df.groupby(history_df["run_id"].astype(str)):
        if run_id not in run_ids and not run_id.startswith("selection:"):
            continue
        phase = "selection" if run_id.startswith("selection:") else "refit"
        last_epoch = grp["epoch"].max()
        final = grp[grp["epoch"] == last_epoch]
        train_loss = final.loc[final["split"] == "train", "loss"]
        mon = final[final["split"] == "monitor"]
        mon_loss = mon["loss"]
        tr = (float(train_loss.iloc[0])
              if len(train_loss) and pd.notna(train_loss.iloc[0]) else None)
        mo = (float(mon_loss.iloc[0])
              if len(mon_loss) and pd.notna(mon_loss.iloc[0]) else None)
        gap = (mo - tr) if (tr is not None and mo is not None) else None
        max_gap = max_epoch = None
        if phase == "selection":
            tr_by_ep = grp.loc[grp["split"] == "train"].set_index("epoch")["loss"]
            mo_by_ep = grp.loc[grp["split"] == "monitor"].set_index("epoch")["loss"]
            paired = pd.concat(
                [tr_by_ep, mo_by_ep], axis=1, keys=("train", "monitor")).dropna()
            if len(paired):
                diffs = paired["monitor"] - paired["train"]
                max_epoch = int(diffs.idxmax())
                max_gap = float(diffs.max())
        rows.append({
            "run_id": run_id,
            "phase": phase,
            "final_train_loss": tr,
            "final_monitor_loss": mo,
            "gap_monitor_minus_train": gap,
            "max_gap_monitor_minus_train": max_gap,
            "max_gap_epoch": max_epoch,
            "final_monitor_l3_macro_f1": (float(mon["l3_macro_f1"].iloc[0])
                                          if len(mon) and pd.notna(mon["l3_macro_f1"].iloc[0]) else None),
            "final_monitor_l3_accuracy": (float(mon["l3_accuracy"].iloc[0])
                                          if len(mon) and pd.notna(mon["l3_accuracy"].iloc[0]) else None),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Честный S06 gate: решение отражает подтверждённость требований файлами
# ---------------------------------------------------------------------------


def _split_audit_integrity(split_audit: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Пересечения нормализованных ключей TF-IDF между частями и учёт row id.

    Возвращает (hard, soft): ненулевые пересечения/дубли row id — нарушение
    целостности протокола; отсутствие проверяемых колонок — неподтверждённость.
    """
    hard: list[str] = []
    soft: list[str] = []
    if not len(split_audit):
        hard.append("Файл аудита сплита пуст (0 строк): фиксация разбиения и "
                    "проверка пересечений (D03) файлами не подтверждены.")
        return hard, soft
    overlap_cols = [c for c in SPLIT_AUDIT_OVERLAP_COLUMNS if c in split_audit.columns]
    if not overlap_cols:
        soft.append(
            "Файл аудита сплита не содержит колонок пересечений "
            f"({list(SPLIT_AUDIT_OVERLAP_COLUMNS)}): отсутствие утечек (D03) "
            "файлами не подтверждено.")
    else:
        for _, r in split_audit.iterrows():
            part = r.get("part", "?")
            for c in overlap_cols:
                v = pd.to_numeric(pd.Series([r[c]]), errors="coerce").iloc[0]
                if pd.notna(v) and float(v) > 0:
                    hard.append(
                        f"Аудит сплита: часть {part!r} имеет ненулевое пересечение "
                        f"нормализованных ключей TF-IDF с другими частями "
                        f"({c}={int(float(v))}) — необъявленная утечка (D03).")
    if "n_row_ids_unique" in split_audit.columns and "n_rows" in split_audit.columns:
        for _, r in split_audit.iterrows():
            part = r.get("part", "?")
            n_rows = pd.to_numeric(pd.Series([r["n_rows"]]), errors="coerce").iloc[0]
            n_unique = pd.to_numeric(pd.Series([r["n_row_ids_unique"]]), errors="coerce").iloc[0]
            if pd.notna(n_rows) and pd.notna(n_unique) and float(n_unique) != float(n_rows):
                hard.append(
                    f"Аудит сплита: в части {part!r} n_row_ids_unique ({int(float(n_unique))}) "
                    f"≠ n_rows ({int(float(n_rows))}) — учёт строк нарушен (D03).")
    return hard, soft


def _decide_gate(
    runs_df: pd.DataFrame,
    history_df: pd.DataFrame,
    training_manifest: dict | None,
    tfidf_verification: dict | None,
    selected: dict | None,
    majority: dict | None,
    gap_df: pd.DataFrame | None,
    split_audit: pd.DataFrame | None,
    selection_report: dict | None,
    per_class_present: bool,
) -> dict[str, Any]:
    """Gate S06: not_ready — обязательное требование нарушено или не
    подтверждено; conditional — требования подтверждены частично, есть пробелы;
    ready — все проверяемые требования подтверждены файлами."""
    hard: list[str] = []
    soft: list[str] = []
    n_planned = len(runs_df)
    n_failed = int((runs_df["status"] == "failed").sum())
    n_skipped = int((runs_df["status"] == "skipped").sum())

    ok_all = runs_df[runs_df["status"] == "success"]
    ok = _comparison_val_success(runs_df)
    n_ok = len(ok)

    def _counts() -> dict[str, Any]:
        return {"decision": decision[0], "reasons": hard + soft,
                "hard_reasons": list(hard), "soft_reasons": list(soft),
                "n_planned": n_planned, "n_success": n_ok,
                "n_failed": n_failed, "n_skipped": n_skipped}

    decision: list[str] = []

    if n_ok == 0:
        hard.append("Нет успешных прогонов с метриками на comparison_val: "
                    "baseline не продемонстрирован.")
        decision.append("not_ready")
        return _counts()

    # --- строгая маркировка validation / отсутствие test-метрик ---
    sealed = ok_all["metric_evaluated_on"].astype(str) == SEALED_TEST
    if bool(sealed.any()):
        hard.append(f"Есть успешные прогоны с metric_evaluated_on={SEALED_TEST!r}: "
                    "test-метрики недопустимы в отчёте S05 (D03/D09).")
    other = ok_all["metric_evaluated_on"].astype(str).isin((COMPARISON_VAL, SEALED_TEST))
    other &= ok_all["metric_evaluated_on"].notna()
    if bool((~other).any()):
        bad = sorted(set(ok_all.loc[~other, "metric_evaluated_on"].astype(str)))
        hard.append(f"Метрики части успешных прогонов размечены иначе чем "
                    f"{COMPARISON_VAL!r}: {bad} — протокол оценки не подтверждён.")

    # --- аудит сплита (D03): пересечения и целостность, не только наличие ---
    if split_audit is None:
        soft.append("Файл аудита сплита (audit/split_audit_corrected.csv) отсутствует: "
                    "фиксация разбиения (D03) не подтверждена файлами.")
    else:
        h, s = _split_audit_integrity(split_audit)
        hard.extend(h)
        soft.extend(s)

    # --- full TF-IDF (D04): проверка должна быть пройдена, а не только
    # существовать; идентификаторы обязаны быть подтверждены с обеих сторон ---
    if tfidf_verification is None:
        soft.append("tfidf_verification.json отсутствует: проверка единого full TF-IDF "
                    "(D04) файлами не подтверждена.")
    elif tfidf_verification.get("decision") != "ok":
        hard.append(f"Проверка full TF-IDF (D04): decision="
                    f"{tfidf_verification.get('decision')!r} ≠ 'ok'.")
    ver_id = (tfidf_verification or {}).get("full_tfidf_id")
    if tfidf_verification is not None and not ver_id:
        hard.append("tfidf_verification.json не содержит full_tfidf_id: идентичность "
                    "единого full TF-IDF (D04) файлами не подтверждена.")
    runs_missing_tfidf_id = ok["full_tfidf_id"].isna()
    if bool(runs_missing_tfidf_id.any()):
        hard.append(f"Успешные прогоны без full_tfidf_id "
                    f"({sorted(ok.loc[runs_missing_tfidf_id, 'run_id'].astype(str))}): "
                    "идентификатор TF-IDF (D04) файлами не подтверждён.")
    run_tfidf_ids = set(ok["full_tfidf_id"].dropna().astype(str))
    if ver_id and run_tfidf_ids and run_tfidf_ids - {str(ver_id)}:
        hard.append("Успешные прогоны ссылаются на full_tfidf_id, отличный от "
                    "проверенного в tfidf_verification.json: TF-IDF не соответствует "
                    "протоколу (D04).")

    # --- содержательность (D07): majority обязан быть записан и превзойден ---
    if majority is None:
        hard.append("Majority-class baseline не записан (majority_baseline.json "
                    "отсутствует): содержательность baseline (D07) не подтверждена.")
    else:
        maj_on = majority.get("evaluated_on")
        if maj_on not in (None, COMPARISON_VAL):
            hard.append(f"Majority-class baseline оценён на {maj_on!r} ≠ "
                        f"{COMPARISON_VAL!r}: сравнение (D07) выполнено не по протоколу.")
        maj_f1 = majority.get("l3_macro_f1")
        if maj_f1 is None:
            hard.append("majority_baseline.json не содержит l3_macro_f1: "
                        "содержательность (D07) не подтверждена.")
        else:
            mean_f1 = float(ok["l3_macro_f1"].astype(float).mean())
            if not mean_f1 > float(maj_f1):
                hard.append(
                    f"Средний macro-F1 успешных seed на comparison_val ({mean_f1:.4f}) "
                    f"не превышает majority-class baseline ({float(maj_f1):.4f}): "
                    "содержательное обучение не показано (D07).")

    # --- кривые обучения (D05/D07): полные кривые train/monitor каждого
    # успешного прогона; одна monitor-строка на все прогоны недостаточна ---
    ok_run_ids = set(ok_all["run_id"].astype(str))
    hist_ok = history_df[history_df["run_id"].astype(str).isin(ok_run_ids)]
    if not len(hist_ok):
        hard.append("history.csv не содержит кривых обучения (train/monitor) "
                    "успешных прогонов: обучение и train–monitor gap не "
                    "продемонстрированы (D05/D07).")
    else:
        for rid in sorted(ok_run_ids):
            grp = hist_ok[hist_ok["run_id"].astype(str) == rid]
            missing_curves = []
            for split_name in ("train", "monitor"):
                split_grp = grp[grp["split"] == split_name]
                if not len(split_grp) or not bool(split_grp["loss"].notna().any()):
                    missing_curves.append(split_name)
            if missing_curves:
                hard.append(f"Прогон {rid!r}: в history.csv отсутствуют непустые "
                            f"кривые {missing_curves}: полная история train/monitor "
                            "и train–monitor gap не подтверждены (D05/D07).")

    # --- несколько seed (D06) ---
    if n_ok < 3:
        soft.append(f"Успешных seed с метриками на comparison_val {n_ok} < 3 (D06): "
                    "разброс недостаточен для S06.")
    if ok["train_seed"].dropna().nunique() < min(3, n_ok):
        soft.append("Число различных train_seed среди успешных прогонов меньше числа "
                    "прогонов: seed не различаются, воспроизводимость по seed не показана.")
    if n_failed or n_skipped:
        soft.append(f"Плановые прогоны неуспешны: failed={n_failed}, skipped={n_skipped} "
                    "— см. runs.csv.")
    if ok["config_id"].astype(str).nunique() > 1:
        soft.append("Успешные прогоны относятся к разным config_id: средние по seed "
                    "необъединимы в один протокол.")

    # --- train-only протокол выбора эпох (D05): пустая структура недостаточна;
    # нужны кандидаты, выбранные эпохи и настоящие кривые selection в history ---
    if selection_report is None:
        soft.append("selection/selection_report.json отсутствует: train-only протокол "
                    "выбора числа эпох (D05) файлами не подтверждён.")
    else:
        candidates = selection_report.get("candidates")
        chosen = selection_report.get("chosen") or {}
        if not candidates or chosen.get("fixed_epochs") is None:
            hard.append("selection_report.json не содержит кандидатов и выбранного "
                        "числа эпох (пустая структура): train-only протокол выбора "
                        "эпох (D05) файлами не подтверждён.")
        else:
            sel_hist = history_df[
                history_df["run_id"].astype(str).str.startswith("selection:")]
            bad_selection_curves: list[str] = []
            for sel_id in sorted(sel_hist["run_id"].astype(str).unique()):
                grp = sel_hist[sel_hist["run_id"].astype(str) == sel_id]
                for split_name in ("train", "monitor"):
                    split_grp = grp[grp["split"] == split_name]
                    if not len(split_grp) or not bool(split_grp["loss"].notna().any()):
                        bad_selection_curves.append(f"{sel_id}:{split_name}")
            if not len(sel_hist) or bad_selection_curves:
                hard.append("Кривые train-only selection (run_id 'selection:*') в "
                            "history.csv отсутствуют или пусты "
                            f"({bad_selection_curves or 'нет строк selection:*'}): "
                            "выбор эпох по train_monitor (D05) файлами не подтверждён.")

    # --- опорный checkpoint (D08): round-trip, provenance и вид checkpoint ---
    if selected is None:
        soft.append("selected_checkpoint.json отсутствует: опорный checkpoint для S06 "
                    "(D08) не зафиксирован.")
    else:
        rt = (selected.get("roundtrip") or {}).get("status")
        if rt != "success":
            hard.append(f"Round-trip проверка опорного checkpoint: status={rt!r} (D08).")
        prov = selected.get("provenance_checks") or {}
        # restored_best — не «проверка пройдена/не пройдена», а фактический
        # признак вида checkpoint: для финального refit корректно только False.
        identity = {k: v for k, v in prov.items()
                    if isinstance(v, bool) and k != "restored_best"}
        if not identity:
            hard.append("selected_checkpoint.json не содержит проверок provenance "
                        "(provenance_checks пуст): контракт checkpoint (D08) файлами "
                        "не подтверждён — roundtrip.status='success' без provenance "
                        "недостаточен.")
        elif not all(identity.values()):
            hard.append(f"Проверки provenance опорного checkpoint не пройдены: "
                        f"{identity} (D08).")
        restored_best = prov.get("restored_best")
        if restored_best is True:
            hard.append("Опорный checkpoint восстанавливал лучшие веса по monitor "
                        "(restored_best=True) — недопустимо для финального refit (D08).")
        elif restored_best is not False:
            hard.append(f"provenance_checks.restored_best={restored_best!r}: не "
                        "подтверждено, что опорный checkpoint — финальный refit без "
                        "выбора по monitor (D08).")

    # --- per-class метрики ---
    if not per_class_present:
        soft.append("per_class.parquet отсутствует: качество по классам (D07) файлами "
                    "не подтверждено (обучающий контур прерван до записи).")

    # --- train↔внутренний-monitor gap: диагностика по настоящему train-only
    # selection (train_fit ↔ train_monitor); refit, где monitor входит в
    # outer train, в максимум не входит ---
    if gap_df is not None and len(gap_df):
        sel = gap_df[gap_df["phase"] == "selection"]
        sel_gaps = pd.to_numeric(sel["max_gap_monitor_minus_train"], errors="coerce")
        if bool(sel_gaps.notna().any()):
            i = sel_gaps.idxmax()
            max_gap = float(sel_gaps.loc[i])
            if max_gap > 1.0:
                soft.append(
                    f"Большой train↔monitor gap по loss в train-only selection "
                    f"(максимум monitor−train по сопоставленным эпохам = {max_gap:.4f} "
                    f"при run_id={str(sel.loc[i, 'run_id'])!r}, эпоха "
                    f"{int(sel.loc[i, 'max_gap_epoch'])}): признаки переобучения; "
                    "характер gap требует объяснения до S06.")

    decision.append("not_ready" if hard else ("conditional" if soft else "ready"))
    return _counts()


# ---------------------------------------------------------------------------
# HTML (автономный, inline SVG через переиспользуемые helpers S03/S04)
# ---------------------------------------------------------------------------


_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 24px;
       color: #1a1a2e; max-width: 1180px; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 32px;
     border-bottom: 2px solid #4a69bd; padding-bottom: 4px; }
table { border-collapse: collapse; margin: 12px 0; font-size: 13px; width: 100%; }
th, td { border: 1px solid #ccd; padding: 4px 8px; text-align: left; }
th { background: #eef1f8; }
.caption, .muted { color: #667; font-size: 12px; }
.tick { font-size: 10px; fill: #556; }
.chart-title { font-size: 13px; font-weight: 600; fill: #1a1a2e; }
.grid { stroke: #dde; stroke-width: 1; }
.axis { stroke: #99a; stroke-width: 1.5; }
.baseline { stroke: #b71540; stroke-dasharray: 5 4; stroke-width: 1.5; }
.baseline-label { fill: #b71540; }
.badge { display: inline-block; padding: 2px 12px; border-radius: 10px;
         font-weight: 600; font-size: 13px; }
.badge.ready { background: #d8f5d0; color: #1e6b2f; }
.badge.conditional { background: #fdf0c8; color: #8a6100; }
.badge.not_ready { background: #f8d7da; color: #8a1c24; }
.note { background: #f4f6fb; border-left: 4px solid #4a69bd; padding: 8px 12px;
        font-size: 13px; margin: 10px 0; }
.warn { background: #fdf0c8; border-left: 4px solid #b71540; padding: 8px 12px;
        font-size: 13px; margin: 10px 0; }
code { background: #f0f0f4; padding: 1px 4px; border-radius: 3px; }
"""


def _fmt_run_metric(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def _html_report(
    output_dir: str,
    runs_df: pd.DataFrame,
    history_df: pd.DataFrame,
    stages_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    training_manifest: dict | None,
    majority: dict | None,
    selected: dict | None,
    tfidf_verification: dict | None,
    selection_report: dict | None,
    split_audit: pd.DataFrame | None,
    split_audit_source: str | None,
    class_support: pd.DataFrame | None,
    class_support_split: pd.DataFrame | None,
    class_support_split_source: str | None,
    data_audit: dict | None,
    seed_stats: list[dict[str, Any]],
    gap_df: pd.DataFrame | None,
    gate: dict[str, Any],
    report_manifest: dict[str, Any],
) -> str:
    p: list[str] = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
                    f"<title>S05 baseline report ({_esc(S05_REPORT_SCHEMA_VERSION)})</title>",
                    f"<style>{_CSS}</style></head><body>"]
    p.append(f"<h1>S05 — аудит ODS и воспроизводимый baseline: агрегированный отчёт</h1>")
    p.append(f"<p class='muted'>Схема {_esc(S05_REPORT_SCHEMA_VERSION)} · сформирован "
             f"{_esc(report_manifest['generated_at'])} из машинных файлов "
             f"<code>{_esc(os.path.abspath(output_dir))}</code> без обучения. "
             "Sealed test не использовался; test-метрик в отчёте нет.</p>")
    p.append(f"<p>Вердикт S06 gate: <span class='badge {_esc(gate['decision'])}'>"
             f"{_esc(gate['decision'])}</span> "
             f"(плановых прогонов {gate['n_planned']}: success {gate['n_success']}, "
             f"failed {gate['n_failed']}, skipped {gate['n_skipped']})</p>")
    p.append("<div class='note'>Шкала gate: <b>not_ready</b> — обязательное требование "
             "нарушено или не подтверждено файлами; <b>conditional</b> — протокол "
             "подтверждён частично, есть пробелы в доказательствах; <b>ready</b> — все "
             "проверяемые требования подтверждены файлами (без заранее назначенного "
             "порога F1).</div>")

    # --- протокол и границы ---
    p.append("<h2>Протокол и границы частей</h2>")
    proto = (training_manifest or {}).get("protocol") or {}
    if proto:
        p.append(_table_html(
            pd.DataFrame([{"раздел": k, "описание": v} for k, v in proto.items()]),
            ["раздел", "описание"]))
    p.append("<div class='note'>Метрики качества вычислены на <b>comparison_val</b> "
             "(raw top-1 L3 без отказа), однократно, после завершения всех refit. "
             "Метки comparison_val/sealed_test не использовались для выбора checkpoint "
             "и числа эпох. Режим full TF-IDF — <b>pretrained</b>, трансдуктивен: тексты "
             "корпуса могут совпадать с train/val/test, метки не переносятся.</div>")

    # --- аудит сплита ---
    p.append("<h2>Аудит разбиения (D03)</h2>")
    p.append(f"<div class='warn'>{_esc(SPLIT_AUDIT_ORIGINAL_ERROR)} "
             f"Источник таблицы: <code>{_esc(split_audit_source or 'файл не найден')}</code>.</div>")
    if split_audit is not None:
        p.append(_table_html(split_audit, list(split_audit.columns),
                             caption="Части сплита: строки, доли, группы, поддержка L3, "
                                     "пересечения ключей TF-IDF между частями, fingerprints."))
    else:
        p.append("<p class='muted'>Файл аудита сплита не найден.</p>")

    # --- аудит данных ---
    if data_audit is not None:
        p.append("<h2>Аудит входной разметки (D02)</h2>")
        audit = data_audit.get("audit") or {}
        summary_rows = [{"показатель": k, "значение": jsonable(v)}
                        for k, v in list(audit.items())[:40]
                        if not isinstance(v, (dict, list))]
        if summary_rows:
            p.append(_table_html(pd.DataFrame(summary_rows), ["показатель", "значение"],
                                 caption=f"Источник: audit/data_audit.json "
                                         f"(сформирован {data_audit.get('generated_at', '—')}). "
                                         "Полные детали — в файле."))

    # --- runs ---
    p.append("<h2>Прогоны обучения (runs.csv)</h2>")
    p.append(_table_html(runs_df, RUNS_REQUIRED,
                         formats={c: _fmt_run_metric for c in RUNS_REQUIRED},
                         caption="Одна строка на плановый прогон, включая failed/skipped; "
                                 "'—' = null (не измерено/не записано), не ноль. "
                                 "Метрики учитываются в агрегатах только при "
                                 "metric_evaluated_on='comparison_val'."))

    # --- seed stats ---
    p.append("<h2>Разброс по seed (mean/std/range на comparison_val)</h2>")
    if seed_stats:
        rows = []
        for st in seed_stats:
            for metric in ("l3_macro_f1", "l3_accuracy"):
                s = st[metric]
                rows.append({"config_id": st["config_id"], "metric": metric,
                             "n": st["n_success"], "mean": s["mean"], "std": s["std"],
                             "min": s["min"], "max": s["max"], "range": s["range"]})
        p.append(_table_html(pd.DataFrame(rows),
                             ["config_id", "metric", "n", "mean", "std", "min", "max", "range"],
                             formats={c: _fmt_run_metric for c in
                                      ("mean", "std", "min", "max", "range")},
                             caption="Средние считаются только внутри одинакового config_id "
                                     "и только по прогонам с метриками на comparison_val "
                                     "(без объединения разных конфигураций и выборок)."))
    else:
        p.append("<p class='muted'>Нет успешных прогонов с метриками на comparison_val — "
                 "статистика по seed недоступна.</p>")

    # --- кривые обучения ---
    p.append("<h2>Кривые обучения (history.csv)</h2>")
    p.append("<div class='warn'>Кривые ниже — финальный refit: «monitor» — это "
             "<b>train_monitor</b>, внутренняя часть <b>outer_train</b>, попавшая в "
             "обучение (а также в train_fit при оптимизации). Это <b>не отложенная "
             "выборка</b>: кривые и gap характеризуют оптимизацию, а не generalization. "
             "Независимая оценка качества — метрики comparison_val в таблице прогонов. "
             "Кривые настоящего train-only selection (run_id «selection:*», train_fit "
             "↔ train_monitor) входят в history.csv и показаны отдельным блоком ниже.</div>")
    ok_ids = runs_df.loc[runs_df["status"] == "success", "run_id"].tolist()
    palette = ["#4a69bd", "#b71540", "#0a8754", "#e58e26", "#6c3483", "#117a8b"]
    history_rids = history_df["run_id"].astype(str)
    if ok_ids and len(history_df):
        epochs = sorted(history_df["epoch"].unique().tolist())
        mon_f1 = []
        for i, rid in enumerate(ok_ids):
            grp = history_df[(history_rids == str(rid)) & (history_df["split"] == "monitor")]
            mon_f1.append((rid, palette[i % len(palette)],
                           {int(r.epoch): (None if pd.isna(r.l3_macro_f1) else float(r.l3_macro_f1))
                            for r in grp.itertuples()}))
        p.append(_svg_line_chart("Monitor L3 macro-F1 по эпохам (train_monitor, внутренний)",
                                 epochs, mon_f1))
        train_loss, mon_loss = [], []
        for i, rid in enumerate(ok_ids):
            grp = history_df[history_rids == str(rid)]
            tr = grp[grp["split"] == "train"]
            mo = grp[grp["split"] == "monitor"]
            train_loss.append((f"{rid} train", palette[i % len(palette)],
                               {int(r.epoch): float(r.loss) for r in tr.itertuples()
                                if pd.notna(r.loss)}))
            mon_loss.append((f"{rid} monitor", palette[i % len(palette)],
                             {int(r.epoch): (None if pd.isna(r.loss) else float(r.loss))
                              for r in mo.itertuples()}))
        p.append(_svg_line_chart("Loss по эпохам: train и внутренний monitor", epochs,
                                 train_loss + mon_loss, y_fmt="{:.4f}"))
    else:
        p.append("<p class='muted'>Нет успешных прогонов или history.csv пуст — кривые "
                 "недоступны, обучение не продемонстрировано.</p>")
    sel_ids = sorted(r for r in history_rids.unique() if r.startswith("selection:"))
    if sel_ids:
        sel_epochs = sorted(history_df.loc[history_rids.isin(sel_ids), "epoch"].unique().tolist())
        sel_f1, sel_loss = [], []
        for i, rid in enumerate(sel_ids):
            grp = history_df[history_rids == rid]
            tr = grp[grp["split"] == "train"]
            mo = grp[grp["split"] == "monitor"]
            sel_f1.append((rid, palette[i % len(palette)],
                           {int(r.epoch): (None if pd.isna(r.l3_macro_f1) else float(r.l3_macro_f1))
                            for r in mo.itertuples()}))
            sel_loss.append((f"{rid} train_fit", palette[i % len(palette)],
                             {int(r.epoch): float(r.loss) for r in tr.itertuples()
                              if pd.notna(r.loss)}))
            sel_loss.append((f"{rid} monitor", palette[i % len(palette)],
                             {int(r.epoch): (None if pd.isna(r.loss) else float(r.loss))
                              for r in mo.itertuples()}))
        p.append(_svg_line_chart("Selection: monitor L3 macro-F1 по эпохам "
                                 "(train_monitor, train-only)", sel_epochs, sel_f1))
        p.append(_svg_line_chart("Selection: loss по эпохам (train_fit ↔ train_monitor)",
                                 sel_epochs, sel_loss, y_fmt="{:.4f}"))
    else:
        p.append("<p class='muted'>Кривые selection (run_id «selection:*») в history.csv "
                 "не найдены.</p>")

    # --- train–monitor gap ---
    p.append("<h2>Train ↔ внутренний monitor gap</h2>")
    p.append("<div class='note'>Максимальный gap (monitor−train по loss) считается "
             "по всем попарно сопоставленным эпохам настоящего train-only selection "
             "(run_id «selection:*», train_fit ↔ train_monitor — независимые части "
             "outer train) с указанием эпохи максимума. Финальный refit в максимум "
             "не входит: там monitor — часть outer train, поэтому для refit приведён "
             "только gap последней сопоставленной эпохи. Это диагностика оптимизации, "
             "не оценка обобщения на отложенной выборке.</div>")
    if gap_df is not None and len(gap_df):
        p.append(_table_html(gap_df, list(gap_df.columns),
                             formats={c: _fmt_run_metric for c in gap_df.columns
                                      if c not in ("run_id", "phase")},
                             caption="По каждому прогону: loss train vs внутренний monitor "
                                     "на последней сопоставленной эпохе (gap_monitor_minus_train, "
                                     "«—» = пара отсутствует или NaN, не ноль) и, для phase=selection, "
                                     "максимум по эпохам с эпохой максимума; финальные метрики monitor. "
                                     "Строки phase=selection — фаза train-only selection "
                                     "(train_fit ↔ train_monitor), остальные — финальный refit."))
    else:
        p.append("<p class='muted'>Нет данных history.csv.</p>")

    # --- majority ---
    p.append("<h2>Сравнение с тривиальным majority-class baseline (D07)</h2>")
    if majority is not None:
        maj_rows = [{
            "метод": "majority (outer_train)", "run_id": majority.get("run_id"),
            "majority_label": majority.get("majority_label"),
            "l3_accuracy": majority.get("l3_accuracy"),
            "l3_macro_f1": majority.get("l3_macro_f1"),
            "evaluated_on": majority.get("evaluated_on"),
        }]
        for _, r in _comparison_val_success(runs_df).iterrows():
            maj_rows.append({"метод": "MLP seed", "run_id": r["run_id"],
                             "majority_label": None, "l3_accuracy": r["l3_accuracy"],
                             "l3_macro_f1": r["l3_macro_f1"],
                             "evaluated_on": r["metric_evaluated_on"]})
        maj_df = pd.DataFrame(maj_rows)
        p.append(_table_html(maj_df, list(maj_df.columns),
                             formats={c: _fmt_run_metric for c in ("l3_accuracy", "l3_macro_f1")},
                             caption="Качество моделей сопоставляется с majority на том же "
                                     "comparison_val, а не с историческими ods_light метриками."))
        groups = list(range(len(maj_rows)))
        series = [("l3_macro_f1", "#4a69bd", {i: (None if pd.isna(r["l3_macro_f1"]) else float(r["l3_macro_f1"])) for i, r in enumerate(maj_rows)}),
                  ("l3_accuracy", "#0a8754", {i: (None if pd.isna(r["l3_accuracy"]) else float(r["l3_accuracy"])) for i, r in enumerate(maj_rows)})]
        p.append(_svg_bar_chart("Majority vs seed: macro-F1 и accuracy (comparison_val)",
                                groups, series, y_fmt="{:.3f}"))
    else:
        p.append("<p class='muted'>majority_baseline.json не найден — сравнение недоступно, "
                 "содержательность не подтверждена.</p>")

    # --- per-class ---
    p.append("<h2>Per-class качество и поддержка L3 (comparison_val)</h2>")
    p.append("<div class='note'>Ограничение: сохранённые предсказания моделей по "
             "comparison_val в output_dir не найдены, поэтому таблица самых частых "
             "L3-путаниц (confusion) и матрица ошибок по предсказаниям не построены. "
             "Для их получения потребовался бы дополнительный inference; по постановке "
             "закрытия этого пробела обучение и inference не запускались — пробел "
             "зафиксирован явно, метрики не выдумываются. Доступная диагностика ошибок "
             "по категориям ниже — агрегат per_class.parquet (error_rate = 1 − recall).</div>")
    pc_ok = per_class_df[per_class_df["run_id"].isin(ok_ids)]
    if len(pc_ok):
        agg = (pc_ok.groupby("L3", as_index=False)
               .agg(support=("support", "max"),
                    precision=("precision", "mean"),
                    recall=("recall", "mean"),
                    f1=("f1", "mean")))
        agg["error_rate"] = 1.0 - agg["recall"]
        agg = agg.sort_values("support", ascending=False)
        p.append(_table_html(agg.head(15), ["L3", "support", "precision", "recall", "f1", "error_rate"],
                             formats={c: _fmt_run_metric for c in
                                      ("precision", "recall", "f1", "error_rate")},
                             caption="Топ-15 классов по support; агрегат по успешным seed "
                                     "(mean). Классы с support=0 имеют null-метрики (—), "
                                     "не нули."))
        tail = agg[agg["support"] <= max(2, int(agg["support"].quantile(0.1)))]
        p.append(f"<p class='muted'>Длинный хвост: {len(tail)} из {len(agg)} классов имеют "
                 f"support ≤ {max(2, int(agg['support'].quantile(0.1)))} на comparison_val.</p>")
        worst = agg.sort_values("f1", na_position="last").head(15)
        p.append(_table_html(worst, ["L3", "support", "precision", "recall", "f1", "error_rate"],
                             formats={c: _fmt_run_metric for c in
                                      ("precision", "recall", "f1", "error_rate")},
                             caption="Классы с наихудшим mean F1 (диаграмма ошибок по "
                                     "категориям: error_rate = 1 − recall)."))
        p.append(_svg_bar_chart("Error rate (1 − recall) по категориям: 15 худших",
                                list(range(len(worst))),
                                [("error_rate", "#b71540",
                                  {i: float(r.error_rate) for i, r in enumerate(worst.itertuples())
                                   if pd.notna(r.error_rate)})],
                                y_fmt="{:.2f}"))
    else:
        p.append("<p class='muted'>Нет per-class данных по успешным прогонам "
                 "(per_class.parquet отсутствует или пуст).</p>")
    if class_support_split is not None:
        sup = class_support_split
        kept = sup[sup["kept_after_min_count"].astype(str) == "True"] if "kept_after_min_count" in sup else sup
        p.append(f"<div class='warn'>{_esc(CLASS_SUPPORT_SPLIT_ORIGINAL_ERROR)} "
                 f"Источник таблицы: <code>{_esc(class_support_split_source or 'файл не найден')}</code>.</div>")
        p.append(f"<p class='muted'>Классов после min_count: {len(kept)}; исключено редких L3: "
                 f"{len(sup) - len(kept)}. Детали — <code>{_esc(class_support_split_source or 'audit/class_support_split.csv')}</code> "
                 f"({len(sup)} строк).</p>")
    elif class_support is not None:
        p.append(f"<p class='muted'>Классов в аудите разметки: {len(class_support)} "
                 "(audit/class_support.csv).</p>")

    # --- ресурсы ---
    p.append("<h2>Ресурсные затраты (stages.csv)</h2>")
    if len(stages_df):
        st = stages_df.copy()
        st["seconds"] = pd.to_numeric(st["seconds"], errors="coerce")
        st["peak_rss_bytes"] = pd.to_numeric(st["peak_rss_bytes"], errors="coerce")
        st["peak_vram_bytes"] = pd.to_numeric(st["peak_vram_bytes"], errors="coerce")
        # min_count=1: группа без единого измерения остаётся null (—), не нулём.
        per_run = (st.groupby(["run_id", "stage", "status"], as_index=False)
                   .agg(seconds=("seconds", lambda s: s.sum(min_count=1)),
                        peak_rss_bytes=("peak_rss_bytes", "max"),
                        peak_vram_bytes=("peak_vram_bytes", "max")))
        p.append(_table_html(per_run, list(per_run.columns),
                             formats={"seconds": _fmt_run_metric,
                                      "peak_rss_bytes": _fmt_mb,
                                      "peak_vram_bytes": _fmt_mb},
                             caption="RSS/VRAM в МиБ; '—' = не измерено (null), а не ноль. "
                                     "run_id=shared_vectorization — разделяемые этапы."))
        tot = st.loc[st["status"] == "success", "seconds"].sum(min_count=1)
        p.append(f"<p class='muted'>Суммарное измеренное время успешных этапов: "
                 f"{_fmt_run_metric(float(tot)) if pd.notna(tot) else '—'} с. GPU: "
                 f"{_esc(str(((training_manifest or {}).get('environment') or {}).get('device', '—')))}; "
                 "недоступность CUDA — 'unavailable', не пройденная проверка.</p>")
    else:
        p.append("<p class='muted'>stages.csv пуст.</p>")

    # --- selection / tfidf / checkpoint ---
    p.append("<h2>Selection (D05), full TF-IDF (D04), checkpoint (D08)</h2>")
    if selection_report is not None:
        ch = selection_report.get("chosen") or {}
        p.append("<div class='note'>Selection (train-only): выбран кандидат "
                 f"<code>{_esc(str(ch.get('name')))}</code>, фиксированные эпохи "
                 f"{_esc(str(ch.get('fixed_epochs')))}; критерий: "
                 f"{_esc(str(ch.get('criterion')))}. Кандидатов: "
                 f"{len(selection_report.get('candidates') or [])}.</div>")
    else:
        p.append("<div class='note'>selection/selection_report.json не найден — "
                 "протокол selection файлами не подтверждён.</div>")
    if tfidf_verification is not None:
        p.append("<div class='note'>Full TF-IDF (D04): decision="
                 f"<code>{_esc(str(tfidf_verification.get('decision')))}</code>, "
                 f"artifact_version={_esc(str(tfidf_verification.get('artifact_version')))}, "
                 f"full_tfidf_id={_esc(str((tfidf_verification.get('full_tfidf_id') or '')[:16]))}…, "
                 f"corpus_matches_source={_esc(str(tfidf_verification.get('corpus_matches_source')))}.</div>")
    if selected is not None:
        rt = selected.get("roundtrip") or {}
        rule = selected.get("selection_rule") or {}
        p.append("<div class='note'>Опорный checkpoint для S06: "
                 f"<code>{_esc(str(selected.get('checkpoint_dir')))}</code>; правило "
                 f"{_esc(str(rule.get('kind')))}={_esc(str(rule.get('value')))} "
                 "(train-only, не по comparison_val); round-trip: "
                 f"<b>{_esc(str(rt.get('status')))}</b>.</div>")

    # --- gate ---
    p.append("<h2>Решение о готовности к S06 (D)</h2>")
    if gate["hard_reasons"]:
        p.append("<p><b>Блокеры (not_ready):</b></p><ul>" +
                 "".join(f"<li>{_esc(r)}</li>" for r in gate["hard_reasons"]) + "</ul>")
    if gate["soft_reasons"]:
        p.append("<p><b>Пробелы в подтверждении (conditional):</b></p><ul>" +
                 "".join(f"<li>{_esc(r)}</li>" for r in gate["soft_reasons"]) + "</ul>")
    if not gate["reasons"]:
        p.append("<div class='note'>Все проверки gate подтверждены файлами: сплиты и "
                 "TF-IDF зафиксированы, кривые обучения и selection сохранены, ≥3 "
                 "успешных seed с разными train_seed, baseline содержательно лучше "
                 "majority, checkpoint проверен round-trip и provenance.</div>")
    p.append("<div class='note'>Ограничения: ODS/PoC — результаты диагностические, не оценка "
             "промышленной применимости; трансдуктивность внешнего full TF-IDF отмечена; "
             "sealed test не вскрывался, test-метрик нет.</div>")
    p.append("</body></html>")
    return "".join(p)


def jsonable(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False, default=str)[:400]
        except Exception:  # noqa: BLE001
            return str(v)[:400]
    return v


# ---------------------------------------------------------------------------
# Публичный вход
# ---------------------------------------------------------------------------


def build_s05_report(output_dir: str = "outputs/S05",
                     report_dir: str | None = None) -> dict[str, str]:
    """Формирует агрегированный отчёт D09 из машинных результатов S05.

    Читает обязательные ``runs.csv``/``history.csv``/``stages.csv`` и
    опциональные ``manifest.json``, ``per_class.parquet`` и сопутствующие файлы
    из ``output_dir``; пишет шесть файлов отчёта в ``report_dir`` (по умолчанию
    ``<output_dir>/report``). Содержимое опциональных внешних источников
    сохраняется в ``manifest.json`` отчёта, поэтому повторная сборка из шести
    файлов ``report/`` даёт тот же HTML и gate без обращения к ``output_dir``.
    Существующие файлы не изменяются; при любом пересечении — отказ до записей.
    """
    report_dir = report_dir or os.path.join(output_dir, REPORT_DIRNAME)
    targets = [os.path.join(report_dir, name) for name in REPORT_FILES]
    _check_new(targets)

    # При пересборке из report/ manifest.json — манифест отчёта; встроенные
    # копии опциональных источников берутся из него.
    training_manifest, manifest_origin = _load_training_manifest(output_dir)
    embedded: dict[str, Any] = {}
    if manifest_origin == "report_manifest":
        raw = _read_json(os.path.join(output_dir, "manifest.json"))
        if isinstance(raw.get("embedded_sources"), dict):
            embedded = raw["embedded_sources"]
    if training_manifest is None:
        log.warning("manifest.json обучающего контура не прочитан (%s): "
                    "запланированные seed и контекст конфигурации недоступны, "
                    "отчёт строится диагностический.", manifest_origin)

    runs_df = _required_frame(os.path.join(output_dir, "runs.csv"), RUNS_REQUIRED, "runs.csv")
    history_df = _required_frame(os.path.join(output_dir, "history.csv"), HISTORY_REQUIRED, "history.csv")
    stages_df = _required_frame(os.path.join(output_dir, "stages.csv"), STAGES_REQUIRED, "stages.csv")

    # Строгая маркировка validation: test-метрики недопустимы (D03/D09).
    sealed = runs_df["metric_evaluated_on"].astype(str) == SEALED_TEST
    leaked = sealed & runs_df[list(METRIC_QUALITY_COLUMNS)].notna().any(axis=1)
    if bool(leaked.any()):
        raise S05ReportError(
            "runs.csv содержит метрики качества с metric_evaluated_on='sealed_test' "
            f"(run_id: {sorted(runs_df.loc[leaked, 'run_id'].astype(str))}): test-метрики "
            "недопустимы в отчёте S05. Отчёт отклонён, файлы не записаны."
        )

    per_class_path = os.path.join(output_dir, "per_class.parquet")
    per_class_present = os.path.isfile(per_class_path)
    if per_class_present:
        per_class_df = _required_frame(per_class_path, PER_CLASS_REQUIRED, "per_class.parquet")
    else:
        log.warning("per_class.parquet отсутствует в %s: per-class данные недоступны, "
                    "отчёт строится диагностический.", output_dir)
        per_class_df = pd.DataFrame(columns=PER_CLASS_REQUIRED)

    majority, _ = _load_optional_json(output_dir, "majority_baseline.json", embedded)
    selected, _ = _load_optional_json(output_dir, "selected_checkpoint.json", embedded)
    tfidf_verification, _ = _load_optional_json(output_dir, "tfidf_verification.json", embedded)
    selection_report, _ = _load_optional_json(
        output_dir, os.path.join("selection", "selection_report.json"), embedded)
    data_audit, _ = _load_optional_json(
        output_dir, os.path.join("audit", "data_audit.json"), embedded)
    split_audit, split_audit_source = _load_split_audit(output_dir, embedded)
    class_support, _ = _load_optional_csv(
        output_dir, os.path.join("audit", "class_support.csv"), embedded)
    class_support_split, class_support_split_source = _load_class_support_split(
        output_dir, embedded)

    runs_full = _with_skipped_runs(runs_df, training_manifest)
    seed_stats = _seed_stats(runs_full)
    gap_df = _train_monitor_gap(history_df, runs_full)
    gate = _decide_gate(runs_full, history_df, training_manifest, tfidf_verification,
                        selected, majority, gap_df, split_audit, selection_report,
                        per_class_present)

    # Встроенные копии опциональных источников — для полного пересбора отчёта
    # из шести файлов report/ без чтения исходного output_dir.
    embedded_sources: dict[str, Any] = {}
    for name, payload in (
        ("majority_baseline.json", majority),
        ("selected_checkpoint.json", selected),
        ("tfidf_verification.json", tfidf_verification),
        ("selection/selection_report.json", selection_report),
        ("audit/data_audit.json", data_audit),
    ):
        if payload is not None:
            embedded_sources[name] = payload
    if split_audit is not None and split_audit_source and "split_audit_corrected" in split_audit_source:
        embedded_sources["audit/split_audit_corrected.csv"] = _embed_csv(split_audit)
    elif split_audit is not None:
        embedded_sources["audit/split_audit.csv"] = _embed_csv(split_audit)
    if class_support is not None:
        embedded_sources["audit/class_support.csv"] = _embed_csv(class_support)
    if class_support_split is not None and class_support_split_source and \
            "class_support_split_corrected" in class_support_split_source:
        embedded_sources["audit/class_support_split_corrected.csv"] = _embed_csv(class_support_split)
    elif class_support_split is not None:
        embedded_sources["audit/class_support_split.csv"] = _embed_csv(class_support_split)

    report_manifest: dict[str, Any] = {
        "schema_version": S05_REPORT_SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "output_dir": os.path.abspath(output_dir),
        "report_dir": os.path.abspath(report_dir),
        "gate": gate,
        "seed_stats_comparison_val": seed_stats,
        "sources": {},
        "split_audit_source": split_audit_source,
        "split_audit_original_error": SPLIT_AUDIT_ORIGINAL_ERROR,
        "class_support_split_source": class_support_split_source,
        "class_support_split_original_error": CLASS_SUPPORT_SPLIT_ORIGINAL_ERROR,
        "embedded_sources": embedded_sources,
        "notes": [
            "Отчёт построен без обучения из машинных файлов обучающего контура.",
            "Отсутствующие значения сохранены как null, нулями не подменяются.",
            "Sealed test не использовался; test-метрик нет.",
            "Метрики качества учитываются только при metric_evaluated_on='comparison_val'.",
            "monitor в history.csv — внутренний train_monitor (часть outer_train), "
            "кривые и gap не являются оценкой generalization.",
            "Кривые selection:* (train-only selection) входят в history.csv и "
            "участвуют в train↔monitor gap (phase=selection).",
            "Максимальный train↔monitor gap считается по сопоставленным эпохам "
            "настоящего train-only selection (train_fit vs train_monitor) с указанием "
            "эпохи; refit (monitor внутри outer train) в максимум не входит, для refit "
            "сохраняется значение последней эпохи отдельно.",
            "Сохранённые предсказания моделей отсутствуют: L3-путаницы (confusion) в "
            "отчёте нет, обучение/inference для них не запускался — ограничение "
            "указано в HTML явно.",
            "embedded_sources: содержимое опциональных внешних источников; отчёт "
            "полностью пересобирается из шести файлов report/ без output_dir.",
        ],
    }
    for name in ("runs.csv", "history.csv", "stages.csv", "per_class.parquet",
                 "manifest.json", "majority_baseline.json", "selected_checkpoint.json",
                 "tfidf_verification.json"):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            report_manifest["sources"][name] = _sha256_file(path)
    audit_corr = os.path.join(output_dir, "audit", "split_audit_corrected.csv")
    audit_orig = os.path.join(output_dir, "audit", "split_audit.csv")
    if os.path.isfile(audit_corr):
        report_manifest["sources"]["audit/split_audit_corrected.csv"] = _sha256_file(audit_corr)
    if os.path.isfile(audit_orig):
        report_manifest["sources"]["audit/split_audit.csv"] = _sha256_file(audit_orig)
    support_corr = os.path.join(output_dir, "audit", "class_support_split_corrected.csv")
    support_orig = os.path.join(output_dir, "audit", "class_support_split.csv")
    if os.path.isfile(support_corr):
        report_manifest["sources"]["audit/class_support_split_corrected.csv"] = _sha256_file(support_corr)
    if os.path.isfile(support_orig):
        report_manifest["sources"]["audit/class_support_split.csv"] = _sha256_file(support_orig)
    # Полный контекст обучения — как есть из контура (входы, конфиги, протокол,
    # hashes, env, seed, статусы). При пересборке из report/ разворачивается
    # обратно из этого поля.
    report_manifest["training_manifest"] = training_manifest

    doc = _html_report(
        output_dir, runs_full, history_df, stages_df, per_class_df,
        training_manifest, majority, selected, tfidf_verification,
        selection_report, split_audit, split_audit_source,
        class_support, class_support_split, class_support_split_source, data_audit,
        seed_stats, gap_df, gate, report_manifest,
    )

    os.makedirs(report_dir, exist_ok=True)
    runs_full.to_csv(os.path.join(report_dir, "runs.csv"), index=False)
    per_class_df.to_parquet(os.path.join(report_dir, "per_class.parquet"), index=False)
    stages_df.to_csv(os.path.join(report_dir, "stages.csv"), index=False)
    history_df.to_csv(os.path.join(report_dir, "history.csv"), index=False)
    _write_json(report_manifest, os.path.join(report_dir, "manifest.json"))
    with open(os.path.join(report_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    log.info("Отчёт S05 (D09): %s; gate=%s (%d runs: %d success, %d failed, %d skipped)",
             report_dir, gate["decision"], gate["n_planned"], gate["n_success"],
             gate["n_failed"], gate["n_skipped"])
    return {
        "report_dir": report_dir,
        "report_html": os.path.join(report_dir, "report.html"),
        "runs_csv": os.path.join(report_dir, "runs.csv"),
        "per_class_parquet": os.path.join(report_dir, "per_class.parquet"),
        "stages_csv": os.path.join(report_dir, "stages.csv"),
        "history_csv": os.path.join(report_dir, "history.csv"),
        "manifest_json": os.path.join(report_dir, "manifest.json"),
        "gate": gate["decision"],
    }
