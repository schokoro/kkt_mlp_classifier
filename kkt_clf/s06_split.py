"""S06 (D03): group-aware split с жёсткими per-class минимумами.

Политика S06 отличается от S05: после фильтра ``L3 support >= 12`` каждая
сохранённая L3 обязана иметь не менее ``min_rows_per_class`` строк в
``comparison_val``, ``sealed_test`` и ``train_monitor`` и не менее
``min_train_fit`` строк в ``train_fit``. Группа — множество строк с одним
char-ключом реального preprocessing TF-IDF (``strip_whitespace(preprocess(name))``);
равенство word-ключа влечёт равенство char-ключа, поэтому группа целиком
попадает в одну часть и пересечений нормализованных текстов между частями
нет ни в одной ветке.

Квоты выводятся из данных, а не назначаются механически: целевые числа
строк частей для класса — метод наибольших остатков от долей
80/10/10, поднятые до минимумов; ``outer_train`` получает остаток, но не
меньше ``min_rows_per_class + min_train_fit`` (иначе невозможно удовлетворить
монитор и fit одновременно). Назначение неделимых групп — жадное по
дефициту квоты с засеянным tie-break; если из-за неделимости групп минимум
нарушен, выполняется ограниченный локальный ремонт (перенос самой малой
подходящей группы от донора, который после отдачи не падает ниже своего
минимума). Ремонт не доказывает несовместимость, поэтому при его неудаче
выполняется детерминированный точный поиск :func:`search_feasible_assignment`
по всем четырём частям сразу (train_fit/train_monitor/comparison_val/
sealed_test); найденное распределение используется, а при неуспехе диагностика
честно различает доказанную несовместимость (квота невыполнима на уровне
строк либо перебор исчерпан) и «решение не найдено в пределах бюджета поиска».
Минимумы протокола жёстки: ``min_rows_per_class >= 2`` и ``min_train_fit >= 2``
(значение 1 отклоняется до любых вычислений). Если ограничения несовместимы —
:class:`S06SplitError` с полной per-class диагностикой: файлы сплита не
записываются, состав данных не меняется, требования не ослабляются.

Метрики качества не вычисляются; ``sealed_test`` используется только для
count/hash/support-проверок целостности. Существующие файлы не перезаписываются.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import REQUIRED_COLUMNS, file_fingerprint, filter_min_count
from .features import preprocess, strip_whitespace
from .freeze_split import (
    PART_NAMES,
    PARTS_MONITOR,
    PARTS_OUTER,
    ROW_ID_COLUMN,
    SPLIT_AUDIT_COLUMNS,
    FreezeSplitError,
    FrozenSplit,
    add_group_keys,
    check_group_label_consistency,
    integrity_checks,
    make_groups,
    validate_rows,
)
from .labels import LEVELS, LabelMaps

SUPPORT_BIN_EDGES = ((2, 4), (5, 9), (10, 19), (20, None))

# Протокол S06 (D03): минимумы per-L3 жёстко равны 2; значение 1 не допускается.
S06_MIN_ROWS_PER_CLASS = 2
S06_MIN_TRAIN_FIT = 2

# Части совместного точного поиска (fallback жадной эвристики) и его бюджет.
JOINT_PARTS = ("train_fit", "train_monitor", "comparison_val", "sealed_test")
SEARCH_NODE_BUDGET = 200_000
SEARCH_MAX_GROUPS = 64


def support_bin(support: int) -> str:
    """Диапазон support comparison_val по протоколу S06."""
    for lo, hi in SUPPORT_BIN_EDGES:
        if support >= lo and (hi is None or support <= hi):
            return f"{lo}-{hi}" if hi is not None else f">={lo}"
    raise ValueError(f"support {support} вне допустимых диапазонов")


def _largest_remainder(n: int, weights: np.ndarray) -> np.ndarray:
    """Целочисленное разбиение ``n`` по весам с суммой ровно ``n``."""
    raw = n * weights
    base = np.floor(raw).astype(np.int64)
    rest = int(n - base.sum())
    if rest > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:rest]] += 1
    return base


def class_outer_quota(
    s: int,
    fractions: tuple[float, float, float],
    min_rows: int,
    min_train_rows: int,
) -> np.ndarray | None:
    """Целевые числа строк (outer_train, comparison_val, sealed_test) класса.

    Минимумы имеют приоритет над пропорциями; при невозможности оставить
    ``outer_train`` хотя бы ``min_train_rows`` возвращает None (квота
    несовместима с минимумами на уровне строк, группы не рассматривались).
    """
    w = np.asarray(fractions, dtype=float)
    base = _largest_remainder(s, w)
    q_val = max(int(base[1]), min_rows)
    q_test = max(int(base[2]), min_rows)
    if s - q_val - q_test < min_train_rows:
        q_val, q_test = min_rows, min_rows
    q_train = s - q_val - q_test
    if q_train < min_train_rows:
        return None
    return np.array([q_train, q_val, q_test], dtype=np.int64)


def assign_groups(
    sizes: np.ndarray,
    quota: np.ndarray,
    minima: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Жадное назначение неделимых групп по частям + локальный ремонт минимумов.

    Группы упорядочиваются по убыванию размера (равные — засеянной
    перестановкой) и идут в часть с максимальным дефицитом ``квота − строки``;
    равенство разрешается по индексу части. Ремонт: пока есть дефицитная
    часть, переносим в неё самую малую группу донора, которая закрывает
    дефицит и не опускает донора ниже его минимума. Возвращает
    (назначение group→part, число строк частей); нарушенные минимумы —
    по ``rows < minima``.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    parts = len(quota)
    perm = rng.permutation(len(sizes))
    order = np.lexsort((perm, -sizes))
    assign = np.zeros(len(sizes), dtype=np.int64)
    rows = np.zeros(parts, dtype=np.int64)
    for gi in order:
        p = int(np.argmax(quota - rows))
        assign[gi] = p
        rows[p] += int(sizes[gi])

    for _ in range(max(1, len(sizes)) * parts):
        deficient = [p for p in range(parts) if rows[p] < minima[p]]
        if not deficient:
            break
        p = min(deficient)
        need = int(minima[p] - rows[p])
        donors = sorted(
            (d for d in range(parts) if d != p and rows[d] > minima[d]),
            key=lambda d: -(rows[d] - minima[d]),
        )
        moved = False
        for d in donors:
            cand = [
                gi
                for gi in np.where(assign == d)[0]
                if sizes[gi] >= need and rows[d] - sizes[gi] >= minima[d]
            ]
            if not cand:
                continue
            gi = int(min(cand, key=lambda g: sizes[g]))
            assign[gi] = p
            rows[d] -= int(sizes[gi])
            rows[p] += int(sizes[gi])
            moved = True
            break
        if not moved:
            break
    return assign, rows


def search_feasible_assignment(
    sizes: np.ndarray,
    minima: np.ndarray,
    node_budget: int = SEARCH_NODE_BUDGET,
) -> tuple[np.ndarray | None, bool, int]:
    """Точный поиск допустимого назначения неделимых групп по частям.

    Жадная эвристика :func:`assign_groups` не доказывает несовместимость:
    например, для размеров групп (1, 1, 4, 4, 4) и минимумов (2, 2, 2, 2)
    допустимое разбиение (4, 1+1, 4, 4) существует, но локальный ремонт его
    может не найти. Этот перебор с отсечением и дедупликацией состояний
    детерминирован (группы по убыванию размера, части по индексу) и ищет
    любое допустимое назначение, удовлетворяющее ``rows >= minima``.

    Возвращает ``(assignment, proven, nodes)``: assignment — массив
    group→part либо None; ``proven`` — True, если перебор завершён полностью
    и допустимого назначения нет (отсутствие решения доказано); при обрыве
    по бюджету узлов ``proven`` = False и отсутствие решения НЕ доказано;
    ``nodes`` — число посещённых узлов. Классы с числом групп больше
    :data:`SEARCH_MAX_GROUPS` не перебираются (proven=False).
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    minima = np.asarray(minima, dtype=np.int64)
    n = len(sizes)
    parts = len(minima)
    if n > SEARCH_MAX_GROUPS:
        return None, False, 0
    order = np.argsort(-sizes, kind="stable")
    s = sizes[order]
    suffix = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(s[::-1], out=suffix[1:])
    suffix = suffix[::-1]  # suffix[i] = сумма размеров групп i..n-1
    assign_sorted = np.zeros(n, dtype=np.int64)
    rows = np.zeros(parts, dtype=np.int64)
    visited: set[tuple[int, tuple[int, ...]]] = set()
    nodes = 0
    exhausted = True

    def dfs(i: int) -> bool:
        nonlocal nodes, exhausted
        if not exhausted:
            return False
        nodes += 1
        if nodes > node_budget:
            exhausted = False
            return False
        state = (i, tuple(int(r) for r in rows))
        if state in visited:
            return False
        visited.add(state)
        if i == n:
            return bool((rows >= minima).all())
        remaining = int(suffix[i])
        for p in range(parts):
            if int(rows[p]) + remaining < int(minima[p]):
                return False
        g = int(s[i])
        for p in range(parts):
            rows[p] += g
            assign_sorted[i] = p
            if dfs(i + 1):
                rows[p] -= g
                return True
            rows[p] -= g
        return False

    if not dfs(0):
        return None, exhausted and n <= SEARCH_MAX_GROUPS, nodes
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = assign_sorted
    return assignment, False, nodes


def _class_assignment_report(
    l3: str,
    stage: str,
    sizes: np.ndarray,
    quota: np.ndarray | None,
    minima: np.ndarray,
    rows: np.ndarray,
    part_names: list[str],
) -> dict:
    return {
        "stage": stage,
        "L3": l3,
        "support": int(sizes.sum()),
        "n_groups": int(len(sizes)),
        "group_sizes": sorted(int(x) for x in sizes),
        "quota": None if quota is None else [int(x) for x in quota],
        "minima": [int(x) for x in minima],
        "achieved": {part_names[p]: int(rows[p]) for p in range(len(part_names))},
        "failed_minima": [
            part_names[p] for p in range(len(part_names)) if rows[p] < minima[p]
        ],
    }


def s06_split(
    df: pd.DataFrame,
    min_count: int = 12,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    monitor_fraction: float = 0.1,
    seed: int = 42,
    min_rows_per_class: int = 2,
    min_train_fit: int = 2,
) -> FrozenSplit:
    """Замораживает S06-сплит; при несовместимости минимумов и групп — S06SplitError.

    ``df`` — исходный размеченный датафрейм (только REQUIRED_COLUMNS);
    позиционные id строк сохраняются в ``row_id``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FreezeSplitError(f"Нет обязательных колонок: {missing}")
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise FreezeSplitError(f"Доли outer-частей должны быть тремя и суммироваться в 1: {fractions}")
    if not (0.0 < monitor_fraction < 0.5):
        raise FreezeSplitError(f"monitor_fraction должен быть в (0, 0.5): {monitor_fraction}")
    if min_rows_per_class < S06_MIN_ROWS_PER_CLASS or min_train_fit < S06_MIN_TRAIN_FIT:
        raise FreezeSplitError(
            f"Протокол S06 требует минимумов >= {S06_MIN_ROWS_PER_CLASS}: "
            f"min_rows_per_class={min_rows_per_class}, min_train_fit={min_train_fit}; "
            "значения 1 не допускаются (comparison_val/sealed_test/train_monitor обязаны "
            "иметь >= 2 строки каждой сохранённой L3)."
        )

    df = df[list(REQUIRED_COLUMNS)].copy()
    df[ROW_ID_COLUMN] = np.arange(len(df), dtype=np.int64)
    df["path"] = df["L1"] + " / " + df["L2"] + " / " + df["L3"]

    validate_rows(df)
    dfk = add_group_keys(df)
    check_group_label_consistency(dfk)

    kept, rare = filter_min_count(dfk, min_count, by="L3")
    if len(kept) == 0:
        raise FreezeSplitError(f"После min_count={min_count} не осталось строк; отказ.")
    groups = make_groups(kept)

    min_train_outer = min_rows_per_class + min_train_fit
    classes = sorted(groups["L3"].unique())
    seed_children = np.random.SeedSequence(seed).spawn(len(classes))

    group_part: dict[str, str] = {}  # group_id -> part
    diagnostics: list[dict] = []
    infeasible: list[dict] = []
    quota_report: list[dict] = []

    for l3, child in zip(classes, seed_children):
        rng = np.random.default_rng(child)
        gdf = groups.loc[groups["L3"] == l3]
        sizes = gdf["n_rows"].to_numpy(dtype=np.int64)
        gids = gdf["group_id"].to_numpy()

        # --- Внешнее разбиение 80/10/10 с минимумами.
        quota = class_outer_quota(int(sizes.sum()), fractions, min_rows_per_class, min_train_outer)
        if quota is None:
            # Квота невыполнима на уровне строк ещё до групп: это доказанная
            # несовместимость минимумов с support класса.
            report = _class_assignment_report(
                l3, "outer-row-level", sizes, None,
                np.array([min_train_outer, min_rows_per_class, min_rows_per_class]),
                np.zeros(3, dtype=np.int64), list(PARTS_OUTER),
            )
            report["proof"] = "row-level"
            infeasible.append(report)
            continue

        assignment: dict[str, str] = {}  # group_id -> part (все 4 части)
        class_reports: list[dict] = []

        minima_outer = np.array([min_train_outer, min_rows_per_class, min_rows_per_class])
        assign, rows = assign_groups(sizes, quota, minima_outer, rng)
        greedy_ok = not (rows < minima_outer).any()
        if greedy_ok:
            # --- Монитор внутри outer_train: те же группы, квоты из фактических строк.
            tr_idx = np.array([i for i, p in enumerate(assign) if p == 0], dtype=np.int64)
            tr_sizes = sizes[tr_idx]
            tr_gids = gids[tr_idx]
            r = int(tr_sizes.sum())
            m_target = max(min_rows_per_class, int(_largest_remainder(r, np.array([1.0 - monitor_fraction, monitor_fraction]))[1]))
            fit_min = min_train_fit
            if r - m_target < fit_min:
                m_target = min_rows_per_class
            quota_mon = np.array([r - m_target, m_target], dtype=np.int64)
            minima_mon = np.array([fit_min, min_rows_per_class], dtype=np.int64)
            assign_mon, rows_mon = assign_groups(tr_sizes, quota_mon, minima_mon, rng)
            greedy_ok = not (rows_mon < minima_mon).any()
        if greedy_ok:
            class_reports.append(
                _class_assignment_report(l3, "outer", sizes, quota, minima_outer, rows, list(PARTS_OUTER))
            )
            class_reports.append(
                _class_assignment_report(l3, "monitor", tr_sizes, quota_mon, minima_mon, rows_mon, list(PARTS_MONITOR))
            )
            for gid, part in zip(gids, (PARTS_OUTER[int(x)] for x in assign)):
                if part != "outer_train":
                    assignment[gid] = part
            for gid, p in zip(tr_gids, (PARTS_MONITOR[int(x)] for x in assign_mon)):
                assignment[gid] = p
        else:
            # Жадная эвристика + локальный ремонт не нашли решение. Это НЕ
            # доказательство несовместимости: ищем точным перебором по всем
            # четырём частям сразу (train_fit/train_monitor/val/test).
            joint_minima = np.array(
                [min_train_fit, min_rows_per_class, min_rows_per_class, min_rows_per_class]
            )
            found, proven, nodes = search_feasible_assignment(sizes, joint_minima)
            if found is not None:
                achieved = np.zeros(4, dtype=np.int64)
                for gi, gid in enumerate(gids):
                    p = int(found[gi])
                    achieved[p] += int(sizes[gi])
                    assignment[gid] = JOINT_PARTS[p]
                report = _class_assignment_report(
                    l3, "exact-search", sizes, None, joint_minima, achieved, list(JOINT_PARTS)
                )
                report["method"] = "exact-search"
                report["nodes"] = int(nodes)
                class_reports.append(report)
            else:
                report = _class_assignment_report(
                    l3, "outer+monitor", sizes, quota, minima_outer, rows, list(PARTS_OUTER)
                )
                report["greedy_failed_stage"] = "monitor" if not (rows < minima_outer).any() else "outer"
                report["search_nodes"] = int(nodes)
                report["search_max_groups"] = int(SEARCH_MAX_GROUPS)
                report["proof"] = "exhaustive" if proven else "budget-cut"
                infeasible.append(report)
                continue
        group_part.update(assignment)
        quota_report.extend(class_reports)
        diagnostics.extend(class_reports)

    if infeasible:
        all_proven = all(x.get("proof") in ("row-level", "exhaustive") for x in infeasible)
        if all_proven:
            message = (
                f"Отказ S06 split: минимумы несовместимы с неделимыми группами "
                f"(доказано на уровне строк или полным перебором) для "
                f"{len(infeasible)} (класс, стадия): "
                f"{[(x['L3'], x['stage']) for x in infeasible[:10]]}. "
                "Требования не ослаблялись; полная диагностика в атрибуте infeasible."
            )
        else:
            # Жадность и ограниченный поиск решения не нашли, но отсутствие
            # решения не доказано: честно сообщаем о границах проверки, не
            # выдавая её за доказанную несовместимость.
            message = (
                f"Отказ S06 split: допустимое разбиение не найдено (жадная эвристика "
                f"и точный поиск с бюджетом {SEARCH_NODE_BUDGET} узлов) для "
                f"{len(infeasible)} (класс, стадия): "
                f"{[(x['L3'], x['stage']) for x in infeasible[:10]]}. "
                "Отсутствие решения не доказано — требуется ручной анализ протокола; "
                "требования не ослаблялись; полная диагностика в атрибуте infeasible."
            )
        err = S06SplitError(message)
        err.infeasible = infeasible
        raise err

    kept = kept.copy()
    kept["_part"] = kept["_char_key"].map(group_part)
    if kept["_part"].isna().any():
        raise FreezeSplitError("Есть строки без назначенной части; отказ.")

    part_rows = {
        "train_fit": kept.loc[kept["_part"] == "train_fit"],
        "train_monitor": kept.loc[kept["_part"] == "train_monitor"],
        "comparison_val": kept.loc[kept["_part"] == "comparison_val"],
        "sealed_test": kept.loc[kept["_part"] == "sealed_test"],
    }
    part_rows["outer_train"] = pd.concat(
        [part_rows["train_fit"], part_rows["train_monitor"]], ignore_index=True
    )
    parts: dict[str, pd.DataFrame] = {}
    for name in PART_NAMES:
        d = part_rows[name].sort_values(ROW_ID_COLUMN, ignore_index=True)
        parts[name] = d.reset_index(drop=True)

    excluded = df.loc[~df[ROW_ID_COLUMN].isin(kept[ROW_ID_COLUMN])].copy()
    maps = LabelMaps.from_df(kept[list(REQUIRED_COLUMNS)])

    # --- Поддержка классов по частям.
    support = kept[list(REQUIRED_COLUMNS)].groupby(["L1", "L2", "L3"]).size().reset_index(name="support_all")
    for name in PART_NAMES:
        support[f"support_{name}"] = (
            parts[name].groupby(["L1", "L2", "L3"]).size().reindex(
                pd.MultiIndex.from_frame(support[["L1", "L2", "L3"]]),
                fill_value=0,
            ).to_numpy()
        )
    support["support_bin_val"] = [support_bin(int(v)) for v in support["support_comparison_val"]]

    # --- Жёсткие проверки минимумов (сводно по всем классам).
    min_violations = []
    for _, row in support.iterrows():
        checks = {
            "support_comparison_val": min_rows_per_class,
            "support_sealed_test": min_rows_per_class,
            "support_train_monitor": min_rows_per_class,
            "support_train_fit": min_train_fit,
        }
        for col, minimum in checks.items():
            if int(row[col]) < minimum:
                min_violations.append({"L3": row["L3"], "part": col, "support": int(row[col]), "min": minimum})
    if min_violations:
        raise S06SplitError(
            f"Отказ S06 split: нарушены per-class минимумы: {min_violations[:20]}; "
            f"требования не ослаблялись."
        )

    # --- Полнота покрытия: каждая L3 в каждой части (label maps идентичны).
    coverage_fails = []
    for name in PART_NAMES:
        missing_l3 = set(maps.l3) - set(parts[name]["L3"].unique())
        if missing_l3:
            coverage_fails.append({"part": name, "missing_l3": sorted(missing_l3)[:10]})
    if coverage_fails:
        raise S06SplitError(f"Отказ S06 split: неполное покрытие L3: {coverage_fails}")

    # --- Доли частей (фиксируются, но не являются жёстким критерием: минимумы
    # мелких классов законно смещают доли от 80/10/10).
    n_total = len(kept)
    share_report: dict[str, dict] = {}
    for name in PART_NAMES:
        if name in PARTS_OUTER:
            target = fractions[PARTS_OUTER.index(name)]
            actual = len(parts[name]) / n_total
            base = "all_kept_rows"
        else:
            target = (1.0 - monitor_fraction) if name == "train_fit" else monitor_fraction
            actual = len(parts[name]) / len(parts["outer_train"])
            base = "outer_train"
        share_report[name] = {
            "target": target,
            "actual": round(actual, 6),
            "n_rows": len(parts[name]),
            "base": base,
        }

    checks = integrity_checks(parts, excluded, n_rows_read=len(df))

    report = {
        "seed": int(seed),
        "min_count": int(min_count),
        "fractions": list(fractions),
        "monitor_fraction": float(monitor_fraction),
        "min_rows_per_class": int(min_rows_per_class),
        "min_train_fit": int(min_train_fit),
        "n_rows_read": int(len(df)),
        "n_rows_kept": int(n_total),
        "n_rows_excluded_min_count": int(len(excluded)),
        "n_l3_kept": int(kept["L3"].nunique()),
        "n_l3_excluded": int(len(rare)),
        "n_groups": int(len(groups)),
        "shares": share_report,
        "support": support,
        "rare_classes": rare,
        "class_assignment": diagnostics,
        "checks": checks,
    }
    return FrozenSplit(parts=parts, excluded=excluded, maps=maps, report=report)


class S06SplitError(FreezeSplitError):
    """Несовместимость минимумов S06 с неделимыми группами; файлы не записываются."""


def _versions() -> dict:
    import pyarrow
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
    }


def _git_info() -> dict:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10
            ).stdout.strip() or None
        except Exception:
            return None

    return {"branch": _run("branch", "--show-current"), "head": _run("rev-parse", "HEAD")}


def save_s06_split(
    frozen: FrozenSplit,
    input_path: str,
    out_dir: str,
    audit_dir: str,
    min_count: int,
    fractions: tuple[float, float, float],
    monitor_fraction: float,
    seed: int,
    min_rows_per_class: int,
    min_train_fit: int,
) -> dict:
    """Сохраняет части, manifest и отчёты S06; существующие файлы не перезаписывает."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)

    part_files = {name: os.path.join(out_dir, f"{name}.parquet") for name in PART_NAMES}
    manifest_path = os.path.join(out_dir, "manifest.json")
    split_audit_path = os.path.join(audit_dir, "split_audit.csv")
    support_path = os.path.join(audit_dir, "class_support_split.csv")
    assignment_path = os.path.join(audit_dir, "class_assignment_quotas.json")
    targets = list(part_files.values()) + [manifest_path, split_audit_path, support_path, assignment_path]
    existing = [p for p in targets if os.path.exists(p)]
    if existing:
        raise FreezeSplitError(
            f"Отказ S06 split: существующие outputs не перезаписываются: {existing}"
        )

    report = frozen.report
    checks = report["checks"]
    if not checks["ok"]:
        raise FreezeSplitError(f"Отказ S06 split: проверки целостности не пройдены: {checks}")

    fingerprints: dict[str, dict] = {}
    part_meta: dict[str, dict] = {}
    for name, path in part_files.items():
        d = frozen.parts[name][list(REQUIRED_COLUMNS) + [ROW_ID_COLUMN]]
        d.to_parquet(path, index=False)
        fp = file_fingerprint(path)
        fingerprints[name] = fp
        share = report["shares"][name]
        part_meta[name] = {
            "file": fp["path"],
            "sha256": fp["sha256"],
            "bytes": fp["bytes"],
            "n_rows": int(len(d)),
            "n_groups": int(frozen.parts[name]["_char_key"].nunique()),
            "share": share["actual"],
            "share_base": share["base"],
        }

    support = report["support"].copy()
    support.insert(3, "kept_after_min_count", True)
    rare = report["rare_classes"]
    if len(rare):
        rare_rows = rare.rename(columns={"count": "support_all"})[["L1", "L2", "L3", "support_all"]].copy()
        for c in [f"support_{n}" for n in PART_NAMES]:
            rare_rows[c] = 0
        rare_rows["kept_after_min_count"] = False
        rare_rows["support_bin_val"] = "excluded"
        support = pd.concat(
            [
                support,
                rare_rows[
                    ["L1", "L2", "L3", "kept_after_min_count", "support_all"]
                    + [f"support_{n}" for n in PART_NAMES] + ["support_bin_val"]
                ],
            ],
            ignore_index=True,
        )
    support = support[
        ["L1", "L2", "L3", "kept_after_min_count"]
        + [f"support_{n}" for n in PART_NAMES]
        + ["support_all", "support_bin_val"]
    ]
    support = support.sort_values(
        ["kept_after_min_count", "support_comparison_val", "L3"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    support.to_csv(support_path, index=False)

    with open(assignment_path, "w", encoding="utf-8") as f:
        json.dump(report["class_assignment"], f, ensure_ascii=False, indent=2)

    input_fp = file_fingerprint(input_path)
    audit_rows = []
    for name in PART_NAMES:
        d = frozen.parts[name]
        sup = report["support"]
        shared_char = sum(
            v["n_shared_keys"] for k, v in checks["overlaps"].items()
            if k.endswith("char_key") and name in k.split("|")[:2]
        )
        shared_word = sum(
            v["n_shared_keys"] for k, v in checks["overlaps"].items()
            if k.endswith("word_key") and name in k.split("|")[:2]
        )
        audit_rows.append({
            "part": name,
            "n_rows": int(len(d)),
            "share_target": report["shares"][name]["target"],
            "share_actual": report["shares"][name]["actual"],
            "n_groups": int(d["_char_key"].nunique()),
            "n_l3": int(d["L3"].nunique()),
            "l3_min_support": int(sup[f"support_{name}"].min()),
            "l3_max_support": int(sup[f"support_{name}"].max()),
            "char_keys_shared_with_other_parts": int(shared_char),
            "word_keys_shared_with_other_parts": int(shared_word),
            "n_row_ids_unique": int(d[ROW_ID_COLUMN].nunique()),
            "sha256": fingerprints[name]["sha256"],
            "bytes": fingerprints[name]["bytes"],
        })
    audit_rows.append({
        "part": "excluded_by_min_count",
        "n_rows": int(report["n_rows_excluded_min_count"]),
        "share_target": None,
        "share_actual": round(report["n_rows_excluded_min_count"] / report["n_rows_read"], 6),
        "n_groups": None,
        "n_l3": int(report["n_l3_excluded"]),
        "l3_min_support": None,
        "l3_max_support": None,
        "char_keys_shared_with_other_parts": None,
        "word_keys_shared_with_other_parts": None,
        "n_row_ids_unique": int(frozen.excluded[ROW_ID_COLUMN].nunique()),
        "sha256": None,
        "bytes": None,
    })
    pd.DataFrame(audit_rows, columns=SPLIT_AUDIT_COLUMNS).to_csv(split_audit_path, index=False)

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": {
            "module": os.path.abspath(__file__),
            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]) if sys.argv else None,
            "cwd": os.getcwd(),
            "git": _git_info(),
            "versions": _versions(),
        },
        "policy": {
            "min_count": int(min_count),
            "min_count_rule": "L3 kept iff support >= min_count (включительно), до всех разбиений",
            "fractions": list(fractions),
            "monitor_fraction": float(monitor_fraction),
            "split_seed": int(seed),
            "min_rows_per_class": int(min_rows_per_class),
            "min_train_fit": int(min_train_fit),
            "quota_rule": "целевые числа строк класса — наибольшие остатки от долей частей, "
                          "поднятые до минимумов; outer_train получает остаток, но не меньше "
                          "min_rows_per_class + min_train_fit; проценты не назначаются механически",
            "assignment_rule": "неделимые группы (char-ключ) жадно по дефициту квоты, "
                               "засеянный tie-break; локальный ремонт переносом самой малой "
                               "подходящей группы; при несовместимости — отказ без записи файлов",
            "group_key": "strip_whitespace(preprocess(name)) — char-ветка TF-IDF; "
                         "word-ключ (collapse whitespace) равен ⇒ char-ключ равен, "
                         "поэтому группа покрывает обе ветки",
            "row_id": "позиционный индекс строки во входном parquet (0-based, порядок файла)",
            "parts": {
                "outer_train": "обучение (fit) и внутренний выбор эпох/конфигурации только по train_fit/train_monitor",
                "comparison_val": "сравнение методов S06; не для выбора checkpoint/эпох",
                "sealed_test": "запечатан; метрики качества и inference не выполнялись, count/hash/support только",
                "train_fit": "внутренний fit внутри outer_train",
                "train_monitor": "внутренний monitor внутри outer_train (early stopping/выбор эпох)",
            },
        },
        "inputs": {"labeled_parquet": input_fp},
        "label_maps": frozen.maps.to_dict(),
        "n_classes": {lvl: frozen.maps.n_classes(lvl) for lvl in LEVELS},
        "parts": part_meta,
        "excluded_min_count": {
            "n_rows": int(len(frozen.excluded)),
            "n_l3": int(report["n_l3_excluded"]),
            "row_ids": [int(i) for i in frozen.excluded[ROW_ID_COLUMN]],
        },
        "checks": checks,
        "notes": [
            "Метрики качества не вычислялись; sealed_test не использовался ни для чего, кроме count/hash/support.",
            "Исключённые min_count строки перечислены по row_id выше; молчаливого удаления нет.",
            "Пересечения нормализованных ключей между частями проверены для пар "
            "outer×outer и train_fit×train_monitor (обе TF-IDF-ветки), все нули.",
            "Per-class минимумы проверены жёстко: comparison_val/sealed_test/train_monitor >= "
            "min_rows_per_class, train_fit >= min_train_fit; классы без выполнимых минимумов "
            "не исключались молча — при несовместимости сплит не записывался бы.",
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    manifest_fp = file_fingerprint(manifest_path)

    return {
        "out_dir": os.path.abspath(out_dir),
        "audit_dir": os.path.abspath(audit_dir),
        "parts": part_meta,
        "manifest": manifest_fp,
        "split_audit": {"path": os.path.abspath(split_audit_path)},
        "class_support_split": {"path": os.path.abspath(support_path)},
        "class_assignment_quotas": {"path": os.path.abspath(assignment_path)},
        "n_rows_kept": report["n_rows_kept"],
        "n_rows_excluded_min_count": report["n_rows_excluded_min_count"],
        "checks_ok": checks["ok"],
    }
