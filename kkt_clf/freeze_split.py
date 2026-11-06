"""S05 (D03): заморозка воспроизводимого group-aware сплита размеченной выборки.

Строит детерминированный стратифицированный group-aware сплит 80/10/10
(``outer train`` / ``comparison validation`` / ``sealed test``) и
train-internal monitor (``train_fit`` / ``train_monitor``) внутри outer train.

Группа — множество строк с одинаковым ключом char-ветки реального
preprocessing TF-IDF (``strip_whitespace(preprocess(name))``). Равенство
word-ключа (схлопывание пробелов после ``preprocess``) влечёт равенство
char-ключа, поэтому при группировке по char-ключу совпадают и строки,
неразличимые для word-ветки: пересечений нормализованных текстов между
частями нет ни в одной ветке. Метрики качества не вычисляются; sealed test
используется только для count/hash/support-проверок целостности.

Отказ (файлы не записываются) при:
- null/пустых значениях в обязательных колонках;
- нарушении таксономии (у L3 не один L2, у L2 не один L1);
- конфликтных метках внутри одной группы;
- невозможности стратификации (L3 отсутствует в какой-либо части);
- отклонении фактических долей частей от целевых сверх допуска;
- пересечении нормализованных ключей между частями (защитная проверка).

Исключение редких L3 по ``min_count`` — объявленная политика, а не молчаливое
удаление: состав исключённых строк и классов фиксируется в manifest и
таблицах поддержки. Существующие файлы не перезаписываются.
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

from .data import REQUIRED_COLUMNS, check_taxonomy, file_fingerprint, filter_min_count
from .features import preprocess, strip_whitespace
from .labels import LEVELS, LabelMaps

PARTS_OUTER = ("outer_train", "comparison_val", "sealed_test")
PARTS_MONITOR = ("train_fit", "train_monitor")
PART_NAMES = PARTS_OUTER + PARTS_MONITOR
ROW_ID_COLUMN = "row_id"

# Столбцы отчётов (единый порядок во всех CSV).
SPLIT_AUDIT_COLUMNS = [
    "part", "n_rows", "share_target", "share_actual", "n_groups", "n_l3",
    "l3_min_support", "l3_max_support", "char_keys_shared_with_other_parts",
    "word_keys_shared_with_other_parts", "n_row_ids_unique", "sha256", "bytes",
]
SUPPORT_COLUMNS = [
    "L1", "L2", "L3", "kept_after_min_count", "support_outer_train",
    "support_train_fit", "support_train_monitor", "support_comparison_val",
    "support_sealed_test",
]


class FreezeSplitError(ValueError):
    """Отказ заморозки сплита; файлы не записываются."""


@dataclass
class FrozenSplit:
    parts: dict[str, pd.DataFrame]
    excluded: pd.DataFrame
    maps: LabelMaps
    report: dict = field(default_factory=dict)


def add_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет ключи идентичности текста обеих TF-IDF-веток."""
    pre = preprocess(df["name"].tolist())
    out = df.copy()
    out["_char_key"] = [strip_whitespace(t) for t in pre]
    out["_word_key"] = [" ".join(t.split()) for t in pre]
    return out


def validate_rows(df: pd.DataFrame) -> None:
    """Отказ при null/пустых значениях и нарушениях таксономии."""
    bad = np.zeros(len(df), dtype=bool)
    counts: dict[str, int] = {}
    for c in REQUIRED_COLUMNS:
        col = df[c]
        mask = col.isna().to_numpy()
        notna = ~mask
        if notna.any():
            mask[notna] = col[notna].astype(str).str.strip().eq("").to_numpy()
        counts[c] = int(mask.sum())
        bad |= mask
    if bad.any():
        raise FreezeSplitError(
            f"Отказ freeze split: null/пустые значения в обязательных колонках "
            f"({counts}); состав данных не изменялся и не будет изменён молча."
        )
    try:
        check_taxonomy(df)
    except Exception as e:
        raise FreezeSplitError(f"Отказ freeze split: нарушена таксономия: {e}") from e


def check_group_label_consistency(dfk: pd.DataFrame, n_examples: int = 10) -> None:
    """Отказ, если внутри группы (char-ключ) встречается больше одного пути меток."""
    paths = dfk.groupby("_char_key")["path"].nunique()
    bad = paths[paths > 1]
    if bad.any():
        examples = [
            {"char_key": k, "n_paths": int(n)}
            for k, n in bad.head(n_examples).items()
        ]
        raise FreezeSplitError(
            f"Отказ freeze split: конфликтные метки внутри группы "
            f"(одинаковый текст после preprocessing обеих TF-IDF-веток несёт "
            f"несколько путей L1/L2/L3): {len(bad)} групп; примеры: {examples}. "
            f"Метки не исправлялись."
        )


def make_groups(dfk: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на группу: ключи, размер, единственный путь меток, row ids."""
    groups = (
        dfk.groupby("_char_key", sort=True)
        .agg(
            n_rows=(ROW_ID_COLUMN, "size"),
            L1=("L1", "first"),
            L2=("L2", "first"),
            L3=("L3", "first"),
            n_paths=("path", "nunique"),
            row_ids=(ROW_ID_COLUMN, list),
        )
        .reset_index()
        .rename(columns={"_char_key": "group_id"})
        .sort_values("group_id", ignore_index=True)
    )
    if (groups["n_paths"] > 1).any():
        raise FreezeSplitError("Внутри группы несколько путей меток; см. check_group_label_consistency")
    word_keys = dfk.groupby("_char_key")["_word_key"].nunique()
    groups["_word_key"] = groups["group_id"].map(
        dfk.groupby("_char_key")["_word_key"].first()
    )
    groups["_word_keys_per_group"] = groups["group_id"].map(word_keys)
    return groups


def _largest_remainder(n: int, weights: np.ndarray) -> np.ndarray:
    """Целочисленное разбиение ``n`` по весам с суммой ровно ``n``."""
    raw = n * weights
    base = np.floor(raw).astype(np.int64)
    rest = int(n - base.sum())
    if rest > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:rest]] += 1
    return base


def stratified_group_assignment(
    groups: pd.DataFrame,
    targets: dict[str, float],
    seed_seq: np.random.SeedSequence,
) -> pd.Series:
    """Детерминированное стратифицированное назначение групп по частям.

    Для каждой L3 независимо: целевые числа строк частей получаются методом
    наибольших остатков от долей ``targets``; группы упорядочиваются по
    убыванию размера (порядок равных — засеянной перестановкой из
    ``seed_seq``) и каждая целиком идёт в часть с максимальным дефицитом
    ``целевые строки − уже назначенные строки``; равенство разрешается по
    индексу части. Возвращает Series ``group_id -> part``.
    """
    parts = list(targets)
    w = np.array([targets[p] for p in parts], dtype=float)
    classes = sorted(groups["L3"].unique())
    children = seed_seq.spawn(len(classes))
    assignment: dict[str, str] = {}
    for l3, child in zip(classes, children):
        gdf = groups.loc[groups["L3"] == l3]
        rng = np.random.default_rng(child)
        perm = rng.permutation(len(gdf))
        sizes = gdf["n_rows"].to_numpy()
        order = np.lexsort((perm, -sizes))
        quota = _largest_remainder(int(sizes.sum()), w)
        rows_p = np.zeros(len(parts), dtype=np.int64)
        gids = gdf["group_id"].to_numpy()
        for gi in order:
            deficit = quota - rows_p
            p = int(np.argmax(deficit))
            rows_p[p] += int(sizes[gi])
            assignment[gids[gi]] = parts[p]
    return pd.Series(assignment, name="part")


def coverage_failures(support: pd.DataFrame, part_cols: list[str]) -> list[dict]:
    """Классы, отсутствующие хотя бы в одной из частей."""
    fails = []
    for _, row in support.iterrows():
        missing = [c for c in part_cols if int(row[c]) == 0]
        if missing:
            fails.append({"L1": row["L1"], "L2": row["L2"], "L3": row["L3"], "missing_in": missing})
    return fails


def freeze_split(
    df: pd.DataFrame,
    min_count: int = 10,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    monitor_fraction: float = 0.1,
    seed: int = 42,
    share_tolerance: float = 0.05,
) -> FrozenSplit:
    """Замораживает сплиты; при невозможности надёжного протокола — FreezeSplitError.

    ``df`` — исходный размеченный датафрейм (только REQUIRED_COLUMNS);
    исходные позиционные id сохраняются в колонке ``row_id``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FreezeSplitError(f"Нет обязательных колонок: {missing}")
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise FreezeSplitError(f"Доли outer-частей должны быть тремя и суммироваться в 1: {fractions}")
    if not (0.0 < monitor_fraction < 0.5):
        raise FreezeSplitError(f"monitor_fraction должен быть в (0, 0.5): {monitor_fraction}")

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

    seed_seqs = np.random.SeedSequence(seed).spawn(2)
    targets_outer = dict(zip(PARTS_OUTER, fractions))
    assignment_outer = stratified_group_assignment(groups, targets_outer, seed_seqs[0])
    for gid in groups["group_id"]:
        if gid not in assignment_outer.index:
            raise FreezeSplitError(f"Группа не распределена: {gid!r}")
    part_of = dict(assignment_outer)

    # Train-internal monitor: те же группы, заново назначенные внутри outer train.
    train_groups = groups.loc[groups["group_id"].map(part_of).eq("outer_train")].copy()
    targets_monitor = {
        "train_fit": 1.0 - monitor_fraction,
        "train_monitor": monitor_fraction,
    }
    assignment_monitor = stratified_group_assignment(train_groups, targets_monitor, seed_seqs[1])
    part_of.update(dict(assignment_monitor))

    kept = kept.copy()
    kept["_part"] = kept["_char_key"].map(part_of)
    if kept["_part"].isna().any():
        raise FreezeSplitError("Есть строки без назначенной части; отказ.")

    # outer_train — объединение train_fit и train_monitor (монитор вырезан
    # из train заново, поэтому отдельной метки outer_train после update нет).
    part_rows: dict[str, pd.DataFrame] = {
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
        # Служебные колонки ключей остаются в частях для проверок целостности;
        # в parquet-файлы они не попадают (save_frozen_split отбрасывает).
        parts[name] = d.reset_index(drop=True)

    excluded = df.loc[~df[ROW_ID_COLUMN].isin(kept[ROW_ID_COLUMN])].copy()

    maps = LabelMaps.from_df(kept[list(REQUIRED_COLUMNS)])

    # --- Контроль стратификации: каждая L3 во всех частях.
    kept_labels = kept[list(REQUIRED_COLUMNS)]
    support = (
        kept_labels.groupby(["L1", "L2", "L3"]).size().reset_index(name="support_all")
    )
    for name in PART_NAMES:
        support[f"support_{name}"] = (
            parts[name].groupby(["L1", "L2", "L3"]).size().reindex(
                pd.MultiIndex.from_frame(support[["L1", "L2", "L3"]]),
                fill_value=0,
            ).to_numpy()
        )
    part_cols = [f"support_{n}" for n in PART_NAMES]
    fails = coverage_failures(support, part_cols)
    if fails:
        raise FreezeSplitError(
            f"Отказ freeze split: стратификация невозможна — {len(fails)} L3 отсутствуют "
            f"в частях (group-aware распределение не покрывает все части); примеры: "
            f"{fails[:10]}. Состав данных не менялся."
        )

    # --- Контроль долей частей (actual и target всегда от одной базы).
    n_total = len(kept)
    share_report: dict[str, dict] = {}
    for name in PART_NAMES:
        if name in PARTS_OUTER:
            target = targets_outer[name]
            actual = len(parts[name]) / n_total
            base = "all_kept_rows"
        else:
            target = targets_monitor[name]
            actual = len(parts[name]) / len(parts["outer_train"])
            base = "outer_train"
        share_report[name] = {
            "target": target,
            "actual": round(actual, 6),
            "n_rows": len(parts[name]),
            "base": base,
        }
        if abs(actual - target) > share_tolerance:
            raise FreezeSplitError(
                f"Отказ freeze split: доля части {name}={actual:.4f} (база {base}) "
                f"отклоняется от целевой {target} сверх допуска {share_tolerance}. "
                f"Распределение: {share_report}."
            )

    # --- Проверки пересечений и полноты учёта строк.
    checks = integrity_checks(parts, excluded, n_rows_read=len(df))

    report = {
        "seed": int(seed),
        "min_count": int(min_count),
        "fractions": list(fractions),
        "monitor_fraction": float(monitor_fraction),
        "share_tolerance": float(share_tolerance),
        "n_rows_read": int(len(df)),
        "n_rows_kept": int(n_total),
        "n_rows_excluded_min_count": int(len(excluded)),
        "n_l3_kept": int(kept["L3"].nunique()),
        "n_l3_excluded": int(len(rare)),
        "n_groups": int(len(groups)),
        "shares": share_report,
        "support": support,
        "rare_classes": rare,
        "checks": checks,
    }
    return FrozenSplit(parts=parts, excluded=excluded, maps=maps, report=report)


def integrity_checks(parts: dict[str, pd.DataFrame], excluded: pd.DataFrame, n_rows_read: int) -> dict:
    """Пересечения нормализованных ключей между частями и учёт row ids.

    Ожидается, что множество row id трёх outer-частей в точности равно
    ``range(n_rows_read)`` без исключённых min_count строк, а monitor-части
    в точности разбивают outer_train.
    """
    overlap_pairs = [
        ("outer_train", "comparison_val"),
        ("outer_train", "sealed_test"),
        ("comparison_val", "sealed_test"),
        ("train_fit", "train_monitor"),
    ]
    overlaps = {}
    ok = True
    for a, b in overlap_pairs:
        for key in ("_char_key", "_word_key"):
            ka = set(parts[a][key])
            kb = set(parts[b][key])
            shared = ka & kb
            overlaps[f"{a}|{b}|{key.strip('_')}"] = {
                "n_shared_keys": len(shared),
                "shared_examples": sorted(shared)[:5],
            }
            if shared:
                ok = False
    kept_expected = set(range(n_rows_read)) - set(int(i) for i in excluded[ROW_ID_COLUMN])
    outer_ids = pd.concat([parts[n][ROW_ID_COLUMN] for n in PARTS_OUTER])
    train_ids = pd.concat([parts["train_fit"][ROW_ID_COLUMN], parts["train_monitor"][ROW_ID_COLUMN]])
    accounting = {
        "outer_parts_cover_kept_exactly_once": bool(
            outer_ids.is_unique and set(int(i) for i in outer_ids) == kept_expected
        ),
        "monitor_parts_cover_outer_train_exactly_once": bool(
            train_ids.is_unique
            and set(int(i) for i in train_ids) == set(int(i) for i in parts["outer_train"][ROW_ID_COLUMN])
        ),
        "excluded_disjoint_from_parts": bool(
            set(int(i) for i in excluded[ROW_ID_COLUMN]).isdisjoint(set(int(i) for i in outer_ids))
        ),
    }
    if not all(accounting.values()):
        ok = False
    # Инвариант: равные word-ключи обязаны делить char-ключ (обе ветки покрыты).
    word_char: dict[str, set] = {}
    for name in PART_NAMES:
        for wk, ck in zip(parts[name]["_word_key"], parts[name]["_char_key"]):
            word_char.setdefault(wk, set()).add(ck)
    word_key_spans_multi_char = {k: sorted(v) for k, v in word_char.items() if len(v) > 1}
    if word_key_spans_multi_char:
        ok = False
    return {
        "ok": ok,
        "overlaps": overlaps,
        "row_accounting": accounting,
        "word_key_spans_multiple_char_keys": word_key_spans_multi_char,
    }


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


def save_frozen_split(
    frozen: FrozenSplit,
    input_path: str,
    out_dir: str,
    audit_dir: str,
    min_count: int,
    fractions: tuple[float, float, float],
    monitor_fraction: float,
    seed: int,
    share_tolerance: float,
) -> dict:
    """Сохраняет части, manifest и отчёты; существующие файлы не перезаписывает.

    Возвращает сводку с путями и sha256 записанных файлов.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)

    part_files = {name: os.path.join(out_dir, f"{name}.parquet") for name in PART_NAMES}
    manifest_path = os.path.join(out_dir, "manifest.json")
    split_audit_path = os.path.join(audit_dir, "split_audit.csv")
    support_path = os.path.join(audit_dir, "class_support_split.csv")
    targets = list(part_files.values()) + [manifest_path, split_audit_path, support_path]
    existing = [p for p in targets if os.path.exists(p)]
    if existing:
        raise FreezeSplitError(
            f"Отказ freeze split: существующие outputs не перезаписываются: {existing}"
        )

    report = frozen.report
    checks = report["checks"]
    if not checks["ok"]:
        raise FreezeSplitError(f"Отказ freeze split: проверки целостности не пройдены: {checks}")

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
        support = pd.concat(
            [
                support,
                rare_rows[
                    ["L1", "L2", "L3", "kept_after_min_count", "support_all"]
                    + [f"support_{n}" for n in PART_NAMES]
                ],
            ],
            ignore_index=True,
        )
    support = support[SUPPORT_COLUMNS + (["support_all"] if "support_all" in support.columns else [])]
    support = support.sort_values(
        ["kept_after_min_count", "support_outer_train", "L3"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    support.to_csv(support_path, index=False)

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
    split_audit = pd.DataFrame(audit_rows, columns=SPLIT_AUDIT_COLUMNS)
    split_audit.to_csv(split_audit_path, index=False)

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": {
            "script": os.path.abspath(sys.argv[0] if sys.argv and sys.argv[0] else __file__),
            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]) if sys.argv else None,
            "cwd": os.getcwd(),
            "git": _git_info(),
            "versions": _versions(),
        },
        "policy": {
            "min_count": int(min_count),
            "fractions": list(fractions),
            "monitor_fraction": float(monitor_fraction),
            "split_seed": int(seed),
            "share_tolerance": float(share_tolerance),
            "group_key": "strip_whitespace(preprocess(name)) — char-ветка TF-IDF; "
                        "word-ключ (collapse whitespace) равен ⇒ char-ключ равен, "
                        "поэтому группа покрывает обе ветки",
            "row_id": "позиционный индекс строки во входном parquet (0-based, порядок файла)",
            "parts": {
                "outer_train": "обучение (fit) и внутренний выбор эпох/конфигурации только по train_fit/train_monitor",
                "comparison_val": "сравнение методов S06; не для выбора checkpoint",
                "sealed_test": "запечатан; метрики качества не вычислялись, count/hash/support только",
                "train_fit": "внутренний fit внутри outer train",
                "train_monitor": "внутренний monitor внутри outer train (early stopping/выбор эпох)",
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
        "n_rows_kept": report["n_rows_kept"],
        "n_rows_excluded_min_count": report["n_rows_excluded_min_count"],
        "checks_ok": checks["ok"],
    }
