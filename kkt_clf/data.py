"""Загрузка и проверка данных, сэмпл, отсечение редких классов, сплит (§4.1, §4.3).

Порядок операций фиксирован спекой: null/пустые → таксономия → дубликаты →
сэмпл ``max_rows`` → ``min_count`` → маппинг → сплит. Его собирает
:func:`prepare_data`.
"""
from __future__ import annotations

import hashlib
import logging
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig
from .labels import LEVELS, LabelMaps

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("name", "L1", "L2", "L3")
SPLITS = ("train", "val", "test")


class DataValidationError(ValueError):
    pass


class TaxonomyError(DataValidationError):
    """Нарушение уникальности путей таксономии; ``violations`` — таблица нарушений."""

    def __init__(self, violations: pd.DataFrame):
        self.violations = violations
        lines = [
            f"  {r.level} {r.child!r} → родители {r.parents}" for r in violations.itertuples()
        ]
        super().__init__(
            f"Нарушена уникальность путей таксономии ({len(violations)} шт.):\n" + "\n".join(lines)
        )


class DuplicateWarning(UserWarning):
    pass


@dataclass
class DuplicateReport:
    n_duplicate_rows: int
    n_duplicate_names: int
    n_conflicting_names: int
    examples: pd.DataFrame


@dataclass
class PreparedData:
    splits: dict[str, pd.DataFrame]
    maps: LabelMaps
    report: dict = field(default_factory=dict)
    rare_classes: pd.DataFrame | None = None
    duplicates: DuplicateReport | None = None


def read_parquet(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def file_fingerprint(path: str, chunk: int = 1 << 20) -> dict:
    """sha256 и размер входного файла для метаданных артефакта."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return {"path": os.path.abspath(path), "sha256": h.hexdigest(), "bytes": os.path.getsize(path)}


def check_columns(df: pd.DataFrame, required=REQUIRED_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Нет обязательных колонок: {missing}; есть: {list(df.columns)}")


def drop_empty(df: pd.DataFrame, columns=REQUIRED_COLUMNS) -> tuple[pd.DataFrame, dict[str, int]]:
    """Отбрасывает строки с null/пустым значением в любой из колонок.

    Возвращает очищенный датафрейм и число «плохих» значений по колонкам
    (плюс ``total`` — число удалённых строк).
    """
    bad = np.zeros(len(df), dtype=bool)
    counts: dict[str, int] = {}
    for c in columns:
        col = df[c]
        mask = col.isna().to_numpy()
        notna = ~mask
        if notna.any():
            mask[notna] = col[notna].astype(str).str.strip().eq("").to_numpy()
        counts[c] = int(mask.sum())
        bad |= mask
    counts["total"] = int(bad.sum())
    if counts["total"]:
        log.warning("Удалено строк с null/пустыми значениями: %d (%s)", counts["total"], counts)
    return df.loc[~bad].reset_index(drop=True), counts


def taxonomy_violations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for child, parent in (("L3", "L2"), ("L2", "L1")):
        g = df.groupby(child, sort=True)[parent].unique()
        for name, parents in g[g.map(len) > 1].items():
            rows.append({"level": child, "child": name, "parents": sorted(parents.tolist())})
    return pd.DataFrame(rows, columns=["level", "child", "parents"])


def check_taxonomy(df: pd.DataFrame) -> None:
    """У каждой L3 ровно один L2, у каждого L2 ровно один L1; иначе TaxonomyError."""
    v = taxonomy_violations(df)
    if len(v):
        raise TaxonomyError(v)


def check_duplicates(df: pd.DataFrame, n_examples: int = 10) -> DuplicateReport:
    """Считает повторяющиеся ``name`` (в т. ч. с разными метками). Данные не меняет."""
    dup_mask = df["name"].duplicated(keep=False)
    n_rows = int(df["name"].duplicated(keep="first").sum())
    if n_rows == 0:
        return DuplicateReport(0, 0, 0, pd.DataFrame(columns=["name", "n_rows", "n_label_paths"]))
    d = df.loc[dup_mask, ["name", "L1", "L2", "L3"]]
    path = d["L1"] + " / " + d["L2"] + " / " + d["L3"]
    g = d.assign(path=path).groupby("name").agg(n_rows=("path", "size"), n_label_paths=("path", "nunique"))
    g = g.sort_values(["n_label_paths", "n_rows"], ascending=False)
    rep = DuplicateReport(
        n_duplicate_rows=n_rows,
        n_duplicate_names=int(len(g)),
        n_conflicting_names=int((g["n_label_paths"] > 1).sum()),
        examples=g.head(n_examples).reset_index(),
    )
    warnings.warn(
        f"В данных есть повторяющиеся name: {rep.n_duplicate_rows} лишних строк, "
        f"{rep.n_duplicate_names} уникальных name, из них с разными метками: "
        f"{rep.n_conflicting_names}. Данные не изменены.",
        DuplicateWarning,
        stacklevel=2,
    )
    return rep


def stratified_sample(df: pd.DataFrame, max_rows: int | None, seed: int, by: str = "L3") -> pd.DataFrame:
    """Стратифицированный по ``by`` сэмпл ровно ``max_rows`` строк (метод наибольших остатков)."""
    if max_rows is None or len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    counts = df[by].value_counts().sort_index()
    quota = counts.to_numpy() * (max_rows / len(df))
    take = np.floor(quota).astype(np.int64)
    rest = max_rows - int(take.sum())
    if rest > 0:
        order = np.lexsort((np.arange(len(quota)), -(quota - take)))
        take[order[:rest]] += 1
    take_by_class = pd.Series(take, index=counts.index)
    perm = rng.permutation(len(df))
    shuffled = df.iloc[perm]
    rank = shuffled.groupby(by, sort=False).cumcount().to_numpy()
    keep = rank < take_by_class.reindex(shuffled[by]).to_numpy()
    return shuffled.loc[keep].sort_index().reset_index(drop=True)


def filter_min_count(df: pd.DataFrame, min_count: int, by: str = "L3") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Исключает классы ``by`` с числом примеров < ``min_count``.

    Возвращает отфильтрованные данные и таблицу исключённых классов (L1, L2, L3, count).
    """
    counts = df[by].value_counts()
    rare = counts[counts < min_count]
    rare_df = (
        df.loc[df[by].isin(rare.index), ["L1", "L2", "L3"]]
        .drop_duplicates(by)
        .assign(count=lambda x: x[by].map(rare).astype(int))
        .sort_values(["count", by], ascending=[False, True])
        .reset_index(drop=True)
    )
    return df.loc[~df[by].isin(rare.index)].reset_index(drop=True), rare_df


def split(df: pd.DataFrame, fractions: tuple[float, float, float], seed: int, by: str = "L3") -> dict[str, pd.DataFrame]:
    """Стратифицированный сплит train/val/test двумя вызовами train_test_split."""
    f_train, f_val, f_test = fractions
    idx = np.arange(len(df))
    y = df[by].to_numpy()
    try:
        tr, rest = train_test_split(idx, test_size=f_val + f_test, stratify=y, random_state=seed)
        va, te = train_test_split(
            rest, test_size=f_test / (f_val + f_test), stratify=y[rest], random_state=seed
        )
    except ValueError as e:
        raise DataValidationError(
            f"Стратифицированный сплит невозможен ({e}). Увеличьте data.min_count."
        ) from e
    return {
        "train": df.iloc[np.sort(tr)].reset_index(drop=True),
        "val": df.iloc[np.sort(va)].reset_index(drop=True),
        "test": df.iloc[np.sort(te)].reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str) -> dict[str, str]:
    """Сохраняет сплиты по содержимому (name, L1..L3) в parquet."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for k, d in splits.items():
        p = os.path.join(out_dir, f"{k}.parquet")
        d[list(REQUIRED_COLUMNS)].to_parquet(p, index=False)
        paths[k] = p
    return paths


def prepare_data(cfg: DataConfig, seed: int) -> PreparedData:
    """Загрузка и подготовка данных в порядке §4.1."""
    df = read_parquet(cfg.path)
    check_columns(df)
    df = df[list(REQUIRED_COLUMNS)]
    n_read = len(df)
    df, dropped = drop_empty(df)                                   # 1
    check_taxonomy(df)                                             # 2
    dups = check_duplicates(df, cfg.n_duplicate_examples)          # 3
    n_clean = len(df)
    df = stratified_sample(df, cfg.max_rows, seed)                 # 4
    n_sampled = len(df)
    df, rare = filter_min_count(df, cfg.min_count)                 # 5
    maps = LabelMaps.from_df(df)                                   # 6
    splits = split(df, cfg.split, seed)                            # 7
    report = {
        "rows_read": n_read,
        "dropped_empty": dropped,
        "rows_after_drop": n_clean,
        "duplicates": {
            "n_duplicate_rows": dups.n_duplicate_rows,
            "n_duplicate_names": dups.n_duplicate_names,
            "n_conflicting_names": dups.n_conflicting_names,
        },
        "rows_after_sample": n_sampled,
        "max_rows": cfg.max_rows,
        "min_count": cfg.min_count,
        "rare_l3_classes": int(len(rare)),
        "rare_l3_rows": int(rare["count"].sum()) if len(rare) else 0,
        "rows_after_min_count": int(len(df)),
        "n_classes": {lvl: maps.n_classes(lvl) for lvl in LEVELS},
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
    }
    return PreparedData(splits=splits, maps=maps, report=report, rare_classes=rare, duplicates=dups)
