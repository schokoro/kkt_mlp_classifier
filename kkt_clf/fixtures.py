"""Детерминированный генератор фикстур из data/ods_labels.csv (§6).

Вызывается скриптом ``scripts/make_fixtures.py`` и тестами.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .data import REQUIRED_COLUMNS, taxonomy_violations

log = logging.getLogger(__name__)

DEFAULT_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ods_labels.csv")

# Строки, дающие нулевой вектор признаков: пустые, пробельные, символы вне словаря.
EMPTY_FEATURE_NAMES = [None, "", "   ", "\t\n", "ꙮ", "ꙮꙮ ꙮꙮ", "⁣⁣"]

UNKNOWN = {"L1": "неизвестный l1", "L2": "неизвестный l2", "L3": "неизвестный l3"}


def load_ods(src: str = DEFAULT_SRC) -> tuple[pd.DataFrame, dict]:
    """ODS → основная фикстура. Нарушения путей и дубликаты (если есть) отфильтровываются."""
    df = pd.read_csv(src, dtype=str, keep_default_na=False)[list(REQUIRED_COLUMNS)]
    info = {"rows_src": int(len(df))}
    v = taxonomy_violations(df)
    bad3 = set(v.loc[v["level"] == "L3", "child"])
    bad2 = set(v.loc[v["level"] == "L2", "child"])
    df = df[~df["L3"].isin(bad3) & ~df["L2"].isin(bad2)]
    info["taxonomy_violations_removed"] = int(len(v))
    n = len(df)
    df = df.drop_duplicates("name", keep=False)
    info["duplicate_rows_removed"] = int(n - len(df))
    info["rows"] = int(len(df))
    return df.reset_index(drop=True), info


def small_sample(df: pd.DataFrame, seed: int, n_classes: int = 40, per_class: int = 50) -> pd.DataFrame:
    """Быстрая фикстура: самые частые L3, по ``per_class`` строк."""
    top = df["L3"].value_counts().sort_index().sort_values(ascending=False, kind="stable").index[:n_classes]
    d = df[df["L3"].isin(top)]
    d = d.sample(frac=1.0, random_state=seed)
    d = d[d.groupby("L3").cumcount() < per_class]
    return d.sort_index().reset_index(drop=True)


def defect_dups_empty(small: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = small.copy()
    dup_same = base.iloc[rng.choice(len(base), 5, replace=False)]
    # дубликат name с другой (валидной) меткой
    conf_src = base.iloc[rng.choice(len(base), 3, replace=False)].copy()
    other = base.drop_duplicates("L3").set_index("L3")
    for i, (idx, r) in enumerate(conf_src.iterrows()):
        alt = other.drop(index=r["L3"]).iloc[i]
        conf_src.loc[idx, ["L1", "L2", "L3"]] = [alt["L1"], alt["L2"], alt.name]
    empties = pd.DataFrame([
        {"name": None, "L1": base.L1[0], "L2": base.L2[0], "L3": base.L3[0]},
        {"name": "", "L1": base.L1[0], "L2": base.L2[0], "L3": base.L3[0]},
        {"name": "   ", "L1": base.L1[0], "L2": base.L2[0], "L3": base.L3[0]},
        {"name": "товар без l2", "L1": base.L1[1], "L2": None, "L3": base.L3[1]},
        {"name": "товар с пустым l3", "L1": base.L1[1], "L2": base.L2[1], "L3": ""},
    ])
    return pd.concat([base, dup_same, conf_src, empties], ignore_index=True)


def defect_taxonomy(small: pd.DataFrame) -> pd.DataFrame:
    d = small.copy()
    l2s = sorted(d["L2"].unique())
    l1s = sorted(d["L1"].unique())
    r0 = d.iloc[0]
    other_l2 = next(x for x in l2s if x != r0["L2"])
    other_l1 = next((x for x in l1s if x != r0["L1"]), "другой l1")
    bad = pd.DataFrame([
        {"name": "нарушение l3 к другому l2", "L1": r0["L1"], "L2": other_l2, "L3": r0["L3"]},
        {"name": "нарушение l2 к другому l1", "L1": other_l1, "L2": r0["L2"], "L3": r0["L3"] + " x"},
    ])
    return pd.concat([d, bad], ignore_index=True)


def inference_inputs(df: pd.DataFrame, seed: int, n: int = 300) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    s = df.iloc[np.sort(rng.choice(len(df), n, replace=False))].reset_index(drop=True)
    empt = pd.DataFrame({"name": EMPTY_FEATURE_NAMES})
    no_labels = pd.concat([s[["name"]], empt], ignore_index=True)
    no_labels.insert(0, "row_id", np.arange(len(no_labels)))
    no_labels["source"] = "fixture"
    with_unknown = s.copy()
    k = len(with_unknown)
    with_unknown.loc[0:9, "L3"] = UNKNOWN["L3"]
    with_unknown.loc[10:14, ["L2", "L3"]] = [UNKNOWN["L2"], UNKNOWN["L3"]]
    with_unknown.loc[15:17, ["L1", "L2", "L3"]] = [UNKNOWN["L1"], UNKNOWN["L2"], UNKNOWN["L3"]]
    e = empt.assign(L1=s.L1[k - 1], L2=s.L2[k - 1], L3=s.L3[k - 1])
    with_unknown = pd.concat([with_unknown, e], ignore_index=True)
    with_unknown.insert(0, "row_id", np.arange(len(with_unknown)))
    empty_only = pd.concat([empt, s[["name"]].head(5)], ignore_index=True)
    return {"infer_no_labels": no_labels, "infer_unknown_labels": with_unknown, "infer_empty_features": empty_only}


def build_all(out_dir: str, src: str = DEFAULT_SRC, seed: int = 42) -> dict[str, str]:
    """Строит весь набор фикстур в ``out_dir``. Возвращает пути."""
    os.makedirs(out_dir, exist_ok=True)
    ods, info = load_ods(src)
    log.info("ODS: %s", info)
    small = small_sample(ods, seed)
    sets = {
        "ods": ods,
        "ods_small": small,
        "defect_dups_empty": defect_dups_empty(small, seed),
        "defect_taxonomy": defect_taxonomy(small),
        **inference_inputs(ods, seed),
    }
    paths = {}
    for name, d in sets.items():
        p = os.path.join(out_dir, f"{name}.parquet")
        d.to_parquet(p, index=False)
        paths[name] = p
    pd.Series(info).to_json(os.path.join(out_dir, "ods_info.json"), force_ascii=False, indent=2)
    return paths
