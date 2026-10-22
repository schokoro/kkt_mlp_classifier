"""Метрики и отчёты (§4.7)."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

UNKNOWN_ROW = "__unknown_label__"
NO_PREDICTION = "__no_prediction__"


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Быстрый macro-F1 по классам, встречающимся в y_true или y_pred (как в sklearn)."""
    if len(y_true) == 0:
        return float("nan")
    tp = np.bincount(y_true[y_true == y_pred], minlength=n_classes).astype(np.float64)
    t = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    p = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    used = (t + p) > 0
    denom = t + p
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[used].mean())


def report_df(y_true, y_pred, n_unknown: int = 0) -> pd.DataFrame:
    """classification_report(output_dict=True) → DataFrame с колонкой support.

    ``n_unknown`` — число строк с метками вне маппинга модели; добавляется
    отдельной строкой ``__unknown_label__`` (метрики NaN).
    """
    if len(y_true):
        # Строки без предсказания считаются ошибкой, но отдельным классом в отчёт не входят.
        labels = sorted((set(y_true) | set(y_pred)) - {NO_PREDICTION})
        with warnings.catch_warnings():
            # на малых выборках (val/test редких классов) sklearn шумит «похоже на регрессию»
            warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")
            d = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        if "accuracy" not in d:
            d["accuracy"] = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    else:
        d = {}
    acc = d.pop("accuracy", None)
    df = pd.DataFrame(d).T
    if df.empty:
        df = pd.DataFrame(columns=["precision", "recall", "f1-score", "support"])
    df.index.name = "label"
    extra = []
    if acc is not None:
        extra.append(pd.DataFrame({"precision": [np.nan], "recall": [np.nan], "f1-score": [acc],
                                   "support": [len(y_true)]}, index=pd.Index(["accuracy"], name="label")))
    if n_unknown:
        extra.append(pd.DataFrame({"precision": [np.nan], "recall": [np.nan], "f1-score": [np.nan],
                                   "support": [n_unknown]}, index=pd.Index([UNKNOWN_ROW], name="label")))
    if extra:
        df = pd.concat([df] + extra)
    df["support"] = df["support"].astype(int)
    return df


def summary_row(rep: pd.DataFrame) -> dict:
    def get(row, col):
        return float(rep.loc[row, col]) if row in rep.index else float("nan")

    return {
        "accuracy": get("accuracy", "f1-score"),
        "macro_f1": get("macro avg", "f1-score"),
        "weighted_f1": get("weighted avg", "f1-score"),
        "support": int(rep.loc["accuracy", "support"]) if "accuracy" in rep.index else 0,
    }


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> pd.DataFrame:
    """Топ-N пар «истинный → предсказанный» по числу ошибок."""
    err = y_true != y_pred
    df = pd.DataFrame({"true": y_true[err], "pred": y_pred[err]})
    if df.empty:
        return pd.DataFrame(columns=["true", "pred", "count"])
    out = df.value_counts().rename("count").reset_index()
    return out.sort_values(["count", "true", "pred"], ascending=[False, True, True]).head(n).reset_index(drop=True)


def mismatch_rates(correct: np.ndarray, mm_l2: np.ndarray, mm_l1: np.ndarray) -> pd.DataFrame:
    """Доля рассогласований голов для верных и неверных предсказаний L3."""
    rows = []
    for name, m in (("all", np.ones_like(correct, dtype=bool)), ("l3_correct", correct), ("l3_wrong", ~correct)):
        n = int(m.sum())
        rows.append({
            "group": name,
            "n": n,
            "mismatch_l2_rate": float(mm_l2[m].mean()) if n else np.nan,
            "mismatch_l1_rate": float(mm_l1[m].mean()) if n else np.nan,
            "mismatch_any_rate": float((mm_l2 | mm_l1)[m].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def f1_vs_frequency(rep_l3: pd.DataFrame, train_counts: dict[str, int]) -> pd.DataFrame:
    """F1 по классам L3 и частота класса в train."""
    skip = {"accuracy", "macro avg", "weighted avg", UNKNOWN_ROW}
    rows = rep_l3.loc[[i for i in rep_l3.index if i not in skip]]
    out = pd.DataFrame({
        "label": rows.index,
        "f1": rows["f1-score"].to_numpy(dtype=float),
        "support": rows["support"].to_numpy(dtype=int),
        "train_count": [int(train_counts.get(l, 0)) for l in rows.index],
    })
    return out.sort_values("train_count", ascending=False).reset_index(drop=True)


def evaluate(
    true_labels: dict[str, np.ndarray],
    known: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    accepted: np.ndarray,
    mm_l2: np.ndarray,
    mm_l1: np.ndarray,
    top_n_errors: int = 50,
    noisy_support: int = 2,
    has_pred: np.ndarray | None = None,
) -> dict:
    """Полный набор отчётов §4.7.

    ``true_labels[L]`` — строки истинных меток; ``known[L]`` — маска «метка есть в
    маппинге»; ``pred`` — строковые предсказания: ``l1_head``, ``l2_head``,
    ``l1_from_l3``, ``l2_from_l3``, ``l3``.
    """
    reports: dict[str, pd.DataFrame] = {}
    summary = []
    variants = {
        "no_reject": [("L1", "head", "l1_head"), ("L1", "from_l3", "l1_from_l3"),
                      ("L2", "head", "l2_head"), ("L2", "from_l3", "l2_from_l3"),
                      ("L3", "head", "l3")],
        "with_reject": [("L1", "from_l3", "l1_from_l3"), ("L2", "from_l3", "l2_from_l3"),
                        ("L3", "head", "l3")],
    }
    for variant, items in variants.items():
        for level, source, key in items:
            m = known[level].copy()
            if variant == "with_reject":
                m &= accepted
            n_unknown = int((~known[level] & (accepted if variant == "with_reject" else True)).sum())
            rep = report_df(true_labels[level][m], pred[key][m], n_unknown)
            name = f"{variant}__{level}__{source}"
            reports[name] = rep
            s = summary_row(rep)
            s.update({"variant": variant, "level": level, "source": source, "n_unknown_labels": n_unknown})
            summary.append(s)
    summary_df = pd.DataFrame(summary)[
        ["variant", "level", "source", "accuracy", "macro_f1", "weighted_f1", "support", "n_unknown_labels"]
    ]
    k3 = known["L3"]
    y3 = true_labels["L3"]
    p3 = pred["l3"]
    correct = (y3 == p3) & k3
    hp = np.ones(len(y3), dtype=bool) if has_pred is None else has_pred
    km = k3 & hp
    coverage = float(accepted.mean()) if len(accepted) else float("nan")
    rep_l3 = reports["no_reject__L3__head"]
    cls_rows = rep_l3.drop(index=[i for i in ("accuracy", "macro avg", "weighted avg", UNKNOWN_ROW) if i in rep_l3.index])
    return {
        "reports": reports,
        "summary": summary_df,
        "coverage": coverage,
        "n_rows": int(len(accepted)),
        "n_accepted": int(accepted.sum()),
        "top_errors": top_confusions(y3[k3], p3[k3], top_n_errors),
        "mismatch_rates": mismatch_rates(correct[km], mm_l2[km], mm_l1[km]),
        "n_noisy_l3_classes": int((cls_rows["support"] <= noisy_support).sum()),
    }


def save_eval(ev: dict, out_dir: str, prefix: str = "") -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, rep in ev["reports"].items():
        p = os.path.join(out_dir, f"{prefix}report__{name}.csv")
        rep.to_csv(p)
        paths[name] = p
    for key in ("summary", "top_errors", "mismatch_rates"):
        p = os.path.join(out_dir, f"{prefix}{key}.csv")
        ev[key].to_csv(p, index=False)
        paths[key] = p
    if "f1_vs_freq" in ev:
        p = os.path.join(out_dir, f"{prefix}f1_vs_freq.csv")
        ev["f1_vs_freq"].to_csv(p, index=False)
        paths["f1_vs_freq"] = p
    return paths
