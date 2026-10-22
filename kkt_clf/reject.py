"""Отказ от классификации и выбор τ (§4.6).

Приоритет причин отказа: ``empty_features`` → ``low_confidence`` → ``mismatch``.
Флаги рассогласования голов считаются всегда, независимо от того, участвуют ли
они в отказе.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import LabelMaps
from .metrics import macro_f1

log = logging.getLogger(__name__)

EMPTY = "empty_features"
LOW_CONF = "low_confidence"
MISMATCH = "mismatch"


class TauTargetWarning(UserWarning):
    pass


@dataclass
class RejectRule:
    tau: float
    reject_on_mismatch_l2: bool = False
    reject_on_mismatch_l1: bool = False

    def to_dict(self) -> dict:
        return {
            "tau": float(self.tau),
            "reject_on_mismatch_l2": bool(self.reject_on_mismatch_l2),
            "reject_on_mismatch_l1": bool(self.reject_on_mismatch_l1),
        }


def mismatch_flags(l3_top1: np.ndarray, l2_head: np.ndarray, l1_head: np.ndarray, maps: LabelMaps):
    """Родитель argmax L3 ≠ argmax L2 (и аналогично для L1)."""
    mm_l2 = maps.l2_of_l3(l3_top1) != l2_head
    mm_l1 = maps.l1_of_l3(l3_top1) != l1_head
    return mm_l2, mm_l1


def apply_reject(
    l3_prob: np.ndarray,
    empty: np.ndarray,
    mm_l2: np.ndarray,
    mm_l1: np.ndarray,
    rule: RejectRule,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (rejected, reject_reason); reason — object-массив со строками или None."""
    reason = np.full(len(l3_prob), None, dtype=object)
    mm = np.zeros(len(l3_prob), dtype=bool)
    if rule.reject_on_mismatch_l2:
        mm |= mm_l2
    if rule.reject_on_mismatch_l1:
        mm |= mm_l1
    reason[mm] = MISMATCH
    reason[l3_prob < rule.tau] = LOW_CONF
    reason[empty] = EMPTY
    return ~pd.isna(reason), reason


def tau_table(
    y_true: np.ndarray,
    l3_top1: np.ndarray,
    l3_prob: np.ndarray,
    empty: np.ndarray,
    mm_l2: np.ndarray,
    mm_l1: np.ndarray,
    grid: list[float],
    n_classes: int,
    reject_on_mismatch_l2: bool = False,
    reject_on_mismatch_l1: bool = False,
) -> pd.DataFrame:
    """«Покрытие / accuracy / macro-F1 принятых» по сетке τ."""
    rows = []
    correct = l3_top1 == y_true
    for tau in sorted(set(float(t) for t in grid)):
        rejected, _ = apply_reject(
            l3_prob, empty, mm_l2, mm_l1, RejectRule(tau, reject_on_mismatch_l2, reject_on_mismatch_l1)
        )
        acc_mask = ~rejected
        n_acc = int(acc_mask.sum())
        rows.append({
            "tau": tau,
            "coverage": n_acc / len(y_true) if len(y_true) else np.nan,
            "n_accepted": n_acc,
            "accuracy": float(correct[acc_mask].mean()) if n_acc else np.nan,
            "macro_f1": macro_f1(y_true[acc_mask], l3_top1[acc_mask], n_classes) if n_acc else np.nan,
        })
    return pd.DataFrame(rows)


def select_tau(table: pd.DataFrame, mode: str, tau: float, target_precision: float) -> dict:
    """Выбор τ: ``fixed`` или минимальный τ с accuracy принятых ≥ цели.

    Если цель недостижима — τ с максимальной accuracy и предупреждение.
    """
    if mode == "fixed":
        return {"tau": float(tau), "mode": mode, "warning": None}
    t = table.dropna(subset=["accuracy"])
    ok = t[t["accuracy"] >= target_precision]
    if len(ok):
        row = ok.sort_values("tau").iloc[0]
        return {"tau": float(row["tau"]), "mode": mode, "target_precision": target_precision,
                "val_accuracy": float(row["accuracy"]), "val_coverage": float(row["coverage"]),
                "warning": None}
    if t.empty:
        msg = "Ни при одном τ из сетки нет принятых примеров; оставлен τ из конфига"
        warnings.warn(msg, TauTargetWarning, stacklevel=2)
        log.warning(msg)
        return {"tau": float(tau), "mode": mode, "target_precision": target_precision, "warning": msg}
    best = t.sort_values(["accuracy", "tau"], ascending=[False, True]).iloc[0]
    msg = (
        f"Целевая точность {target_precision:.4f} недостижима на val ни при одном τ из сетки; "
        f"взят τ={best['tau']:.2f} с максимальной точностью {best['accuracy']:.4f} "
        f"(покрытие {best['coverage']:.4f})"
    )
    warnings.warn(msg, TauTargetWarning, stacklevel=2)
    log.warning(msg)
    return {"tau": float(best["tau"]), "mode": mode, "target_precision": target_precision,
            "val_accuracy": float(best["accuracy"]), "val_coverage": float(best["coverage"]),
            "warning": msg}
