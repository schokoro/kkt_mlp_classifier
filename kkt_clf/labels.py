"""Маппинги классов label ↔ id и таблицы родителей (§4.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS = ("L1", "L2", "L3")


@dataclass
class LabelMaps:
    l1: list[str]
    l2: list[str]
    l3: list[str]
    parent_l2_of_l3: np.ndarray
    parent_l1_of_l2: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "LabelMaps":
        """Строит маппинги по данным (после фильтрации редких классов).

        Порядок детерминирован: сортировка по строке. Таксономия должна быть
        проверена заранее (у L3 один L2, у L2 один L1).
        """
        l1 = sorted(df["L1"].unique().tolist())
        l2 = sorted(df["L2"].unique().tolist())
        l3 = sorted(df["L3"].unique().tolist())
        l1_id = {v: i for i, v in enumerate(l1)}
        l2_id = {v: i for i, v in enumerate(l2)}
        l3_id = {v: i for i, v in enumerate(l3)}
        pairs32 = df[["L3", "L2"]].drop_duplicates()
        pairs21 = df[["L2", "L1"]].drop_duplicates()
        if len(pairs32) != len(l3) or len(pairs21) != len(l2):
            raise ValueError("Таксономия неоднозначна: сначала вызовите check_taxonomy")
        p23 = np.full(len(l3), -1, dtype=np.int64)
        p12 = np.full(len(l2), -1, dtype=np.int64)
        for c, p in zip(pairs32["L3"], pairs32["L2"]):
            p23[l3_id[c]] = l2_id[p]
        for c, p in zip(pairs21["L2"], pairs21["L1"]):
            p12[l2_id[c]] = l1_id[p]
        return cls(l1, l2, l3, p23, p12)

    def labels(self, level: str) -> list[str]:
        return {"L1": self.l1, "L2": self.l2, "L3": self.l3}[level]

    def n_classes(self, level: str) -> int:
        return len(self.labels(level))

    def encode(self, level: str, values) -> np.ndarray:
        """Строки → id; неизвестные метки (и null) → -1."""
        index = pd.Index(self.labels(level))
        return index.get_indexer(pd.Series(values, dtype=object)).astype(np.int64)

    def decode(self, level: str, ids: np.ndarray) -> np.ndarray:
        """id → строки; -1 → None."""
        labels = np.array(self.labels(level) + [None], dtype=object)
        ids = np.asarray(ids)
        return labels[np.where(ids < 0, len(labels) - 1, ids)]

    def l2_of_l3(self, l3_ids: np.ndarray) -> np.ndarray:
        return self.parent_l2_of_l3[l3_ids]

    def l1_of_l3(self, l3_ids: np.ndarray) -> np.ndarray:
        return self.parent_l1_of_l2[self.parent_l2_of_l3[l3_ids]]

    def to_dict(self) -> dict:
        return {
            "L1": self.l1,
            "L2": self.l2,
            "L3": self.l3,
            "parent_l2_of_l3": self.parent_l2_of_l3.tolist(),
            "parent_l1_of_l2": self.parent_l1_of_l2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelMaps":
        return cls(
            list(d["L1"]),
            list(d["L2"]),
            list(d["L3"]),
            np.asarray(d["parent_l2_of_l3"], dtype=np.int64),
            np.asarray(d["parent_l1_of_l2"], dtype=np.int64),
        )
