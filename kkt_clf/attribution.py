"""Gradient × Input по L3-margin (S04): атрибуция, рейтинги и class-aware отбор.

Метод объясняет отрыв логита истинной L3 от сильнейшего конкурента. Для
обучающего примера ``i`` с истинным классом ``y_i`` и логитами головы L3
``z_i``:

- конкурент ``r_i = argmax_{c ≠ y_i} z_{i,c}`` — сильнейший **другой** класс;
  при равенстве логитов побеждает наименьший индекс класса (детерминированно);
- margin ``m_i = z_{i,y_i} − z_{i,r_i}`` (логит, не вероятность/CE);
- для **активного** TF-IDF признака ``f``: ``a_{i,f} = x_{i,f} · ∂m_i/∂x_{i,f}``
  (Gradient × Input). Индекс ``r_i`` фиксируется до градиентного шага и внутри
  него не дифференцируется.

Свойства реализации:

- атрибуция выполняется для обученной модели в ``model.eval()`` (dropout выкл.,
  BatchNorm по running statistics) **без обновления весов**: градиент считается
  ``torch.autograd.grad`` по значениям ненулевых элементов CSR
  (``per_sample_weights`` EmbeddingBag), поэтому не материализуются и не
  требуются плотные матрицы ``N×F`` и ``B×F``;
- градиенты для отсутствующих признаков не создаются: вклад существует только
  в позициях ненулевых значений CSR; пустые TF-IDF-векторы обрабатываются явно
  (margin для них определён через bias-путь, вкладов нет, но в N_c класса
  входят, поэтому класс только из пустых строк имеет пустой список
  положительных кандидатов, а не исчезает из рейтингов);
- ошибочно классифицированные примеры сохраняются; конкурент определяется
  одинаково для правильных и ошибочных предсказаний; фиксируются доли
  correct/incorrect и диагностика по margin;
- class-aware ранжирование: ``score(c,f) = (1/N_c)·Σ_{i: y_i=c} max(0, a_{i,f})``,
  где ``N_c`` — число **всех** примеров класса в атрибуции (пустые TF-IDF-строки
  входят в знаменатель, вкладов не добавляя; неналичие признака и
  неположительные вклады дают ноль в сумме); отрицательные и нулевые вклады в
  рейтинг не попадают, но полностью отражаются в диагностике (число, сумма
  модулей, доля); сохранённые ``rank_counts`` — числа положительных кандидатов
  в списках классов, а ``N_c`` хранятся отдельно (``class_counts``). Рейтинги
  char/word раздельны; равные оценки разрываются возрастанием индекса исходного
  признака;
- объединение категорий — тот же равноправный циклический механизм S03
  (:func:`kkt_clf.selection.roundrobin_select_from_rank_files`) по полным
  рейтингам на диске (RAM не хранит ``C×F`` и ``C×K``); при нехватке
  положительных кандидатов — детерминированный добор из глобального рейтинга
  положительных атрибуций, затем из остальных признаков в стабильном порядке
  с обязательной маркировкой ``remainder_fill``;
- метки validation/test в атрибуцию и ранжирование не попадают: вход — только
  размеченный train (возможен воспроизводимый сэмплинг строк, см.
  :class:`AttributionConfig`).

Полные рейтинги сохраняются в том же формате, что и S03
(``kkt-selection-ratings@2``), — по ним можно повторно применить бюджет без
пересчёта атрибуции. Write path дополняет манифест контрольными суммами файлов
(``finalize_ratings_manifest``), а повторное использование каталога —
:func:`select_features_gradient_x_input` — проходит fail-closed проверку
целостности (``validate_ratings_dir``): отсутствие любого объявленного файла
обеих веток, несогласованность классов/счётчиков/длин/ширин, неполная секция
контрольных сумм (при её наличии она обязана покрывать каждый объявленный
файл) или подмена содержимого останавливают отбор с ``SelectionArtifactError``,
не молча меняя результат; ``class_counts`` обеих веток обязаны совпадать
поэлементно и суммой сходиться с ``attribution.n_attributed_rows``. Это инструментология PoC: интерпретация обученной модели
не доказывает полезность признака после переобучения, а результаты на ODS не
переносятся в закрытый контур.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from .artifact import environment_versions
from .model import SparseBatch, csr_to_batch
from .selection import (
    RATINGS_FORMAT_VERSION,
    BranchSelection,
    SelectionArtifactError,
    SelectionConfig,
    SelectionError,
    SelectionResult,
    branch_budgets,
    finalize_ratings_manifest,
    roundrobin_select_from_rank_files,
    validate_ratings_dir,
)

log = logging.getLogger(__name__)

METHOD = "gradient_x_input"
TARGET_NAME = "true_l3_vs_best_other_l3_logit_margin"
# Имя файла provenance в каталоге checkpoint'а attribution MLP: проверяемый
# отпечаток полного контракта признаков и фактических сплитов, сохранённый при
# обучении checkpoint'а и обязательный при его загрузке (fail closed).
ATTRIBUTION_PROVENANCE_FILENAME = "attribution_provenance.json"
FORMULA = (
    "m_i = z_{i,y_i} - z_{i,r_i}, r_i = argmax_{c != y_i} z_{i,c} "
    "(ties: lowest class index, fixed before the gradient step); "
    "a_{i,f} = x_{i,f} * dm_i/dx_{i,f} over nonzero CSR values; "
    "score(c,f) = (1/N_c) * sum_{i: y_i=c} max(0, a_{i,f}), "
    "N_c = number of train rows of class c (empty TF-IDF rows included, "
    "contributing 0)"
)

SAMPLING_HEAD = "head"
SAMPLING_RANDOM = "random"
SAMPLINGS = (SAMPLING_HEAD, SAMPLING_RANDOM)


class AttributionError(RuntimeError):
    """Базовая ошибка операций Gradient × Input-атрибуции."""


@dataclass
class AttributionConfig:
    """Параметры Gradient × Input-атрибуции (D02).

    ``checkpoint_path`` — загрузить проверенный checkpoint MLP на полном
    TF-IDF; ``None`` — обучить attribution MLP из конфигурации эксперимента
    (общие model/loss/train, full TF-IDF, early stopping по validation в рамках
    существующего протокола). ``max_rows``/``sampling``/seed эксперимента —
    воспроизводимый отбор размеченных train-строк для атрибуции (None — все
    строки). ``batch_size`` — батчи ненулезных CSR-значений; плотные ``B×F`` не
    создаются при любом размере.
    """

    checkpoint_path: str | None = None
    max_rows: int | None = None
    sampling: str = SAMPLING_HEAD
    batch_size: int = 4096

    def validate(self) -> None:
        if self.max_rows is not None and (not isinstance(self.max_rows, int) or self.max_rows <= 0):
            raise AttributionError(f"attribution.max_rows: None или целое > 0, получено {self.max_rows!r}")
        if self.sampling not in SAMPLINGS:
            raise AttributionError(f"attribution.sampling: одно из {SAMPLINGS}, получено {self.sampling!r}")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise AttributionError(f"attribution.batch_size: целое > 0, получено {self.batch_size!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class MarginAttributions:
    """Результат Gradient × Input по margin на размеченном train.

    ``values`` — CSR float64 той же формы, что и входная матрица: значения
    ``a_{i,f} = x_{i,f}·∂m_i/∂x_{i,f}`` ровно в позициях ненулевых элементов
    (градиенты отсутствующих признаков не материализуются). Для пустых строк
    значений нет. ``margins``/``competitor``/``correct`` определены для всех
    строк (включая пустые: margin через bias-путь), ``empty`` — маска пустых.
    ``rows`` — исходные индексы строк входной матрицы (после сэмплинга).
    """

    values: sp.csr_matrix
    margins: np.ndarray       # float64 [N]
    competitor: np.ndarray    # int64 [N], класс-индекс r_i (всегда != y_i)
    correct: np.ndarray       # bool [N], top-1 == y_i
    empty: np.ndarray         # bool [N], нулевой TF-IDF-вектор
    y: np.ndarray             # int64 [N], истинные id L3 (только train)
    rows: np.ndarray          # int64 [N], исходные индексы строк входа
    n_classes: int

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])


def competitor_indices(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Детерминированный конкурент ``argmax_{c ≠ y_i} z_{i,c}``.

    Истинный класс маскируется ``-inf``; при равенстве логитов выбирается
    наименьший индекс класса (первое вхождение максимума, гарантия numpy).
    Работает одинаково для правильных и ошибочных предсказаний.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if z.ndim != 2 or z.shape[0] != len(y):
        raise AttributionError(f"logits/y не согласованы: {z.shape} и {len(y)}")
    masked = z.copy()
    masked[np.arange(len(y)), y] = -np.inf
    return np.argmax(masked, axis=1).astype(np.int64)


@torch.no_grad()
def _l3_logits(model: torch.nn.Module, batch: SparseBatch, device: torch.device) -> torch.Tensor:
    return model(batch.to(device))[2]


def compute_margin_attributions(
    model: torch.nn.Module,
    X: sp.csr_matrix,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> MarginAttributions:
    """Gradient × Input по margin для обученной модели (D03, D04).

    Модель переводится в ``eval()``; веса не обновляются (градиент —
    ``torch.autograd.grad`` только по значениям ненулевых CSR-элементов,
    параметры в ``.grad`` не накапливаются). Конкурент фиксируется по логитам
    из отдельного прохода под ``no_grad`` и в градиентном шаге не
    дифференцируется. ``X`` — train-матрица полного TF-IDF (CSR); ``y`` —
    целочисленные id L3 только размеченного train (значения < 0 отклоняются).
    Возвращает разреженные вклады той же структуры, что и ``X``.
    """
    if not sp.isspmatrix_csr(X):
        X = sp.csr_matrix(X)
    y = np.asarray(y, dtype=np.int64)
    if X.shape[0] != len(y):
        raise AttributionError(f"X/y не согласованы: X{X.shape}, len(y)={len(y)}")
    if X.shape[1] == 0:
        raise AttributionError("Пустое пространство признаков: атрибуция невозможна")
    if len(y) == 0:
        raise AttributionError("Пустая выборка: атрибуция невозможна")
    if np.any(y < 0):
        raise AttributionError("В y есть неразмеченные строки (-1): атрибуция выполняется только по размеченному train")
    n_classes = int(model.n_classes[2]) if hasattr(model, "n_classes") else int(y.max()) + 1
    if n_classes < 2:
        raise AttributionError("Для margin нужно не менее двух классов L3")
    if batch_size <= 0:
        raise AttributionError(f"batch_size: целое > 0, получено {batch_size!r}")

    was_training = model.training
    model.eval()
    device = torch.device(device)
    n = X.shape[0]
    att_data = np.empty(X.nnz, dtype=np.float64)
    margins = np.empty(n, dtype=np.float64)
    competitor = np.empty(n, dtype=np.int64)
    correct = np.empty(n, dtype=bool)
    empty = np.diff(X.indptr) == 0
    pos = 0
    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X[start:end]
            batch = csr_to_batch(Xb)
            logits = _l3_logits(model, batch, device).float().cpu().numpy()
            yb = y[start:end]
            competitor[start:end] = competitor_indices(logits, yb)
            margins[start:end] = logits[np.arange(len(yb)), yb] - logits[
                np.arange(len(yb)), competitor[start:end]
            ]
            correct[start:end] = np.argmax(logits, axis=1) == yb
            nnz_b = Xb.nnz
            if nnz_b:
                weights = torch.from_numpy(Xb.data.astype(np.float32, copy=True)).to(device)
                weights.requires_grad_(True)
                # Все тензоры батча на целевом устройстве: indices/offsets при
                # градиентном проходе тоже переносятся (иначе на CUDA ошибка
                # смешанных устройств EmbeddingBag).
                dev_batch = SparseBatch(
                    batch.indices.to(device), batch.offsets.to(device), weights
                )
                _, _, lg3 = model(dev_batch)
                arange = torch.arange(len(yb), device=device)
                r = torch.from_numpy(competitor[start:end]).to(device)
                yy = torch.from_numpy(yb).to(device)
                margin_sum = (lg3[arange, yy] - lg3[arange, r]).sum()
                grad = torch.autograd.grad(margin_sum, weights)[0]
                att_data[pos : pos + nnz_b] = (weights.detach() * grad).double().cpu().numpy()
                pos += nnz_b
    finally:
        if was_training:
            model.train()
    if pos != X.nnz:
        raise AttributionError(f"Внутренняя ошибка атрибуции: собрано {pos} вкладов при nnz={X.nnz}")
    values = sp.csr_matrix(
        (att_data, X.indices.astype(np.int64), X.indptr.astype(np.int64)), shape=X.shape
    )
    return MarginAttributions(
        values=values,
        margins=margins,
        competitor=competitor,
        correct=correct,
        empty=empty,
        y=y,
        rows=np.arange(n, dtype=np.int64),
        n_classes=n_classes,
    )


def select_attributed_rows(n_rows: int, cfg: AttributionConfig, seed: int) -> np.ndarray:
    """Воспроизводимый отбор строк train для атрибуции (D02).

    ``max_rows=None`` (или >= n_rows) — все строки в исходном порядке;
    ``sampling="head"`` — первые ``max_rows``; ``"random"`` — детерминированный
    выбор без возвращения через ``np.random.default_rng(seed)``, отсортированный
    по возрастанию индекса (порядок батчей стабилен).
    """
    if n_rows <= 0:
        raise AttributionError("Пустой train: отбирать нечего")
    if cfg.max_rows is None or cfg.max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    if cfg.sampling == SAMPLING_HEAD:
        return np.arange(cfg.max_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=cfg.max_rows, replace=False)).astype(np.int64)


def _class_segments(y_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Границы сегментов классов в отсортированном y: (classes, start, end)."""
    classes, starts, counts = np.unique(y_sorted, return_index=True, return_counts=True)
    return classes.astype(np.int64), starts.astype(np.int64), (starts + counts).astype(np.int64)


def build_gxi_ratings(
    att: MarginAttributions,
    n_char: int,
    ratings_dir: str,
    l3_labels: list[str] | None = None,
) -> tuple[dict, dict]:
    """Class-aware рейтинги Gradient × Input на диске (D05, D06).

    Считает ``score(c,f) = mean_{i∈c} max(0, a_{i,f})`` по всем строкам train
    класса: ``N_c`` включает пустые TF-IDF-векторы (вкладов нет, но класс из
    одних пустых строк сохраняется с пустым списком кандидатов). Пишет полные
    рейтинги в ``ratings_dir`` в формате S03 (``kkt-selection-ratings@2``):
    глобальный рейтинг ветки (средний положительный вклад по всем
    атрибутированным строкам), списки положительных кандидатов категорий
    (``-score`` ↓, равные — по индексу признака); ``rank_counts`` — длины
    этих списков (числа кандидатов, они же смещения сегментов), а ``N_c``
    сохраняются отдельно в ``class_counts``. Формирует диагностику по всем
    строкам: число/сумма/доля положительных, отрицательных (сумма модулей) и
    нулевых вкладов, доли correct/incorrect, пустые строки, статистика margin,
    число строк по L3 и покрытие классов.

    Память: последовательная агрегация — плотный вектор одной ветки одного
    класса за раз (``O(F_branch)``, при ~1000 L3 и 400k признаках ≈ 3 МБ на
    класс, не ``C×F``); списки кандидатов всех классов в RAM не накапливаются
    и не конкатенируются — сегменты классов пишутся в файлы рейтингов потоково,
    поэтому худший случай не растёт как число пар «класс–признак»; исходные
    вклады не сохраняются построчно.
    """
    if not 0 < n_char < att.values.shape[1]:
        raise AttributionError(f"n_char: граница веток в (0, {att.values.shape[1]}), получено {n_char!r}")
    if att.n_rows == 0:
        raise AttributionError("Нет строк для агрегации рейтингов")
    os.makedirs(ratings_dir, exist_ok=True)

    n_attributed = int(att.n_rows)
    n_nonempty = int((~att.empty).sum())
    n_empty = int(att.empty.sum())
    # Пустые TF-IDF-строки не дают вкладов (суммы сегментов нулевые), но входят
    # в N_c знаменателя класса и не исключают класс из рейтингов — у класса
    # только из пустых строк список положительных кандидатов пуст. Число
    # атрибутированных строк — по всем строкам (margin и классификация
    # вычислены для каждой, пустые входят в N_c), поэтому correct+incorrect и
    # сумма per_class_counts сходятся с ним; непустые зафиксированы отдельно
    # (ревью 2, находка 4).
    n_total_values = int(att.values.nnz)

    order = np.argsort(att.y, kind="stable")
    # Единственная упорядоченная по классам копия вкладов: сегменты классов
    # читаются из неё, положительные части сегментов считаются транзитно
    # (копия одного сегмента за раз). Полные списки кандидатов всех классов в
    # RAM не накапливаются: рейтинги пишутся на диск потоково, класс за классом,
    # поэтому память агрегации ограничена ~O(F_branch) на активный сегмент, а не
    # O(Σ пар «класс–признак»).
    att_vals_sorted = att.values[order]
    y_s = att.y[order]
    classes, starts, ends = _class_segments(y_s)
    n_classes = len(classes)
    n_branch = {"char": int(n_char), "word": int(att.values.shape[1] - n_char)}

    # Диагностика по знакам вкладов (все значения, включая пустые строки — там
    # вкладов нет): положительные идут в рейтинг, отрицательные/нулевые — только
    # в диагностику, нулевые не выдумываются.
    data = att.values.data
    pos_mask = data > 0
    neg_mask = data < 0
    n_positive = int(pos_mask.sum())
    n_negative = int(neg_mask.sum())
    n_zero = int(n_total_values - n_positive - n_negative)
    contributions = {
        "n_values_total": n_total_values,
        "positive": {
            "count": n_positive,
            "share": n_positive / n_total_values if n_total_values else None,
            "sum": float(data[pos_mask].sum()) if n_positive else 0.0,
        },
        "negative": {
            "count": n_negative,
            "share": n_negative / n_total_values if n_total_values else None,
            "abs_sum": float(np.abs(data[neg_mask]).sum()) if n_negative else 0.0,
        },
        "zero": {
            "count": n_zero,
            "share": n_zero / n_total_values if n_total_values else None,
        },
    }

    manifest: dict[str, Any] = {
        "format": RATINGS_FORMAT_VERSION,
        "method": METHOD,
        "branches": {},
        "attribution": {
            "target": TARGET_NAME,
            "formula": FORMULA,
            "n_attributed_rows": n_attributed,
            "n_nonempty_rows": n_nonempty,
            "n_empty_rows": n_empty,
            "n_classes": n_classes,
        },
    }
    per_class: list[dict[str, Any]] = []
    branch_diag: dict[str, Any] = {}
    for branch, offset in (("char", 0), ("word", n_char)):
        width = n_branch[branch]
        g_global = np.zeros(width, dtype=np.float64)
        counts = np.zeros(n_classes, dtype=np.int64)        # положительные кандидаты в списке класса
        class_counts = np.zeros(n_classes, dtype=np.int64)  # N_c: все строки класса (включая пустые)
        idx_path = os.path.join(ratings_dir, f"{branch}_rank_idx.npy")
        score_path = os.path.join(ratings_dir, f"{branch}_rank_score.npy")
        counts_path = os.path.join(ratings_dir, f"{branch}_rank_counts.npy")
        class_counts_path = os.path.join(ratings_dir, f"{branch}_class_counts.npy")
        classes_path = os.path.join(ratings_dir, f"{branch}_classes.npy")
        global_path = os.path.join(ratings_dir, f"{branch}_global.npy")
        f_idx = f_score = None
        n_written = 0
        try:
            f_idx = open(idx_path, "wb")
            f_score = open(score_path, "wb")
            for ci in range(n_classes):
                seg = att_vals_sorted[starts[ci] : ends[ci]]
                n_c = int(ends[ci] - starts[ci])
                class_counts[ci] = n_c
                # Положительная часть сегмента: транзитная копия только сегмента
                # класса (пустые строки вкладов не имеют — суммы нулевые, но в
                # N_c входят).
                pos = seg.copy()
                pos.data = np.maximum(pos.data, 0.0)
                pos.eliminate_zeros()
                sums = np.asarray(pos.sum(axis=0), dtype=np.float64).ravel()[offset : offset + width]
                scores = sums / n_c
                g_global += sums
                pos_idx = np.flatnonzero(scores > 0)
                order_c = np.lexsort((pos_idx, -scores[pos_idx]))
                ranked = pos_idx[order_c]
                counts[ci] = len(ranked)
                # Потоковая запись: списки кандидатов всех классов в RAM не
                # собираются (длины — в counts, смещения восстанавливаются
                # кумулятивной суммой при чтении, см. selection._RankStream).
                ranked.astype(np.int64).tofile(f_idx)
                scores[ranked].astype(np.float64).tofile(f_score)
                n_written += len(ranked)
        finally:
            if f_score is not None:
                f_score.close()
            if f_idx is not None:
                f_idx.close()
        # Глобальный рейтинг ветки: средний положительный вклад по непустым
        # строкам (пустые вкладов не имеют и в знаменатель глобальной средней
        # не входят; в class-aware знаменатель N_c входят — см. формулу выше).
        g_global /= max(n_nonempty, 1)
        counts.tofile(counts_path)
        class_counts.tofile(class_counts_path)
        np.asarray(classes, dtype=np.int64).tofile(classes_path)
        g_global.tofile(global_path)
        manifest["branches"][branch] = {
            "n_features": width,
            "global_scores": f"{branch}_global.npy",
            "classes": f"{branch}_classes.npy",
            "rank_indices": f"{branch}_rank_idx.npy",
            "rank_scores": f"{branch}_rank_score.npy",
            "rank_counts": f"{branch}_rank_counts.npy",
            "class_counts": f"{branch}_class_counts.npy",
        }
        branch_diag[branch] = {"n_positive_candidates": int(n_written)}

    # Статистика по классам (по сегментам отсортированной матрицы вкладов):
    # классы учитываются по всем строкам, включая пустые TF-IDF-векторы.
    correct_sorted = att.correct[order]
    margins_sorted = att.margins[order]
    cls_stats: list[dict[str, Any]] = []
    for ci in range(n_classes):
        cls = int(classes[ci])
        n_c = int(ends[ci] - starts[ci])
        seg_vals = att_vals_sorted[starts[ci] : ends[ci]]
        d = seg_vals.data
        pmask = d > 0
        nmask = d < 0
        row_correct = correct_sorted[starts[ci] : ends[ci]]
        row_margins = margins_sorted[starts[ci] : ends[ci]]
        entry: dict[str, Any] = {
            "class_id": cls,
            "label": str(l3_labels[cls]) if l3_labels is not None and cls < len(l3_labels) else None,
            "n_attributed": n_c,
            "n_correct": int(row_correct.sum()),
            "correct_fraction": float(row_correct.mean()) if n_c else None,
            "margin_mean": float(row_margins.mean()) if n_c else None,
            "positive": {
                "count": int(pmask.sum()),
                "sum": float(d[pmask].sum()) if pmask.any() else 0.0,
            },
            "negative": {
                "count": int(nmask.sum()),
                "abs_sum": float(np.abs(d[nmask]).sum()) if nmask.any() else 0.0,
            },
            "zero": {"count": int((~pmask & ~nmask).sum())},
        }
        n_pos_c = entry["positive"]["count"]
        if n_pos_c == 0:
            entry["reason"] = "no positive Gradient×Input contributions for the class"
        cls_stats.append(entry)

    finalize_ratings_manifest(ratings_dir, manifest)
    with open(os.path.join(ratings_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    report: dict[str, Any] = {
        "target": TARGET_NAME,
        "formula": FORMULA,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": environment_versions(),
        "n_rows_total": int(att.n_rows),
        "n_attributed_rows": n_attributed,
        "n_nonempty_rows": n_nonempty,
        "n_empty_rows": n_empty,
        "n_classes": n_classes,
        "n_classes_with_positive": int(sum(1 for e in cls_stats if e["positive"]["count"] > 0)),
        "n_classes_without_positive": int(sum(1 for e in cls_stats if e["positive"]["count"] == 0)),
        "correct": {
            "count": int(att.correct.sum()),
            "fraction": float(att.correct.mean()) if att.n_rows else None,
        },
        "incorrect": {
            "count": int((~att.correct).sum()),
            "fraction": float((~att.correct).mean()) if att.n_rows else None,
        },
        "margins": {
            "mean": float(att.margins.mean()) if att.n_rows else None,
            "std": float(att.margins.std()) if att.n_rows else None,
            "min": float(att.margins.min()) if att.n_rows else None,
            "max": float(att.margins.max()) if att.n_rows else None,
        },
        "contributions": contributions,
        "branches": branch_diag,
        "per_class": cls_stats,
        "notes": [
            "Атрибуция и агрегирование выполнены только по размеченному train; метки val/test не использовались.",
            "n_attributed_rows — число всех атрибутированных train-строк, включая пустые "
            "TF-IDF-векторы (margin и классификация вычислены для каждой строки, пустые "
            "входят в N_c знаменателя класса); непустые строки зафиксированы отдельно "
            "как n_nonempty_rows, пустые — как n_empty_rows.",
            "N_c — число всех train-примеров класса, включая пустые TF-IDF-векторы "
            "(вкладов у них нет, но они входят в знаменатель score(c,f) = сумма/N_c); "
            "их число зафиксировано отдельно.",
            "Пустые TF-IDF-строки не исключают класс из рейтингов: класс только из "
            "пустых строк имеет пустой список положительных кандидатов (counts=0).",
            "rank_counts — число положительных кандидатов в списке каждого класса "
            "(длины сегментов rank_indices/rank_scores); N_c (число примеров класса) "
            "хранится отдельно в class_counts.",
            "Положительные вклады: max(0, a) > 0; отрицательные/нулевые в рейтинги не входят и здесь учтены "
            "как диагностика (число, сумма модулей, доля). Нулевые вклады не выдумываются.",
            "Полные рейтинги — формат kkt-selection-ratings@2 (те же соглашения, что S03): повторное "
            "применение бюджета без пересчёта атрибуции; оценки float64, ties по индексу признака.",
            "Значения вкладов для каждой строки не сохраняются: только рейтинги и сводная диагностика.",
        ],
    }
    log.info(
        "G×I-рейтинги: строк %d (непустых %d, пустых %d), классов %d (с положительными %d), "
        "вкладов +%d/−%d/0:%d",
        n_attributed, n_nonempty, n_empty, n_classes, report["n_classes_with_positive"],
        n_positive, n_negative, n_zero,
    )
    return manifest, report


def select_features_gradient_x_input(
    ratings_dir: str,
    cfg: SelectionConfig,
    n_char: int,
    n_total: int,
    l3_labels: list[str] | None = None,
) -> SelectionResult:
    """Class-aware Top-K отбор из сохранённых G×I-рейтингов (D05, D06).

    Рейтинги строятся один раз (:func:`build_gxi_ratings`) и повторно
    используются для каждого бюджета K: здесь выполняется только равноправный
    циклический round-robin категорий с дедупликацией в пределах квот веток
    char/word, детерминированный добор из глобального рейтинга положительных
    атрибуций и (при нехватке положительных кандидатов) добор остальных
    признаков в стабильном порядке с маркировкой ``remainder_fill``. Индексы
    результата — полно-пространные (word со смещением ``n_char``).

    Загрузка каталога рейтингов — **fail closed**
    (:func:`kkt_clf.selection.validate_ratings_dir`): отсутствие любого
    объявленного файла обеих веток (включая ``class_counts``, который
    round-robin не читает), несогласованность классов/счётчиков/длин/ширин,
    отсутствие ``class_counts`` в какой-либо ветке, расхождение per-class N_c
    между ветками или с ``attribution.n_attributed_rows``, а также
    несовпадение/неполнота контрольных сумм (если секция записана, она обязана
    покрывать каждый объявленный файл) останавливает отбор с
    :class:`~kkt_clf.selection.SelectionArtifactError`; ранжированные списки
    при проверке в RAM не загружаются и далее читаются потоками.
    """
    if cfg.k is None or cfg.k <= 0:
        raise SelectionError(f"selection.k: обязательное целое > 0, получено {cfg.k!r}")
    if not 0 < n_char < n_total:
        raise SelectionError(f"Граница веток n_char: в (0, {n_total}), получено {n_char!r}")
    if not os.path.isdir(ratings_dir):
        raise SelectionError(f"Каталог G×I-рейтингов {ratings_dir!r} не найден")
    meta_path = os.path.join(ratings_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise SelectionError(f"В каталоге рейтингов {ratings_dir!r} нет meta.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SelectionArtifactError(f"meta.json рейтингов {ratings_dir!r} не читается: {e}") from e
    arrays = validate_ratings_dir(
        ratings_dir, manifest, method=METHOD,
        widths={"char": int(n_char), "word": int(n_total - n_char)},
    )
    k_char, k_word = branch_budgets(cfg.k, cfg.char_share, n_char, n_total - n_char)
    report: dict[str, Any] = {
        "method": METHOD,
        "k_requested": int(cfg.k),
        "char_share": float(cfg.char_share),
        "seed": int(cfg.seed),
        "target": TARGET_NAME,
        "formula": FORMULA,
        "ratings_dir": os.path.abspath(ratings_dir),
        "tie_policy": "competitor ties -> lowest class index; score ties -> lowest feature index; no RNG",
        "budgets": {"k_char": int(k_char), "k_word": int(k_word)},
        "n_features_total": int(n_total),
        "positive_filter": "max(0, a) > 0 only; negative/zero contribute to diagnostics only",
    }
    branches: dict[str, BranchSelection] = {}
    for branch, k_b, offset in (("char", k_char, 0), ("word", k_word, n_char)):
        width = n_char if branch == "char" else n_total - n_char
        entry = (manifest.get("branches") or {}).get(branch) or {}
        branch_arrays = arrays[branch]
        classes = branch_arrays["classes"]
        counts = branch_arrays["counts"]
        global_scores = branch_arrays["global_scores"]
        sel, rep = roundrobin_select_from_rank_files(
            os.path.join(ratings_dir, entry["rank_indices"]),
            os.path.join(ratings_dir, entry["rank_scores"]),
            counts, classes, k_b, width, global_scores,
            positive_only_fallback=True,
        )
        branches[branch] = sel
        report[branch] = {
            "n_selected": len(sel.indices),
            "n_classes": rep["n_classes"],
            "n_positive_per_class": rep["n_positive_per_class"],
            "n_fallback": rep["n_fallback"],
            "n_remainder_fill": rep["n_remainder_fill"],
        }
    sel_w = branches["word"]
    word = BranchSelection(
        indices=sel_w.indices + n_char if len(sel_w.indices) else sel_w.indices,
        scores=sel_w.scores,
        origin=sel_w.origin,
    )
    classes_labels = None
    if l3_labels is not None:
        present = arrays["char"]["classes"]
        classes_labels = [str(l3_labels[int(i)]) for i in present]
    log.info(
        "G×I-отбор: k=%d → char %d/%d (fallback %d, remainder %d), word %d/%d (fallback %d, remainder %d)",
        cfg.k, len(branches["char"].indices), k_char,
        report["char"]["n_fallback"], report["char"]["n_remainder_fill"],
        len(word.indices), k_word,
        report["word"]["n_fallback"], report["word"]["n_remainder_fill"],
    )
    return SelectionResult(
        method=METHOD,
        k_requested=int(cfg.k),
        char_share=float(cfg.char_share),
        seed=int(cfg.seed),
        char=branches["char"],
        word=word,
        classes=classes_labels,
        report=report,
        ratings_dir=ratings_dir,
        ratings_manifest=manifest,
    )


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# Атрибуты TfidfVectorizer, определяющие фактическое преобразование текста в
# значения TF-IDF. Включаются в fingerprint контракта целиком: изменение любого
# из них меняет матрицу при той же размерности и незаметно ломает совместимость
# checkpoint'а (ревью 2, находка 1).
_TRANSFORM_PARAM_ATTRS = (
    "analyzer",
    "lowercase",
    "strip_accents",
    "token_pattern",
    "tokenizer",
    "preprocessor",
    "ngram_range",
    "min_df",
    "max_df",
    "max_features",
    "binary",
    "sublinear_tf",
    "norm",
    "use_idf",
    "smooth_idf",
    "dtype",
)


def _canonical_param(value: Any) -> Any:
    """JSON-безопасное каноническое представление параметра преобразования.

    Кортежи → списки (равенство словарей fingerprint не должно зависеть от
    типа последовательности); вызываемые объекты — «module.qualname», чтобы
    отпечаток был детерминирован между процессами (repr лямбда и локальных
    функций содержит адрес памяти).
    """
    if isinstance(value, (list, tuple)):
        return [_canonical_param(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical_param(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if callable(value):
        module = getattr(value, "__module__", None) or ""
        qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
        if qualname and "<locals>" not in qualname and qualname != "<lambda>":
            return f"{module}.{qualname}"
        return f"{module}.<unnamed {type(value).__name__}>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _transform_params_fingerprint(vec: Any) -> dict[str, Any]:
    """Канонические значения параметров преобразования одного векторизатора."""
    return {
        attr: _canonical_param(getattr(vec, attr, None))
        for attr in _TRANSFORM_PARAM_ATTRS
        if hasattr(vec, attr)
    }


def feature_contract_fingerprint(builder: Any) -> dict[str, Any]:
    """Проверяемый отпечаток полного контракта преобразования TF-IDF.

    Фиксирует всё, что реально определяет совместимость checkpoint'а MLP с
    признаковым пространством: для каждой ветки (char/word) — отображение
    «термин → столбец» (порядок словаря), вектор IDF, выровненный по этим
    столбцам, и полный набор параметров преобразования (``binary``,
    ``sublinear_tf``, ``norm``, ``ngram_range``, токенизация/препроцессоры и
    пр., см. :data:`_TRANSFORM_PARAM_ATTRS`), а также хук предобработки
    ``builder.preprocess``. Векторизаторы с одинаковой размерностью, но другим
    порядком словаря, иными IDF или иными параметрами ``transform`` дают разный
    отпечаток: без этого checkpoint проходил проверку с матрицей, значения
    которой отличались от использованных при его обучении. Подмена содержимого
    parquet по тому же пути изменяет отпечаток сплита, а не этого контракта.
    Результат JSON-сериализуем и сравнивается по полному равенству.
    """
    import hashlib
    import struct

    fp: dict[str, Any] = {
        "n_features": int(builder.n_features),
        "n_char_features": int(builder.n_char_features),
        "n_word_features": int(builder.n_word_features),
        "preprocess": _canonical_param(getattr(builder, "preprocess", None)),
        "branches": {},
    }
    for name in ("char", "word"):
        vec = getattr(builder, name)
        vocab = dict(vec.vocabulary_)
        idf = np.ascontiguousarray(np.asarray(vec.idf_, dtype=np.float64))
        h = hashlib.sha256()
        for term in sorted(vocab):
            h.update(str(term).encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
            h.update(struct.pack("<q", int(vocab[term])))
        h.update(idf.tobytes())
        fp["branches"][name] = {
            "vocabulary_idf_sha256": h.hexdigest(),
            "vocabulary_size": len(vocab),
            "idf_len": int(idf.shape[0]),
            "transform_params": _transform_params_fingerprint(vec),
        }
    return fp


def checkpoint_fingerprint(art_dir: str) -> dict[str, Any]:
    """Отпечаток checkpoint MLP: веса (model.pt) и конфиг (config.json)."""
    fp: dict[str, Any] = {"artifact_dir": os.path.abspath(art_dir)}
    for name in ("model.pt", "config.json"):
        path = os.path.join(art_dir, name)
        if os.path.isfile(path):
            fp[name] = {"sha256": _sha256_file(path), "bytes": os.path.getsize(path)}
    maps_path = os.path.join(art_dir, "label_maps.json")
    if os.path.isfile(maps_path):
        fp["label_maps_sha256"] = _sha256_file(maps_path)
    return fp


def verify_attribution_checkpoint(
    art_dir: str,
    model: torch.nn.Module,
    artifact_config: dict,
    n_features_full: int,
    label_map_sha256: str | None,
    data_cfg: dict,
    seed: int,
    full_tfidf_fp: dict[str, Any] | None = None,
    train_split_sha256: str | None = None,
) -> dict[str, Any]:
    """Проверяет идентичность checkpoint'а атрибуции и эксперимента (D02, D08).

    Контролируются: входная размерность модели равна полному TF-IDF;
    fingerprint label maps checkpoint'а совпадает с экспериментным
    ``label_map_sha256`` (одинаковый состав L3 ⇒ согласованное кодирование);
    data-конфиг (путь/сплит/min_count/max_rows/seed) checkpoint'а совпадает с
    экспериментным — вместе с общим seed это фиксирует те же splits. Дополнительно
    (S04, ревью 1): проверяемый отпечаток полного контракта признаков
    (словари char/word + IDF, см. :func:`feature_contract_fingerprint`) и sha256
    фактического train-сплита, сохранённые при обучении checkpoint'а в
    ``attribution_provenance.json``. Ревью 2: туда же сохраняется sha256 файла
    весов ``model.pt`` и при загрузке сверяется с фактическим хешем файла —
    подмена совместимого по размерности checkpoint'а незаметно не проходит.
    Принцип fail closed: отсутствие или
    несовпадение сохранённых отпечатков (в том числе у checkpoint'ов, обученных
    до появления соответствующего поля provenance) — ``AttributionError``:
    молчаливая подмена источника признаков/данных/весов недопустима.
    Возвращает словарь выполненных проверок.
    """
    checks: dict[str, Any] = {}
    n_feat_model = int(getattr(model, "n_features", -1))
    checks["n_features_matches_full_tfidf"] = {
        "expected": int(n_features_full), "actual": n_feat_model,
        "ok": n_feat_model == int(n_features_full),
    }
    # config.json артефакта модели оборачивает RunConfig в {"experiment_run",
    # "config": <RunConfig>}: принимаем оба расположения.
    inner = artifact_config.get("config") if isinstance(artifact_config, dict) else None
    if not isinstance(inner, dict):
        inner = artifact_config if isinstance(artifact_config, dict) else {}
    cfg_data = inner.get("data")
    checks["data_config_matches_experiment"] = {
        "expected": data_cfg, "actual": cfg_data, "ok": cfg_data == data_cfg,
    }
    cfg_seed = inner.get("seed")
    checks["seed_matches_experiment"] = {
        "expected": int(seed), "actual": cfg_seed, "ok": cfg_seed == int(seed),
    }
    maps_sha = checkpoint_fingerprint(art_dir).get("label_maps_sha256")
    checks["label_maps_match_experiment"] = {
        "expected": label_map_sha256, "actual": maps_sha,
        "ok": bool(label_map_sha256) and maps_sha == label_map_sha256,
    }
    # Проверяемые отпечатки полного контракта признаков и фактического
    # train-сплита, сохранённые при обучении checkpoint'а.
    prov_path = os.path.join(art_dir, ATTRIBUTION_PROVENANCE_FILENAME)
    stored: dict[str, Any] | None = None
    if os.path.isfile(prov_path):
        try:
            with open(prov_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                stored = loaded
        except (OSError, json.JSONDecodeError):
            stored = None
    checks["attribution_provenance_present"] = {
        "expected": ATTRIBUTION_PROVENANCE_FILENAME,
        "actual": os.path.isfile(prov_path),
        "ok": stored is not None,
    }
    if stored is not None:
        stored_fp = stored.get("full_tfidf_fingerprint")
        checks["full_tfidf_fingerprint_match"] = {
            "expected": stored_fp, "actual": full_tfidf_fp,
            "ok": stored_fp is not None and stored_fp == full_tfidf_fp,
        }
        stored_train = stored.get("train_split_sha256")
        checks["train_split_match"] = {
            "expected": stored_train, "actual": train_split_sha256,
            "ok": bool(stored_train) and stored_train == train_split_sha256,
        }
        # Криптографическая связка provenance с самими весами checkpoint'а:
        # замена model.pt на совместимый по размерности файл (переобученная или
        # чужая модель) меняет sha256 и отклоняется, а не проходит по общей
        # размерности и словарю (ревью 2, находка 1).
        stored_weights = stored.get("model_weights_sha256")
        weights_path = os.path.join(art_dir, "model.pt")
        actual_weights = _sha256_file(weights_path) if os.path.isfile(weights_path) else None
        checks["model_weights_match"] = {
            "expected": stored_weights, "actual": actual_weights,
            "ok": bool(stored_weights) and stored_weights == actual_weights,
        }
    failed = {k: v for k, v in checks.items() if not v["ok"]}
    if failed:
        details = "; ".join(
            f"{k}: ожидалось {str(v['expected'])[:80]!r}, получено {str(v['actual'])[:80]!r}"
            for k, v in failed.items()
        )
        raise AttributionError(
            f"Checkpoint атрибуции {art_dir!r} не совместим с экспериментом ({details}): "
            "идентичность full TF-IDF (fingerprint словаря/IDF и параметров "
            "преобразования), весов model.pt (sha256), label maps и фактического "
            "train split не подтверждена, атрибуция остановлена"
        )
    return checks


def save_attribution_report(path: str, report: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return path
