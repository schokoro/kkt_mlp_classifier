"""Веса классов, multi-task лосс, цикл обучения, предсказание батчами (§4.5)."""
from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .config import HEADS, LossConfig, TrainConfig
from .metrics import macro_f1
from .model import SparseMLP, csr_to_batch

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def compute_class_weights(y: np.ndarray, n_classes: int, mode: str, alpha: float) -> np.ndarray | None:
    """``none`` → None; ``inv_freq`` → 1/freq^alpha, нормировка на средний вес 1.

    Классы без примеров в train получают вес 0 (в CE они не встречаются).
    """
    if mode == "none":
        return None
    if mode != "inv_freq":
        raise ValueError(mode)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(n_classes, dtype=np.float64)
    w[present] = 1.0 / counts[present] ** alpha
    w[present] /= w[present].mean()
    return w.astype(np.float32)


class MultiTaskLoss(nn.Module):
    """w1·CE(L1) + w2·CE(L2) + w3·CE(L3)."""

    def __init__(self, head_weights: dict[str, float], class_weights: dict[str, np.ndarray | None]):
        super().__init__()
        self.head_weights = dict(head_weights)
        self.ce = nn.ModuleDict()
        for h in HEADS:
            cw = class_weights.get(h)
            self.ce[h] = nn.CrossEntropyLoss(weight=None if cw is None else torch.as_tensor(cw))

    def forward(self, logits, targets) -> tuple[torch.Tensor, dict[str, float]]:
        total = 0.0
        parts = {}
        for h, lg, t in zip(HEADS, logits, targets):
            l = self.ce[h](lg, t)
            parts[h] = float(l.detach())
            total = total + self.head_weights[h] * l
        return total, parts


@dataclass
class Predictions:
    """Компактные выходы трёх голов (без хранения полного softmax)."""

    l3_top1: np.ndarray
    l3_prob: np.ndarray
    l3_topk: np.ndarray
    l3_topk_prob: np.ndarray
    l2_head: np.ndarray
    l2_prob: np.ndarray
    l1_head: np.ndarray
    l1_prob: np.ndarray

    @classmethod
    def concat(cls, parts: list["Predictions"]) -> "Predictions":
        return cls(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in cls.__dataclass_fields__})


def _make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer in ("adamw", "adam"):
        cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        # fused-реализация в разы быстрее на 100M+ параметрах первого слоя (CPU и CUDA)
        try:
            return cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
        except (RuntimeError, TypeError, ValueError):
            return cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)


def _targets(ys: tuple[np.ndarray, np.ndarray, np.ndarray], rows: np.ndarray, device) -> list[torch.Tensor]:
    return [torch.from_numpy(y[rows]).to(device) for y in ys]


@torch.no_grad()
def predict(
    model: SparseMLP,
    X: sp.csr_matrix,
    device: torch.device,
    batch_size: int = 8192,
    top_k: int = 1,
    ys: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    loss_fn: MultiTaskLoss | None = None,
) -> tuple[Predictions, float | None]:
    """Прогон модели батчами. Если переданы ``ys`` и ``loss_fn`` — ещё и средний лосс."""
    model.eval()
    k = max(1, min(top_k, model.n_classes[2]))
    parts: list[Predictions] = []
    loss_sum, n_sum = 0.0, 0
    for start in range(0, X.shape[0], batch_size):
        rows = np.arange(start, min(start + batch_size, X.shape[0]))
        batch = csr_to_batch(X[start : rows[-1] + 1]).to(device)
        lg1, lg2, lg3 = model(batch)
        if ys is not None and loss_fn is not None:
            loss, _ = loss_fn((lg1, lg2, lg3), _targets(ys, rows, device))
            loss_sum += float(loss) * len(rows)
            n_sum += len(rows)
        p3 = torch.softmax(lg3.float(), dim=1)
        p2 = torch.softmax(lg2.float(), dim=1)
        p1 = torch.softmax(lg1.float(), dim=1)
        tk_prob, tk = p3.topk(k, dim=1)
        pr2, i2 = p2.max(dim=1)
        pr1, i1 = p1.max(dim=1)
        parts.append(
            Predictions(
                l3_top1=tk[:, 0].cpu().numpy(),
                l3_prob=tk_prob[:, 0].cpu().numpy(),
                l3_topk=tk.cpu().numpy(),
                l3_topk_prob=tk_prob.cpu().numpy(),
                l2_head=i2.cpu().numpy(),
                l2_prob=pr2.cpu().numpy(),
                l1_head=i1.cpu().numpy(),
                l1_prob=pr1.cpu().numpy(),
            )
        )
    if not parts:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float32)
        return Predictions(empty_i, empty_f, np.zeros((0, k), np.int64), np.zeros((0, k), np.float32),
                           empty_i, empty_f, empty_i, empty_f), None
    return Predictions.concat(parts), (loss_sum / n_sum if n_sum else None)


def train_model(
    model: SparseMLP,
    X_train: sp.csr_matrix,
    y_train: tuple[np.ndarray, np.ndarray, np.ndarray],
    X_val: sp.csr_matrix,
    y_val: tuple[np.ndarray, np.ndarray, np.ndarray],
    loss_cfg: LossConfig,
    train_cfg: TrainConfig,
    class_weights: dict[str, np.ndarray | None],
    device: torch.device,
    seed: int,
) -> tuple[list[dict], dict]:
    """Цикл обучения с early stopping по val. Возвращает историю по эпохам и инфо о лучшей эпохе.

    Строки train с пустым вектором признаков в обучение не попадают.
    Лучшие веса загружаются в ``model`` по окончании.
    """
    if train_cfg.early_stopping_metric not in ("l3_macro_f1", "l3_accuracy", "val_loss"):
        raise ValueError("train.early_stopping_metric: l3_macro_f1 | l3_accuracy | val_loss")
    model.to(device)
    loss_fn = MultiTaskLoss(loss_cfg.head_weights, class_weights).to(device)
    opt = _make_optimizer(model, train_cfg)
    rng = np.random.default_rng(seed)
    train_rows = np.flatnonzero(np.diff(X_train.indptr) > 0)
    n_classes_l3 = model.n_classes[2]
    higher_better = train_cfg.early_stopping_metric != "val_loss"
    best = {"epoch": -1, "score": -np.inf if higher_better else np.inf}
    best_state = None
    history: list[dict] = []
    bad_epochs = 0
    bs = train_cfg.batch_size
    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(train_rows)
        loss_sum, n_sum = 0.0, 0
        for start in range(0, len(perm), bs):
            rows = np.sort(perm[start : start + bs])
            if len(rows) < 2 and any(isinstance(m, nn.BatchNorm1d) for m in model.modules()):
                continue
            batch = csr_to_batch(X_train[rows]).to(device)
            logits = model(batch)
            loss, _ = loss_fn(logits, _targets(y_train, rows, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(rows)
            n_sum += len(rows)
        train_time = time.perf_counter() - t0
        preds, val_loss = predict(model, X_val, device, train_cfg.eval_batch_size, 1, y_val, loss_fn)
        yv3 = y_val[2]
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_sum, 1),
            "val_loss": val_loss,
            "val_l3_accuracy": float((preds.l3_top1 == yv3).mean()),
            "val_l3_macro_f1": macro_f1(yv3, preds.l3_top1, n_classes_l3),
            "val_l2_head_accuracy": float((preds.l2_head == y_val[1]).mean()),
            "val_l1_head_accuracy": float((preds.l1_head == y_val[0]).mean()),
            "epoch_time_s": train_time,
        }
        history.append(rec)
        score = {
            "l3_macro_f1": rec["val_l3_macro_f1"],
            "l3_accuracy": rec["val_l3_accuracy"],
            "val_loss": rec["val_loss"],
        }[train_cfg.early_stopping_metric]
        improved = score > best["score"] if higher_better else score < best["score"]
        log.info(
            "эпоха %d: train_loss=%.4f val_loss=%.4f val_acc_L3=%.4f val_mF1_L3=%.4f (%.1fs)%s",
            epoch, rec["train_loss"], rec["val_loss"], rec["val_l3_accuracy"],
            rec["val_l3_macro_f1"], train_time, " *" if improved else "",
        )
        if improved:
            best = {"epoch": epoch, "score": float(score)}
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg.patience:
                log.info("early stopping: %d эпох без улучшения", bad_epochs)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    best["metric"] = train_cfg.early_stopping_metric
    best["n_train_rows_used"] = int(len(train_rows))
    return history, best
