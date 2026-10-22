import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from kkt_clf.config import ModelConfig, build_config
from kkt_clf.labels import LabelMaps
from kkt_clf.model import SparseLinear, SparseMLP, csr_to_batch, param_counts
from kkt_clf.reject import (
    EMPTY,
    LOW_CONF,
    MISMATCH,
    RejectRule,
    TauTargetWarning,
    apply_reject,
    mismatch_flags,
    select_tau,
)
from kkt_clf.train import MultiTaskLoss, compute_class_weights


def test_sparse_linear_equals_dense():
    torch.manual_seed(0)
    D = sp.random(7, 30, density=0.2, dtype=np.float32, random_state=0).toarray()
    D[3] = 0  # пустая строка
    X = sp.csr_matrix(D)
    lin = SparseLinear(30, 5)
    out = lin(csr_to_batch(X))
    dense = torch.from_numpy(X.toarray()) @ lin.bag.weight + lin.bias
    assert torch.allclose(out, dense, atol=1e-6)


def test_architecture_from_config():
    m = SparseMLP(100, (2, 3, 4), ModelConfig([32, 16, 8], [0.1, 0.2, 0.0], "gelu", "layernorm"))
    names = [r["layer"] for r in param_counts(m)]
    assert "trunk.2.0" in names and names[-1] == "TOTAL"
    assert param_counts(m)[0]["params"] + param_counts(m)[1]["params"] == 100 * 32 + 32
    with pytest.raises(ValueError):
        build_config("default", {"model": {"hidden": [64, 32], "dropout": [0.1]}})


def test_class_weights():
    y = np.array([0, 0, 0, 0, 1, 2, 2])
    assert compute_class_weights(y, 4, "none", 0.5) is None
    w = compute_class_weights(y, 3, "inv_freq", 1.0)
    raw = 1 / np.array([4, 1, 2])
    assert np.allclose(w, raw / raw.mean())
    assert np.isclose(w.mean(), 1.0)
    assert np.allclose(compute_class_weights(y, 3, "inv_freq", 0.0), 1.0)
    w4 = compute_class_weights(y, 4, "inv_freq", 0.5)
    assert w4[3] == 0 and np.isclose(w4[:3].mean(), 1.0)


def test_multitask_loss_weighted():
    torch.manual_seed(0)
    logits = [torch.randn(6, 3), torch.randn(6, 4), torch.randn(6, 5)]
    targets = [torch.randint(0, 3, (6,)), torch.randint(0, 4, (6,)), torch.randint(0, 5, (6,))]
    cw3 = np.array([1, 2, 0.5, 1, 0.5], dtype=np.float32)
    loss_fn = MultiTaskLoss({"l1": 0.2, "l2": 0.3, "l3": 1.0}, {"l1": None, "l2": None, "l3": cw3})
    total, _ = loss_fn(logits, targets)
    ce = torch.nn.functional.cross_entropy
    exp = (0.2 * ce(logits[0], targets[0]) + 0.3 * ce(logits[1], targets[1])
           + ce(logits[2], targets[2], weight=torch.from_numpy(cw3)))
    assert torch.allclose(total, exp)


def _maps():
    df = pd.DataFrame({"L1": ["A", "A", "B"], "L2": ["a1", "a2", "b1"], "L3": ["x", "y", "z"], "name": ["1", "2", "3"]})
    return LabelMaps.from_df(df)


def test_mismatch_flags():
    m = _maps()  # x→a1→A, y→a2→A, z→b1→B
    l3 = np.array([0, 1, 2, 2])
    l2 = np.array([0, 0, 2, 0])
    l1 = np.array([0, 0, 1, 0])
    mm2, mm1 = mismatch_flags(l3, l2, l1, m)
    assert mm2.tolist() == [False, True, False, True]
    assert mm1.tolist() == [False, False, False, True]


def test_apply_reject_rules():
    prob = np.array([0.9, 0.3, 0.9, 0.9, np.nan, 0.2])
    empty = np.array([False, False, False, False, True, False])
    mm2 = np.array([False, False, True, False, False, True])
    mm1 = np.array([False, False, False, True, False, False])
    rej, why = apply_reject(prob, empty, mm2, mm1, RejectRule(0.5))
    assert rej.tolist() == [False, True, False, False, True, True]
    assert why.tolist() == [None, LOW_CONF, None, None, EMPTY, LOW_CONF]
    rej, why = apply_reject(prob, empty, mm2, mm1, RejectRule(0.5, reject_on_mismatch_l2=True))
    assert why.tolist() == [None, LOW_CONF, MISMATCH, None, EMPTY, LOW_CONF]
    rej, why = apply_reject(prob, empty, mm2, mm1, RejectRule(0.5, True, True))
    assert why.tolist() == [None, LOW_CONF, MISMATCH, MISMATCH, EMPTY, LOW_CONF]
    # пустые признаки — отказ безусловный даже при τ=0
    rej, why = apply_reject(np.array([0.99]), np.array([True]), np.array([False]), np.array([False]), RejectRule(0.0))
    assert rej.tolist() == [True] and why.tolist() == [EMPTY]


def test_select_tau():
    t = pd.DataFrame({
        "tau": [0.1, 0.2, 0.3, 0.4, 0.5],
        "coverage": [1.0, 0.9, 0.8, 0.7, 0.6],
        "accuracy": [0.80, 0.91, 0.89, 0.93, 0.95],
        "macro_f1": [0.5] * 5,
    })
    assert select_tau(t, "fixed", 0.5, 0.9)["tau"] == 0.5
    s = select_tau(t, "target_precision", 0.5, 0.90)
    assert s["tau"] == 0.2 and s["warning"] is None   # минимальный τ, немонотонность
    with pytest.warns(TauTargetWarning):
        s = select_tau(t, "target_precision", 0.5, 0.99)
    assert s["tau"] == 0.5 and "недостижима" in s["warning"]
