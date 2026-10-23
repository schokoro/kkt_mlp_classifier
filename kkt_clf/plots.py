"""Графики по уже посчитанным данным (§4.7)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curves(history: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="val loss")
    axes[0].set_xlabel("эпоха")
    axes[0].set_title("Лосс")
    axes[0].legend()
    for col, label in (("val_l3_accuracy", "val accuracy L3"), ("val_l3_macro_f1", "val macro-F1 L3"),
                       ("val_l2_head_accuracy", "val accuracy L2 (голова)"),
                       ("val_l1_head_accuracy", "val accuracy L1 (голова)")):
        axes[1].plot(history["epoch"], history[col], marker="o", label=label)
    axes[1].set_xlabel("эпоха")
    axes[1].set_title("Метрики на val")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_tau_curve(table: pd.DataFrame, tau: float | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["tau"], table["coverage"], label="покрытие")
    ax.plot(table["tau"], table["accuracy"], label="accuracy принятых")
    ax.plot(table["tau"], table["macro_f1"], label="macro-F1 принятых")
    if tau is not None:
        ax.axvline(tau, color="grey", ls="--", label=f"τ = {tau:.2f}")
    ax.set_xlabel("τ")
    ax.set_title("Покрытие / качество по τ (val)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_f1_vs_freq(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(np.maximum(df["train_count"], 1), df["f1"], s=10, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("число примеров класса в train (log)")
    ax.set_ylabel("F1 на test")
    ax.set_title("F1 классов L3 vs частота")
    fig.tight_layout()
    return fig


def plot_lr(trace: pd.DataFrame):
    """lr по шагам и val loss на подвыборке val (для plateau)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trace["step"], trace["lr"], color="C0", label="lr")
    ax.set_yscale("log")
    ax.set_xlabel("шаг")
    ax.set_ylabel("lr", color="C0")
    if trace["val_loss_subset"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(trace["step"], trace["val_loss_subset"], color="C1", alpha=0.7, label="val loss (подвыборка)")
        ax2.set_ylabel("val loss (подвыборка)", color="C1")
    ax.set_title("LR scheduler")
    fig.tight_layout()
    return fig
