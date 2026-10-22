"""Конфигурация пайплайна: dataclass'ы, пресеты и сборка из словаря.

Пресет ``default`` рассчитан на обучение на ~6 млн строк (24 ГБ RAM, один GPU),
пресет ``ods_light`` — на быстрые прогоны на ODS (~18 тыс. строк) на CPU.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

ACTIVATIONS = ("relu", "gelu", "silu", "tanh")
NORMS = (None, "layernorm", "batchnorm")
OPTIMIZERS = ("adamw", "adam", "sgd")
CLASS_WEIGHT_MODES = ("none", "inv_freq")
TAU_MODES = ("fixed", "target_precision")
HEADS = ("l1", "l2", "l3")


@dataclass
class DataConfig:
    path: str = "data.parquet"
    max_rows: int | None = None
    min_count: int = 10
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    n_duplicate_examples: int = 10


@dataclass
class BranchConfig:
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int | float = 1
    max_features: int | None = None
    sublinear_tf: bool = True


@dataclass
class FeatureConfig:
    char: BranchConfig = field(default_factory=lambda: BranchConfig((2, 5), 5, 300_000, True))
    word: BranchConfig = field(default_factory=lambda: BranchConfig((1, 1), 3, 100_000, True))
    vectorizer_fit_rows: int | None = 1_500_000
    transform_chunk_rows: int = 200_000


@dataclass
class ModelConfig:
    hidden: list[int] = field(default_factory=lambda: [512, 256])
    dropout: list[float] = field(default_factory=lambda: [0.3, 0.3])
    activation: str = "relu"
    norm: str | None = None


@dataclass
class LossConfig:
    head_weights: dict[str, float] = field(default_factory=lambda: {"l1": 0.3, "l2": 0.3, "l3": 1.0})
    class_weight_mode: dict[str, str] = field(
        default_factory=lambda: {"l1": "none", "l2": "none", "l3": "none"}
    )
    class_weight_alpha: float = 0.5


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 2048
    eval_batch_size: int = 8192
    epochs: int = 10
    early_stopping_metric: str = "l3_macro_f1"
    patience: int = 2
    device: str = "auto"
    num_threads: int | None = None


@dataclass
class RejectConfig:
    tau_mode: str = "fixed"
    tau: float = 0.5
    target_precision: float = 0.95
    tau_grid: list[float] = field(default_factory=lambda: [round(i * 0.01, 2) for i in range(100)])
    reject_on_mismatch_l2: bool = False
    reject_on_mismatch_l1: bool = False


@dataclass
class ReportConfig:
    top_n_errors: int = 50
    noisy_support: int = 2


@dataclass
class RunConfig:
    """Конфиг обучающего прогона."""

    output_dir: str = "outputs/train"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reject: RejectConfig = field(default_factory=RejectConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def validate(self) -> None:
        validate_config(self)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        return _merge(cls(), d)


@dataclass
class InferenceConfig:
    """Конфиг ноутбука инференса."""

    artifact_dir: str = "outputs/train/artifact"
    input_path: str = "input.parquet"
    output_path: str = "outputs/inference/predictions.parquet"
    report_dir: str | None = "outputs/inference/reports"
    batch_size: int = 50_000
    tau: float | None = None
    reject_on_mismatch_l2: bool | None = None
    reject_on_mismatch_l1: bool | None = None
    top_k: int = 3
    device: str = "auto"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InferenceConfig":
        return _merge(cls(), d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(obj: Any, overrides: dict[str, Any]) -> Any:
    """Рекурсивно накладывает словарь на dataclass; неизвестные ключи — ошибка."""
    obj = copy.deepcopy(obj)
    names = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in overrides.items():
        if key not in names:
            raise KeyError(f"{type(obj).__name__}: неизвестный параметр конфига {key!r}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            setattr(obj, key, _merge(current, value))
        elif isinstance(current, dict) and isinstance(value, dict):
            setattr(obj, key, {**current, **copy.deepcopy(value)})
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, copy.deepcopy(value))
    return obj


def _normalize(cfg: RunConfig) -> None:
    """Приводит типы после JSON (списки → кортежи)."""
    cfg.data.split = tuple(cfg.data.split)
    cfg.features.char.ngram_range = tuple(cfg.features.char.ngram_range)
    cfg.features.word.ngram_range = tuple(cfg.features.word.ngram_range)
    cfg.model.hidden = list(cfg.model.hidden)
    cfg.model.dropout = list(cfg.model.dropout)


def validate_config(cfg: RunConfig) -> None:
    _normalize(cfg)
    m = cfg.model
    if len(m.hidden) == 0:
        raise ValueError("model.hidden: нужен хотя бы один скрытый слой")
    if len(m.hidden) != len(m.dropout):
        raise ValueError(
            f"model.hidden и model.dropout должны быть одной длины: {len(m.hidden)} != {len(m.dropout)}"
        )
    if any(h <= 0 for h in m.hidden):
        raise ValueError("model.hidden: размерности должны быть > 0")
    if any(not 0.0 <= p < 1.0 for p in m.dropout):
        raise ValueError("model.dropout: значения в [0, 1)")
    if m.activation not in ACTIVATIONS:
        raise ValueError(f"model.activation: одно из {ACTIVATIONS}")
    if m.norm not in NORMS:
        raise ValueError(f"model.norm: одно из {NORMS}")
    s = cfg.data.split
    if len(s) != 3 or any(x <= 0 for x in s) or abs(sum(s) - 1.0) > 1e-6:
        raise ValueError("data.split: три положительные доли с суммой 1")
    if cfg.data.min_count < 2:
        raise ValueError("data.min_count: не меньше 2 (нужно для стратифицированного сплита)")
    if cfg.data.max_rows is not None and cfg.data.max_rows <= 0:
        raise ValueError("data.max_rows: None или > 0")
    for h in HEADS:
        if h not in cfg.loss.head_weights:
            raise ValueError(f"loss.head_weights: нет веса для {h}")
        if cfg.loss.class_weight_mode.get(h) not in CLASS_WEIGHT_MODES:
            raise ValueError(f"loss.class_weight_mode[{h}]: одно из {CLASS_WEIGHT_MODES}")
    if cfg.train.optimizer not in OPTIMIZERS:
        raise ValueError(f"train.optimizer: одно из {OPTIMIZERS}")
    if cfg.reject.tau_mode not in TAU_MODES:
        raise ValueError(f"reject.tau_mode: одно из {TAU_MODES}")
    if not 0.0 <= cfg.reject.tau <= 1.0:
        raise ValueError("reject.tau: в [0, 1]")


def default_config() -> RunConfig:
    """Дефолты под ~6 млн строк (см. DECISION_LOG)."""
    return RunConfig()


def ods_light_config() -> RunConfig:
    """Лёгкий пресет для ODS (~18 тыс. строк) на CPU."""
    return RunConfig.from_dict(
        {
            "features": {
                "char": {"ngram_range": [2, 5], "min_df": 2, "max_features": 50_000},
                "word": {"ngram_range": [1, 1], "min_df": 1, "max_features": 20_000},
                "vectorizer_fit_rows": None,
                "transform_chunk_rows": 5_000,
            },
            "model": {"hidden": [256], "dropout": [0.3]},
            "train": {"batch_size": 256, "eval_batch_size": 4096, "epochs": 15, "patience": 3},
        }
    )


PRESETS = {"default": default_config, "ods_light": ods_light_config}


def build_config(preset: str = "default", overrides: dict[str, Any] | None = None) -> RunConfig:
    if preset not in PRESETS:
        raise KeyError(f"Неизвестный пресет {preset!r}, есть: {list(PRESETS)}")
    cfg = _merge(PRESETS[preset](), overrides or {})
    validate_config(cfg)
    return cfg
