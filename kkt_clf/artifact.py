"""Сериализация артефакта модели (§4.8).

Артефакт — каталог::

    model.pt            state_dict модели
    model_spec.json     n_features, число классов, конфиг модели
    config.json         полный конфиг обучающего прогона
    features.joblib     FeatureBuilder: предобработка + оба векторизатора
    label_maps.json     маппинги label↔id и таблицы родителей
    reject.json         τ, правила отказа, информация о выборе τ
    class_weights.json  веса классов CE и частоты классов в train
    meta.json           версии, seed, дата, размеры выборок, отпечаток входа
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import sklearn
import torch

from .config import ModelConfig
from .features import FeatureBuilder
from .labels import LabelMaps
from .model import SparseMLP
from .reject import RejectRule

FILES = (
    "model.pt", "model_spec.json", "config.json", "features.joblib",
    "label_maps.json", "reject.json", "class_weights.json", "meta.json",
)


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def environment_versions() -> dict:
    import pandas
    import scipy

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "platform": platform.platform(),
    }


@dataclass
class Artifact:
    model: SparseMLP
    features: FeatureBuilder
    maps: LabelMaps
    rule: RejectRule
    config: dict
    reject_info: dict
    class_weights: dict
    meta: dict


def save_artifact(
    out_dir: str,
    model: SparseMLP,
    model_cfg: ModelConfig,
    features: FeatureBuilder,
    maps: LabelMaps,
    rule: RejectRule,
    reject_info: dict,
    class_weights: dict,
    config: dict,
    meta: dict,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out_dir, "model.pt"))
    _dump(
        {"n_features": model.n_features, "n_classes": list(model.n_classes),
         "model": {"hidden": model_cfg.hidden, "dropout": model_cfg.dropout,
                   "activation": model_cfg.activation, "norm": model_cfg.norm}},
        os.path.join(out_dir, "model_spec.json"),
    )
    _dump(config, os.path.join(out_dir, "config.json"))
    joblib.dump(features, os.path.join(out_dir, "features.joblib"))
    _dump(maps.to_dict(), os.path.join(out_dir, "label_maps.json"))
    _dump({**rule.to_dict(), "selection": reject_info}, os.path.join(out_dir, "reject.json"))
    _dump(class_weights, os.path.join(out_dir, "class_weights.json"))
    full_meta = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": environment_versions(),
        **meta,
    }
    _dump(full_meta, os.path.join(out_dir, "meta.json"))
    return out_dir


def load_artifact(path: str, device: torch.device | str = "cpu") -> Artifact:
    missing = [f for f in FILES if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(f"В артефакте {path} нет файлов: {missing}")
    meta = _load(os.path.join(path, "meta.json"))
    saved_skl = meta.get("versions", {}).get("sklearn")
    if saved_skl and saved_skl != sklearn.__version__:
        warnings.warn(f"Артефакт сохранён с sklearn {saved_skl}, загружается в {sklearn.__version__}")
    spec = _load(os.path.join(path, "model_spec.json"))
    mcfg = ModelConfig(**spec["model"])
    model = SparseMLP(spec["n_features"], tuple(spec["n_classes"]), mcfg)
    state = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    features = joblib.load(os.path.join(path, "features.joblib"))
    maps = LabelMaps.from_dict(_load(os.path.join(path, "label_maps.json")))
    rj = _load(os.path.join(path, "reject.json"))
    rule = RejectRule(rj["tau"], rj["reject_on_mismatch_l2"], rj["reject_on_mismatch_l1"])
    return Artifact(
        model=model,
        features=features,
        maps=maps,
        rule=rule,
        config=_load(os.path.join(path, "config.json")),
        reject_info=rj.get("selection", {}),
        class_weights=_load(os.path.join(path, "class_weights.json")),
        meta=meta,
    )
