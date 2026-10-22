import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kkt_clf import build_config, run_training  # noqa: E402
from kkt_clf.fixtures import build_all  # noqa: E402


@pytest.fixture(scope="session")
def fx(tmp_path_factory):
    """Все фикстуры, сгенерированные из ODS во временный каталог."""
    return build_all(str(tmp_path_factory.mktemp("fixtures")))


def small_config(fx, out_dir, **model):
    return build_config("ods_light", {
        "output_dir": str(out_dir),
        "data": {"path": fx["ods_small"], "min_count": 10},
        "features": {"char": {"max_features": 5000}, "word": {"max_features": 2000}},
        "model": {"hidden": [64], "dropout": [0.1], **model},
        "train": {"epochs": 8, "batch_size": 64, "lr": 3e-3, "patience": 2, "device": "cpu"},
    })


@pytest.fixture(scope="session")
def trained(fx, tmp_path_factory):
    cfg = small_config(fx, tmp_path_factory.mktemp("train"))
    return run_training(cfg)
