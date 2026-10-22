"""Классификатор позиций кассовых чеков L1 → L2 → L3 (итерация 1, baseline)."""
from .config import InferenceConfig, RunConfig, build_config, default_config, ods_light_config
from .inference import Predictor, run_inference
from .pipeline import TrainResult, run_training, setup_logging, verify_roundtrip

__all__ = [
    "InferenceConfig", "RunConfig", "build_config", "default_config", "ods_light_config",
    "Predictor", "run_inference", "TrainResult", "run_training", "setup_logging", "verify_roundtrip",
]
