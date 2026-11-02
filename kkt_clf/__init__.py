"""Классификатор позиций кассовых чеков L1 → L2 → L3 (итерация 1, baseline)."""
from .config import (
    InferenceConfig,
    RunConfig,
    build_config,
    build_feature_config,
    default_config,
    ods_light_config,
)
from .inference import Predictor, run_inference
from .pipeline import TrainResult, run_training, setup_logging, verify_roundtrip
from .vectorizer import (
    LoadedVectorizer,
    VectorizerArtifactError,
    VectorizerError,
    fit_and_save_vectorizer,
    fit_vectorizer,
    load_vectorizer_artifact,
    load_name_corpus,
    save_vectorizer_artifact,
)

__all__ = [
    "InferenceConfig", "RunConfig", "build_config", "build_feature_config",
    "default_config", "ods_light_config",
    "Predictor", "run_inference", "TrainResult", "run_training", "setup_logging", "verify_roundtrip",
    "LoadedVectorizer", "VectorizerArtifactError", "VectorizerError",
    "fit_and_save_vectorizer", "fit_vectorizer", "load_vectorizer_artifact",
    "load_name_corpus", "save_vectorizer_artifact",
]
