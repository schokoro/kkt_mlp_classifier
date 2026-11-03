"""Классификатор позиций кассовых чеков L1 → L2 → L3 (итерация 1, baseline)."""
from .config import (
    DataConfig,
    InferenceConfig,
    RunConfig,
    build_config,
    build_feature_config,
    default_config,
    ods_light_config,
)
from .experiment import (
    ExperimentConfig,
    ExperimentError,
    SelectionPlanConfig,
    build_report,
    planned_runs,
    run_experiment,
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
    "DataConfig",
    "InferenceConfig", "RunConfig", "build_config", "build_feature_config",
    "default_config", "ods_light_config",
    "ExperimentConfig", "SelectionPlanConfig", "ExperimentError",
    "build_report", "planned_runs", "run_experiment",
    "Predictor", "run_inference", "TrainResult", "run_training", "setup_logging", "verify_roundtrip",
    "LoadedVectorizer", "VectorizerArtifactError", "VectorizerError",
    "fit_and_save_vectorizer", "fit_vectorizer", "load_vectorizer_artifact",
    "load_name_corpus", "save_vectorizer_artifact",
]
