"""
Immutable Dataclasses representing output artifacts produced by each pipeline stage.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class DataIngestionArtifact:
    raw_file_path: Path
    train_file_path: Path
    test_file_path: Path


@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool
    status_file_path: Path
    drift_report_file_path: Path
    eda_summary_file_path: Path
    drift_detected: bool


@dataclass(frozen=True)
class DataTransformationArtifact:
    preprocessor_object_file_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path


@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_file_path: Path
    best_model_name: str
    best_r2_score: float
    best_params: Dict[str, Any]
    training_report_path: Path


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    is_model_accepted: bool
    r2_score: float
    rmse: float
    mae: float
    adjusted_r2: float
    metric_file_path: Path
