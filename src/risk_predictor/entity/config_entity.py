"""
Immutable Dataclasses representing configuration inputs for each pipeline stage.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_dataset: str
    source_url: str
    raw_data_file: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float
    random_state: int


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    status_file: Path
    drift_report_file: Path
    eda_summary_file: Path
    all_schema: Dict[str, Any]
    constraints: Dict[str, Any]


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    numerical_columns: List[str]
    categorical_columns: List[str]
    target_column: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    transformed_train_path: Path
    transformed_test_path: Path
    trained_model_path: Path
    training_report_path: Path
    expected_r2_threshold: float
    model_params: Dict[str, Any]
    mlflow_uri: str
    experiment_name: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    transformed_test_path: Path
    model_path: Path
    preprocessor_path: Path
    metric_file_name: Path
    mlflow_uri: str
    experiment_name: str
    registered_model_name: str
    target_column: str
