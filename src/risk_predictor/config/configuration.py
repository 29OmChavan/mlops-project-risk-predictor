"""
ConfigurationManager: Central orchestrator that reads YAML files and returns typed Config entities.
"""
from pathlib import Path
from src.risk_predictor.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)
from src.risk_predictor.utils.common import read_yaml, create_directories
from src.risk_predictor.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    """Loads config.yaml, params.yaml, and schema.yaml and constructs stage configs."""

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        create_directories([config["root_dir"]])

        return DataIngestionConfig(
            root_dir=Path(config["root_dir"]),
            kaggle_dataset=config["kaggle_dataset"],
            source_url=config["source_url"],
            raw_data_file=Path(config["raw_data_file"]),
            train_data_path=Path(config["train_data_path"]),
            test_data_path=Path(config["test_data_path"]),
            test_size=float(config["test_size"]),
            random_state=int(config["random_state"]),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config["data_validation"]
        create_directories([config["root_dir"]])

        return DataValidationConfig(
            root_dir=Path(config["root_dir"]),
            train_data_path=Path(config["train_data_path"]),
            test_data_path=Path(config["test_data_path"]),
            status_file=Path(config["status_file"]),
            drift_report_file=Path(config["drift_report_file"]),
            eda_summary_file=Path(config["eda_summary_file"]),
            all_schema=self.schema.get("columns", {}),
            constraints=self.schema.get("constraints", {}),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config["data_transformation"]
        create_directories([config["root_dir"]])

        return DataTransformationConfig(
            root_dir=Path(config["root_dir"]),
            train_data_path=Path(config["train_data_path"]),
            test_data_path=Path(config["test_data_path"]),
            preprocessor_path=Path(config["preprocessor_path"]),
            transformed_train_path=Path(config["transformed_train_path"]),
            transformed_test_path=Path(config["transformed_test_path"]),
            numerical_columns=self.schema.get("numerical_columns", []),
            categorical_columns=self.schema.get("categorical_columns", []),
            target_column=self.schema["target_column"]["name"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config["model_trainer"]
        eval_config = self.config["model_evaluation"]
        create_directories([config["root_dir"]])

        return ModelTrainerConfig(
            root_dir=Path(config["root_dir"]),
            transformed_train_path=Path(config["transformed_train_path"]),
            transformed_test_path=Path(config["transformed_test_path"]),
            trained_model_path=Path(config["trained_model_path"]),
            training_report_path=Path(config["training_report_path"]),
            expected_r2_threshold=float(config["expected_r2_threshold"]),
            model_params=self.params.get("models", {}),
            mlflow_uri=eval_config["mlflow_uri"],
            experiment_name=eval_config["experiment_name"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config["model_evaluation"]
        create_directories([config["root_dir"]])

        return ModelEvaluationConfig(
            root_dir=Path(config["root_dir"]),
            test_data_path=Path(config["test_data_path"]),
            transformed_test_path=Path(config["transformed_test_path"]),
            model_path=Path(config["model_path"]),
            preprocessor_path=Path(config["preprocessor_path"]),
            metric_file_name=Path(config["metric_file_name"]),
            mlflow_uri=config["mlflow_uri"],
            experiment_name=config["experiment_name"],
            registered_model_name=config["registered_model_name"],
            target_column=self.schema["target_column"]["name"],
        )
