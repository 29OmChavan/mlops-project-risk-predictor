"""
End-to-End Training Pipeline Orchestrator.
Executes all 5 stages sequentially:
Data Ingestion -> Data Validation -> Data Transformation -> Model Trainer -> Model Evaluation.
"""
import sys
from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException
from src.risk_predictor.config.configuration import ConfigurationManager
from src.risk_predictor.components.data_ingestion import DataIngestion
from src.risk_predictor.components.data_validation import DataValidation
from src.risk_predictor.components.data_transformation import DataTransformation
from src.risk_predictor.components.model_trainer import ModelTrainer
from src.risk_predictor.components.model_evaluation import ModelEvaluation
from src.risk_predictor.entity.artifact_entity import ModelEvaluationArtifact


class TrainingPipeline:
    """Runs the complete 5-stage MLOps training workflow."""

    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run_pipeline(self) -> ModelEvaluationArtifact:
        try:
            logger.info("=" * 70)
            logger.info(">>>>> STAGE 01: DATA INGESTION STARTED <<<<<")
            ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=ingestion_config)
            ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"Stage 01 Artifact: {ingestion_artifact}")
            logger.info(">>>>> STAGE 01: DATA INGESTION COMPLETED <<<<<\n")

            logger.info(">>>>> STAGE 02: DATA VALIDATION & EDA STARTED <<<<<")
            validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=validation_config)
            validation_artifact = data_validation.initiate_data_validation()
            if not validation_artifact.validation_status:
                raise ValueError(
                    "Data Validation failed! Check artifacts/data_validation/eda_summary.json"
                )
            logger.info(f"Stage 02 Artifact: {validation_artifact}")
            logger.info(">>>>> STAGE 02: DATA VALIDATION & EDA COMPLETED <<<<<\n")

            logger.info(">>>>> STAGE 03: DATA TRANSFORMATION STARTED <<<<<")
            transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=transformation_config)
            transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info(f"Stage 03 Artifact: {transformation_artifact}")
            logger.info(">>>>> STAGE 03: DATA TRANSFORMATION COMPLETED <<<<<\n")

            logger.info(">>>>> STAGE 04: MODEL TRAINER + MLFLOW TRACKING STARTED <<<<<")
            trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=trainer_config)
            trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info(f"Stage 04 Artifact: {trainer_artifact}")
            logger.info(">>>>> STAGE 04: MODEL TRAINER COMPLETED <<<<<\n")

            logger.info(">>>>> STAGE 05: MODEL EVALUATION & REGISTRY STARTED <<<<<")
            eval_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=eval_config)
            eval_artifact = model_evaluation.log_into_mlflow()
            logger.info(f"Stage 05 Artifact: {eval_artifact}")
            logger.info(">>>>> STAGE 05: MODEL EVALUATION COMPLETED <<<<<")
            logger.info("=" * 70)

            return eval_artifact

        except Exception as e:
            raise CustomException(e, sys)
