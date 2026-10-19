"""
CLI Entry Point to execute the End-to-End MLOps Training Pipeline.
Usage:
    python main.py
"""
from src.risk_predictor.logger import logger
from src.risk_predictor.pipeline.training_pipeline import TrainingPipeline


def main():
    logger.info("Starting End-to-End Risk Predictor Training Pipeline...")
    pipeline = TrainingPipeline()
    eval_artifact = pipeline.run_pipeline()
    logger.info(
        f"Training Pipeline Finished Successfully! "
        f"Test R2={eval_artifact.r2_score} | RMSE=${eval_artifact.rmse:,.2f} | "
        f"MAE=${eval_artifact.mae:,.2f} | Accepted={eval_artifact.is_model_accepted}"
    )


if __name__ == "__main__":
    main()
