"""
Stage 05: Model Evaluation & Model Registry Component
Computes comprehensive regression metrics (R2, Adjusted R2, RMSE, MAE, MAPE),
extracts feature importances, saves metrics.json, and registers the champion model in MLflow.
"""
import sys
from urllib.parse import urlparse
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException
from src.risk_predictor.utils.common import load_object, save_json
from src.risk_predictor.components.model_trainer import configure_mlflow
from src.risk_predictor.entity.config_entity import ModelEvaluationConfig
from src.risk_predictor.entity.artifact_entity import ModelEvaluationArtifact


class ModelEvaluation:
    """Evaluates the champion model and versions it in MLflow Model Registry."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def eval_metrics(actual: np.ndarray, pred: np.ndarray, n_features: int):
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae = float(mean_absolute_error(actual, pred))
        r2 = float(r2_score(actual, pred))
        n = len(actual)
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / max(n - n_features - 1, 1))
        mape = float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1e-6))) * 100.0)
        return rmse, mae, r2, adj_r2, mape

    def log_into_mlflow(self) -> ModelEvaluationArtifact:
        logger.info("Entered ModelEvaluation.log_into_mlflow()")
        try:
            test_arr = np.load(self.config.transformed_test_path)
            model = load_object(self.config.model_path)

            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            predicted_qualities = model.predict(X_test)

            rmse, mae, r2, adj_r2, mape = self.eval_metrics(
                y_test, predicted_qualities, X_test.shape[1]
            )

            scores = {
                "model_class": type(model).__name__,
                "r2_score": round(r2, 4),
                "adjusted_r2": round(adj_r2, 4),
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "mape_percent": round(mape, 2),
                "test_samples": int(len(y_test)),
                "num_transformed_features": int(X_test.shape[1]),
            }

            if hasattr(model, "feature_importances_"):
                scores["feature_importances"] = [
                    round(float(x), 4) for x in model.feature_importances_
                ]

            save_json(path=self.config.metric_file_name, data=scores)

            active_uri = configure_mlflow(
                self.config.mlflow_uri, self.config.experiment_name
            )
            tracking_url_type_store = urlparse(active_uri).scheme

            try:
                with mlflow.start_run(run_name=f"champion_evaluation_{type(model).__name__}"):
                    mlflow.set_tag("stage", "champion_evaluation")
                    mlflow.set_tag("champion_model", type(model).__name__)
                    mlflow.log_metric("r2_score", round(r2, 4))
                    mlflow.log_metric("adjusted_r2", round(adj_r2, 4))
                    mlflow.log_metric("rmse", round(rmse, 4))
                    mlflow.log_metric("mae", round(mae, 4))
                    mlflow.log_metric("mape_percent", round(mape, 4))
                    mlflow.log_artifact(str(self.config.metric_file_name))

                    if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(
                            model,
                            "model",
                            registered_model_name=self.config.registered_model_name,
                            serialization_format="cloudpickle",
                        )
                    else:
                        mlflow.sklearn.log_model(
                            model,
                            "model",
                            serialization_format="cloudpickle",
                        )
                logger.info("Successfully logged champion evaluation & model to MLflow.")
            except Exception as mlf_err:
                logger.warning(f"MLflow model registry step warning: {mlf_err}")

            return ModelEvaluationArtifact(
                is_model_accepted=bool(r2 >= 0.75),
                r2_score=round(r2, 4),
                rmse=round(rmse, 2),
                mae=round(mae, 2),
                adjusted_r2=round(adj_r2, 4),
                metric_file_path=self.config.metric_file_name,
            )

        except Exception as e:
            raise CustomException(e, sys)
