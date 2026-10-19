"""
Stage 04: Model Trainer + Experiment Tracking Component
Trains 6 regression families with GridSearchCV, logs every candidate model's hyperparameters
and metrics to MLflow (DagsHub remote or local fallback), and saves the champion model.
"""
import os
import sys
from typing import Dict, Any
import numpy as np
import mlflow
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException
from src.risk_predictor.utils.common import save_object, save_json
from src.risk_predictor.entity.config_entity import ModelTrainerConfig
from src.risk_predictor.entity.artifact_entity import ModelTrainerArtifact


def configure_mlflow(mlflow_uri: str, experiment_name: str) -> str:
    """
    Configures MLflow tracking URI (uses DagsHub remote if MLFLOW_TRACKING_USERNAME/PASSWORD
    are set and reachable, otherwise falls back gracefully to local mlruns).
    """
    use_remote = bool(
        os.environ.get("MLFLOW_TRACKING_USERNAME")
        and os.environ.get("MLFLOW_TRACKING_PASSWORD")
        and os.environ.get("DISABLE_REMOTE_MLFLOW", "0") != "1"
    )
    active_uri = mlflow_uri if use_remote else "file:./mlruns"
    try:
        mlflow.set_tracking_uri(active_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow configured at URI: {active_uri} (Experiment: {experiment_name})")
    except Exception as e:
        logger.warning(f"Remote MLflow connection failed ({e}). Falling back to local ./mlruns")
        active_uri = "file:./mlruns"
        mlflow.set_tracking_uri(active_uri)
        mlflow.set_experiment(experiment_name)
    return active_uri


class ModelTrainer:
    """Trains multiple models with hyperparameter tuning and logs all runs to MLflow."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Entered ModelTrainer.initiate_model_trainer()")
        try:
            train_arr = np.load(self.config.transformed_train_path)
            test_arr = np.load(self.config.transformed_test_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "RandomForestRegressor": RandomForestRegressor(random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(random_state=42, verbosity=0),
                "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
                "Ridge": Ridge(random_state=42),
                "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            }

            configure_mlflow(self.config.mlflow_uri, self.config.experiment_name)

            leaderboard: Dict[str, Dict[str, Any]] = {}
            best_model_name = ""
            best_model_obj = None
            best_r2 = float("-inf")
            best_params: Dict[str, Any] = {}

            for model_name, estimator in models.items():
                param_grid = self.config.model_params.get(model_name, {})
                logger.info(f"Tuning {model_name} with GridSearchCV: {param_grid}")

                gs = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    cv=5,
                    scoring="r2",
                    n_jobs=-1,
                )
                gs.fit(X_train, y_train)

                tuned_model = gs.best_estimator_
                y_train_pred = tuned_model.predict(X_train)
                y_test_pred = tuned_model.predict(X_test)

                train_r2 = float(r2_score(y_train, y_train_pred))
                test_r2 = float(r2_score(y_test, y_test_pred))
                test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
                test_mae = float(mean_absolute_error(y_test, y_test_pred))
                cv_r2 = float(gs.best_score_)

                # Log candidate model run to MLflow
                try:
                    with mlflow.start_run(run_name=f"candidate_{model_name}"):
                        mlflow.set_tag("model_family", model_name)
                        mlflow.set_tag("stage", "model_training_selection")
                        mlflow.log_params(gs.best_params_)
                        mlflow.log_metrics(
                            {
                                "cv_r2": round(cv_r2, 4),
                                "train_r2": round(train_r2, 4),
                                "test_r2": round(test_r2, 4),
                                "test_rmse": round(test_rmse, 4),
                                "test_mae": round(test_mae, 4),
                            }
                        )
                except Exception as mlf_err:
                    logger.warning(f"MLflow run logging warning for {model_name}: {mlf_err}")

                leaderboard[model_name] = {
                    "cv_r2": round(cv_r2, 4),
                    "train_r2": round(train_r2, 4),
                    "test_r2": round(test_r2, 4),
                    "test_rmse": round(test_rmse, 2),
                    "test_mae": round(test_mae, 2),
                    "best_params": gs.best_params_,
                }

                logger.info(
                    f"[{model_name}] CV R2={cv_r2:.4f} | Train R2={train_r2:.4f} | "
                    f"Test R2={test_r2:.4f} | RMSE={test_rmse:.2f}"
                )

                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model_name = model_name
                    best_model_obj = tuned_model
                    best_params = gs.best_params_

            if best_r2 < self.config.expected_r2_threshold:
                raise ValueError(
                    f"No model met the expected R2 threshold of {self.config.expected_r2_threshold}. "
                    f"Best was {best_model_name} with R2={best_r2:.4f}"
                )

            logger.info(
                f"Champion Model Selected: {best_model_name} (Test R2={best_r2:.4f}, Params={best_params})"
            )

            save_object(file_path=self.config.trained_model_path, obj=best_model_obj)

            report_payload = {
                "champion_model": best_model_name,
                "champion_test_r2": round(best_r2, 4),
                "champion_params": best_params,
                "leaderboard": leaderboard,
            }
            save_json(self.config.training_report_path, report_payload)

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_path,
                best_model_name=best_model_name,
                best_r2_score=best_r2,
                best_params=best_params,
                training_report_path=self.config.training_report_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
