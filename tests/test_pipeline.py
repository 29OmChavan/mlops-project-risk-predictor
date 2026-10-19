"""
Automated Unit & Integration Test Suite for Project 01 MLOps Pipeline.
Tests:
1. Configuration loading & YAML schema integrity
2. Domain feature engineering transformer (`RiskDomainFeatureGenerator`)
3. Full 5-stage training pipeline execution & artifact verification
4. PredictionPipeline inference accuracy & risk tier bounds
5. FastAPI REST endpoints (`/health`, `/metrics`, `/predict`, validation rejection)
"""
import os
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from src.risk_predictor.config.configuration import ConfigurationManager
from src.risk_predictor.components.data_transformation import RiskDomainFeatureGenerator
from src.risk_predictor.pipeline.training_pipeline import TrainingPipeline
from src.risk_predictor.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from app import app

client = TestClient(app)


def test_configuration_manager_loads_valid_configs():
    cm = ConfigurationManager()
    ing = cm.get_data_ingestion_config()
    val = cm.get_data_validation_config()
    trans = cm.get_data_transformation_config()
    trainer = cm.get_model_trainer_config()

    assert ing.test_size == 0.2
    assert "charges" in val.all_schema
    assert trans.target_column == "charges"
    assert trainer.expected_r2_threshold >= 0.70


def test_domain_feature_generator():
    df = pd.DataFrame(
        {
            "age": [25, 45],
            "sex": ["female", "male"],
            "bmi": [22.5, 33.0],
            "children": [0, 2],
            "smoker": ["no", "yes"],
            "region": ["southwest", "southeast"],
        }
    )
    gen = RiskDomainFeatureGenerator()
    out = gen.fit_transform(df)
    assert "is_obese" in out.columns
    assert "obese_smoker" in out.columns
    assert "bmi_age_interaction" in out.columns
    assert out.loc[0, "is_obese"] == 0.0
    assert out.loc[1, "is_obese"] == 1.0
    assert out.loc[1, "obese_smoker"] == 1.0


def test_end_to_end_pipeline_and_artifacts():
    if not Path("artifacts/model_trainer/model.pkl").exists():
        os.environ["DISABLE_REMOTE_MLFLOW"] = "1"
        pipeline = TrainingPipeline()
        eval_artifact = pipeline.run_pipeline()
        assert eval_artifact.is_model_accepted is True
        assert eval_artifact.r2_score >= 0.80

    assert Path("artifacts/data_ingestion/train.csv").exists()
    assert Path("artifacts/data_validation/drift_report.yaml").exists()
    assert Path("artifacts/data_transformation/preprocessor.pkl").exists()
    assert Path("artifacts/model_trainer/model.pkl").exists()
    assert Path("artifacts/model_evaluation/metrics.json").exists()


def test_prediction_pipeline_low_vs_high_risk():
    low_risk = CustomData(
        age=22, sex="female", bmi=21.5, children=0, smoker="no", region="northwest"
    ).get_data_as_dataframe()
    high_risk = CustomData(
        age=52, sex="male", bmi=35.2, children=2, smoker="yes", region="southeast"
    ).get_data_as_dataframe()

    pred_pipe = PredictionPipeline()
    res_low = pred_pipe.predict(low_risk)
    res_high = pred_pipe.predict(high_risk)

    assert res_low["predicted_charges_usd"] > 0
    assert res_high["predicted_charges_usd"] > res_low["predicted_charges_usd"] * 2.5
    assert res_high["risk_tier"] == "High Risk"


def test_fastapi_endpoints():
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["evaluation_metrics"]["r2_score"] >= 0.80

    pred_resp = client.post(
        "/predict",
        json={
            "age": 34,
            "sex": "male",
            "bmi": 31.5,
            "children": 2,
            "smoker": "yes",
            "region": "southeast",
        },
    )
    assert pred_resp.status_code == 200
    body = pred_resp.json()
    assert body["predicted_charges_usd"] > 20000
    assert body["risk_tier"] == "High Risk"

    # Invalid input (age < 18) should return 422 Unprocessable Entity
    invalid_resp = client.post(
        "/predict",
        json={
            "age": 12,
            "sex": "male",
            "bmi": 25.0,
            "children": 0,
            "smoker": "no",
            "region": "southeast",
        },
    )
    assert invalid_resp.status_code == 422
