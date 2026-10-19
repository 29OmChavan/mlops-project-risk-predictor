"""
FastAPI Production Prediction Service for Medical Insurance & Financial Risk Predictor.
Provides:
- GET  /          : Interactive Web UI Dashboard
- GET  /health    : Kubernetes / Docker / Load-Balancer readiness & liveness probe
- GET  /metrics   : Latest model evaluation metrics & leaderboard
- POST /predict   : JSON REST API endpoint with strict Pydantic validation
- POST /train     : Triggers full training pipeline execution
"""
from pathlib import Path
from typing import Literal
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.risk_predictor.logger import logger
from src.risk_predictor.utils.common import load_json
from src.risk_predictor.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.risk_predictor.pipeline.training_pipeline import TrainingPipeline

app = FastAPI(
    title="Production MLOps: Medical Insurance & Financial Risk Predictor API",
    description=(
        "End-to-End MLOps service with automated Data Validation, sklearn Feature Engineering, "
        "MLflow Experiment Tracking on DagsHub, and FastAPI inference."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")


class RiskPredictionRequest(BaseModel):
    """Pydantic request schema enforcing domain constraints at the API boundary."""

    age: int = Field(..., ge=18, le=100, description="Primary beneficiary age (18–100)")
    sex: Literal["female", "male"] = Field(..., description="Biological sex ('female' or 'male')")
    bmi: float = Field(..., ge=10.0, le=70.0, description="Body Mass Index (kg/m^2, 10.0–70.0)")
    children: int = Field(..., ge=0, le=10, description="Number of dependents covered (0–10)")
    smoker: Literal["yes", "no"] = Field(..., description="Smoking status ('yes' or 'no')")
    region: Literal["southwest", "southeast", "northwest", "northeast"] = Field(
        ..., description="US residential region"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 34,
                "sex": "male",
                "bmi": 31.5,
                "children": 2,
                "smoker": "yes",
                "region": "southeast",
            }
        }
    }


class RiskPredictionResponse(BaseModel):
    """Structured API prediction response."""

    predicted_charges_usd: float
    risk_tier: str
    model_used: str


@app.get("/health", tags=["Operations"])
async def health_check():
    """Health check endpoint verifying model and preprocessor artifacts exist."""
    model_exists = Path("artifacts/model_trainer/model.pkl").exists()
    preprocessor_exists = Path("artifacts/data_transformation/preprocessor.pkl").exists()
    return {
        "status": "healthy" if (model_exists and preprocessor_exists) else "degraded",
        "artifacts_ready": bool(model_exists and preprocessor_exists),
        "version": "1.0.0",
    }


@app.get("/metrics", tags=["Operations"])
async def get_model_metrics():
    """Returns the latest model evaluation metrics and training leaderboard."""
    metrics_path = Path("artifacts/model_evaluation/metrics.json")
    leaderboard_path = Path("artifacts/model_trainer/model_comparison.json")
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Run /train or `python main.py` first.",
        )
    result = {"evaluation_metrics": load_json(metrics_path)}
    if leaderboard_path.exists():
        result["training_comparison"] = load_json(leaderboard_path)
    return result


@app.post("/predict", response_model=RiskPredictionResponse, tags=["Inference"])
async def predict_risk(payload: RiskPredictionRequest):
    """JSON REST endpoint for low-latency risk cost prediction."""
    try:
        custom_data = CustomData(
            age=payload.age,
            sex=payload.sex,
            bmi=payload.bmi,
            children=payload.children,
            smoker=payload.smoker,
            region=payload.region,
        )
        pred_df = custom_data.get_data_as_dataframe()
        pipeline = PredictionPipeline()
        result = pipeline.predict(pred_df)
        return RiskPredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def home_page(request: Request):
    """Renders interactive web UI for risk prediction."""
    metrics_path = Path("artifacts/model_evaluation/metrics.json")
    metrics = load_json(metrics_path) if metrics_path.exists() else None
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None, "metrics": metrics, "form_data": None},
    )


@app.post("/predict-form", response_class=HTMLResponse, tags=["UI"])
async def predict_from_form(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...),
):
    """Handles HTML form submission and renders prediction result."""
    try:
        custom_data = CustomData(
            age=age,
            sex=sex,
            bmi=bmi,
            children=children,
            smoker=smoker,
            region=region,
        )
        pred_df = custom_data.get_data_as_dataframe()
        pipeline = PredictionPipeline()
        result = pipeline.predict(pred_df)
        metrics_path = Path("artifacts/model_evaluation/metrics.json")
        metrics = load_json(metrics_path) if metrics_path.exists() else None
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": result,
                "metrics": metrics,
                "form_data": {
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "children": children,
                    "smoker": smoker,
                    "region": region,
                },
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", tags=["Pipeline"])
async def trigger_training():
    """Triggers the end-to-end 5-stage training pipeline."""
    try:
        pipeline = TrainingPipeline()
        eval_artifact = pipeline.run_pipeline()
        return {
            "status": "Training Completed Successfully",
            "r2_score": eval_artifact.r2_score,
            "adjusted_r2": eval_artifact.adjusted_r2,
            "rmse": eval_artifact.rmse,
            "mae": eval_artifact.mae,
            "is_model_accepted": eval_artifact.is_model_accepted,
        }
    except Exception as e:
        logger.error(f"Training endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
