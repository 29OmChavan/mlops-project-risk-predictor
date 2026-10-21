"""
FastAPI Production Multi-Country Clinical & Actuarial Risk Platform.
Provides:
- GET  /                  : Page 1 — Multi-Country Actuarial & Clinical Underwriting Studio
- GET  /analytics         : Page 2 — Country ML Models, Datasets & Leaderboard Explorer
- GET  /explainability    : Page 3 — Explainable AI (SHAP-Style Waterfall & Counterfactual Lab)
- GET  /document-vault    : Page 4 — Ephemeral Medical Document Parser & Zero-Retention Privacy Audit Log
- GET  /health            : Kubernetes / Docker / Load-Balancer readiness & liveness probe
- GET  /metrics           : Latest model evaluation metrics & country leaderboards
- GET  /countries         : Supported countries, Public/Private insurance tiers & disease catalog
- POST /predict           : Backward-compatible & enhanced REST API endpoint
- POST /predict-country   : Full Multi-Country + Height/Weight + Explainability + AI Agent endpoint
- POST /analyze-document  : Ephemeral medical document parser with mandatory Terms & immediate deletion
- POST /train             : Triggers training pipeline execution
"""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

from src.risk_predictor.logger import logger
from src.risk_predictor.utils.common import load_json
from src.risk_predictor.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.risk_predictor.pipeline.training_pipeline import TrainingPipeline
from src.risk_predictor.components.country_risk_engine import (
    COUNTRY_CONFIGS,
    SUPPORTED_DISEASES,
    CountryRiskEngine,
    compute_bmi_from_height_weight,
)
from src.risk_predictor.components.explainability_agent import (
    EphemeralDocumentProcessor,
    UnderwritingExplanationAgent,
)

app = FastAPI(
    title="Production MLOps: Multi-Country Health Insurance & Clinical Risk Platform",
    description=(
        "End-to-End Multi-Country Actuarial & Clinical Risk Predictor featuring Height/Weight BMI calculation, "
        "Public vs. Private Insurance Tiers, Chronic Disease Comorbidities, Ephemeral Zero-Retention Medical "
        "Document Parsing, SHAP-Style Additive Explainability, and an AI Underwriting Explanation Agent."
    ),
    version="2.0.0",
)

templates = Jinja2Templates(directory="templates")
country_engine = CountryRiskEngine()
doc_processor = EphemeralDocumentProcessor()
ai_agent = UnderwritingExplanationAgent()


class RiskPredictionRequest(BaseModel):
    """
    Pydantic request schema supporting both legacy (bmi) and enhanced
    (height_cm + weight_kg, country, insurance_type, alcohol, diseases) requests.
    """

    age: int = Field(..., ge=18, le=100, description="Primary beneficiary age (18–100)")
    sex: Literal["female", "male"] = Field(..., description="Biological sex ('female' or 'male')")
    bmi: Optional[float] = Field(
        default=None, ge=10.0, le=70.0, description="Body Mass Index (computed automatically if height_cm & weight_kg provided)"
    )
    height_cm: Optional[float] = Field(
        default=None, ge=100.0, le=240.0, description="Height in centimeters (100–240 cm)"
    )
    weight_kg: Optional[float] = Field(
        default=None, ge=25.0, le=250.0, description="Weight in kilograms (25–250 kg)"
    )
    children: int = Field(default=0, ge=0, le=10, description="Number of dependents covered (0–10)")
    smoker: Literal["yes", "no", "occasional"] = Field(..., description="Smoking status ('no', 'occasional', 'yes')")
    region: str = Field(default="southeast", description="Residential region within selected country")
    country: Literal["USA", "India", "UK", "Germany"] = Field(
        default="USA", description="Target Healthcare Country Ecosystem"
    )
    insurance_type: Optional[str] = Field(
        default=None, description="Public, Hybrid, or Private insurance plan tier"
    )
    alcohol_consumption: Literal["none", "social_moderate", "regular", "heavy"] = Field(
        default="none", description="Alcohol consumption frequency"
    )
    activity_level: Literal["sedentary", "moderate", "active"] = Field(
        default="moderate", description="Physical activity level"
    )
    past_medical_history: Literal[
        "none", "prior_hospitalization", "major_surgery", "family_hereditary_history"
    ] = Field(default="none", description="Past surgical or medical hospitalization history")
    diseases: List[str] = Field(
        default_factory=list,
        description="List of pre-existing or chronic disease keys",
    )

    @model_validator(mode="after")
    def ensure_bmi_or_anthropometrics(self) -> "RiskPredictionRequest":
        if self.bmi is None and (self.height_cm is None or self.weight_kg is None):
            raise ValueError("Provide either (height_cm and weight_kg) or bmi.")
        if self.height_cm is not None and self.weight_kg is not None:
            info = compute_bmi_from_height_weight(self.height_cm, self.weight_kg)
            self.bmi = info["bmi"]
        return self


@app.get("/health", tags=["Operations"])
async def health_check():
    """Health check endpoint verifying model and preprocessor artifacts exist."""
    model_exists = Path("artifacts/model_trainer/model.pkl").exists()
    preprocessor_exists = Path("artifacts/data_transformation/preprocessor.pkl").exists()
    country_models = list(Path("artifacts/country_models").glob("*_bundle.pkl"))
    return {
        "status": "healthy" if (model_exists and preprocessor_exists) else "degraded",
        "artifacts_ready": bool(model_exists and preprocessor_exists),
        "trained_country_models": [p.stem.replace("_bundle", "").upper() for p in country_models],
        "version": "2.0.0",
    }


@app.get("/countries", tags=["Multi-Country Metadata"])
async def get_countries_metadata():
    """Returns supported countries, Public vs. Private insurance tiers, and chronic disease catalog."""
    return {
        "countries": COUNTRY_CONFIGS,
        "supported_diseases": SUPPORTED_DISEASES,
        "trained_country_summaries": country_engine.get_all_country_summaries(),
    }


@app.get("/metrics", tags=["Operations"])
async def get_model_metrics():
    """Returns the baseline model evaluation metrics and multi-country leaderboards."""
    metrics_path = Path("artifacts/model_evaluation/metrics.json")
    leaderboard_path = Path("artifacts/model_trainer/model_comparison.json")
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Run /train or `python main.py` first.",
        )
    result: Dict[str, Any] = {
        "evaluation_metrics": load_json(metrics_path),
        "country_models": country_engine.get_all_country_summaries(),
    }
    if leaderboard_path.exists():
        result["training_comparison"] = load_json(leaderboard_path)
    return result


@app.post("/predict", tags=["Inference"])
async def predict_risk(payload: RiskPredictionRequest):
    """
    REST endpoint supporting both standard 6-feature US baseline inference and
    multi-country clinical + explainability + AI Agent inference.
    """
    try:
        # Use baseline US model when no extended multi-country/disease parameters are requested
        is_legacy_simple = (
            payload.country == "USA"
            and payload.height_cm is None
            and payload.weight_kg is None
            and payload.insurance_type is None
            and payload.alcohol_consumption == "none"
            and payload.past_medical_history == "none"
            and len(payload.diseases) == 0
            and payload.smoker in ("yes", "no")
            and payload.region in ("southwest", "southeast", "northwest", "northeast")
        )
        if is_legacy_simple:
            custom_data = CustomData(
                age=payload.age,
                sex=payload.sex,
                bmi=float(payload.bmi or 25.0),
                children=payload.children,
                smoker=payload.smoker,
                region=payload.region,
            )
            pred_df = custom_data.get_data_as_dataframe()
            pipeline = PredictionPipeline()
            base_res = pipeline.predict(pred_df)
            ext_bundle = country_engine.predict_with_explainability(payload.model_dump())
            agent_feedback = ai_agent.generate_feedback(ext_bundle)
            return {
                **base_res,
                "country": "USA",
                "bmi": float(payload.bmi or 25.0),
                "explainability": ext_bundle["explainability"],
                "ai_agent_feedback": agent_feedback,
            }

        ext_bundle = country_engine.predict_with_explainability(payload.model_dump())
        agent_feedback = ai_agent.generate_feedback(ext_bundle)
        return {
            **ext_bundle,
            "ai_agent_feedback": agent_feedback,
        }
    except Exception as e:
        logger.error(f"Prediction API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-country", tags=["Inference"])
async def predict_country_detailed(payload: RiskPredictionRequest):
    """Full Multi-Country Clinical, Public/Private Insurance, Explainability & AI Agent endpoint."""
    try:
        ext_bundle = country_engine.predict_with_explainability(payload.model_dump())
        agent_feedback = ai_agent.generate_feedback(ext_bundle)
        return {
            **ext_bundle,
            "ai_agent_feedback": agent_feedback,
        }
    except Exception as e:
        logger.error(f"Country prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-document", tags=["Ephemeral Document Vault"])
async def analyze_medical_document(
    medical_doc: UploadFile = File(...),
    accept_terms: bool = Form(False),
):
    """
    Processes an uploaded medical document in-memory, extracts clinical risk indicators,
    and immediately zero-fills & permanently deletes the file.
    """
    if not accept_terms:
        raise HTTPException(
            status_code=400,
            detail="You must accept the Terms & Conditions acknowledging immediate document deletion.",
        )
    try:
        raw_bytes = await medical_doc.read()
        receipt = doc_processor.process_and_purge(
            filename=medical_doc.filename or "uploaded_report.txt",
            raw_bytes=raw_bytes,
            accept_terms=accept_terms,
        )
        return receipt
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_common_ui_context(selected_country: str = "USA") -> Dict[str, Any]:
    metrics_path = Path("artifacts/model_evaluation/metrics.json")
    metrics = load_json(metrics_path) if metrics_path.exists() else None
    country_summaries = country_engine.get_all_country_summaries()
    return {
        "countries": COUNTRY_CONFIGS,
        "selected_country": selected_country if selected_country in COUNTRY_CONFIGS else "USA",
        "supported_diseases": SUPPORTED_DISEASES,
        "metrics": metrics,
        "country_summaries": country_summaries,
        "audit_log": doc_processor.get_audit_log(limit=12),
    }


@app.get("/", response_class=HTMLResponse, tags=["UI Pages"])
async def home_page(request: Request, country: str = "USA"):
    """Page 1: Interactive Multi-Country Clinical & Actuarial Underwriting Studio."""
    ctx = _build_common_ui_context(selected_country=country)
    ctx.update(
        {
            "active_page": "studio",
            "result": None,
            "agent_feedback": None,
            "doc_receipt": None,
            "error_message": None,
            "form_data": None,
        }
    )
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.post("/predict-form", response_class=HTMLResponse, tags=["UI Pages"])
async def predict_from_form(
    request: Request,
    country: str = Form("USA"),
    age: int = Form(34),
    sex: str = Form("male"),
    height_cm: float = Form(174.0),
    weight_kg: float = Form(82.0),
    children: int = Form(1),
    smoker: str = Form("no"),
    alcohol_consumption: str = Form("none"),
    activity_level: str = Form("moderate"),
    past_medical_history: str = Form("none"),
    region: str = Form("southeast"),
    insurance_type: str = Form("Private - PPO Comprehensive"),
    diseases: List[str] = Form(default=[]),
    accept_terms: Optional[str] = Form(default=None),
    medical_doc: Optional[UploadFile] = File(default=None),
):
    """Handles multi-country clinical form + optional ephemeral medical document upload."""
    ctx = _build_common_ui_context(selected_country=country)
    ctx["active_page"] = "studio"

    try:
        doc_receipt = None
        merged_diseases = list(diseases) if diseases else []

        if medical_doc and medical_doc.filename and medical_doc.filename.strip() != "":
            terms_ok = str(accept_terms).lower() in ("true", "on", "1", "yes")
            if not terms_ok:
                ctx.update(
                    {
                        "result": None,
                        "agent_feedback": None,
                        "doc_receipt": None,
                        "error_message": (
                            "Please check 'Accept Terms & Conditions' to upload a medical document. "
                            "All uploaded documents are permanently deleted immediately after analysis."
                        ),
                        "form_data": None,
                    }
                )
                return templates.TemplateResponse(request=request, name="index.html", context=ctx)

            raw_bytes = await medical_doc.read()
            if len(raw_bytes) > 0:
                doc_receipt = doc_processor.process_and_purge(
                    filename=medical_doc.filename,
                    raw_bytes=raw_bytes,
                    accept_terms=True,
                )
                for det_d in doc_receipt.get("detected_diseases", []):
                    if det_d not in merged_diseases:
                        merged_diseases.append(det_d)

        payload = {
            "country": country,
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "children": children,
            "smoker": smoker,
            "alcohol_consumption": alcohol_consumption,
            "activity_level": activity_level,
            "past_medical_history": past_medical_history,
            "region": region,
            "insurance_type": insurance_type,
            "diseases": merged_diseases,
        }

        result = country_engine.predict_with_explainability(payload)
        agent_feedback = ai_agent.generate_feedback(result, document_receipt=doc_receipt)

        ctx.update(
            {
                "result": result,
                "agent_feedback": agent_feedback,
                "doc_receipt": doc_receipt,
                "error_message": None,
                "form_data": payload,
                "audit_log": doc_processor.get_audit_log(limit=12),
            }
        )
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)
    except Exception as e:
        logger.error(f"Form prediction error: {e}")
        ctx.update(
            {
                "result": None,
                "agent_feedback": None,
                "doc_receipt": None,
                "error_message": f"Error processing assessment: {str(e)}",
                "form_data": None,
            }
        )
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/analytics", response_class=HTMLResponse, tags=["UI Pages"])
async def analytics_page(request: Request, country: str = "USA"):
    """Page 2: Multi-Country ML Models, Actuarial Leaderboards & Healthcare System Comparison."""
    ctx = _build_common_ui_context(selected_country=country)
    ctx["active_page"] = "analytics"
    return templates.TemplateResponse(request=request, name="analytics.html", context=ctx)


@app.get("/explainability", response_class=HTMLResponse, tags=["UI Pages"])
async def explainability_page(request: Request, country: str = "USA"):
    """Page 3: Explainable AI (SHAP-Style Marginal Coalition Attribution & Counterfactual Lab)."""
    ctx = _build_common_ui_context(selected_country=country)
    sample_payload = {
        "country": country if country in COUNTRY_CONFIGS else "USA",
        "age": 46,
        "sex": "male",
        "height_cm": 175.0,
        "weight_kg": 96.0,
        "children": 2,
        "smoker": "yes",
        "alcohol_consumption": "regular",
        "activity_level": "sedentary",
        "past_medical_history": "prior_hospitalization",
        "diseases": ["hypertension", "diabetes_type_2"],
    }
    sample_result = country_engine.predict_with_explainability(sample_payload)
    sample_agent = ai_agent.generate_feedback(sample_result)
    ctx.update(
        {
            "active_page": "explainability",
            "sample_result": sample_result,
            "sample_agent": sample_agent,
        }
    )
    return templates.TemplateResponse(request=request, name="explainability.html", context=ctx)


@app.get("/document-vault", response_class=HTMLResponse, tags=["UI Pages"])
async def document_vault_page(request: Request):
    """Page 4: Ephemeral Medical Document Parser & Zero-Retention Privacy Audit Log."""
    ctx = _build_common_ui_context(selected_country="USA")
    ctx.update({"active_page": "vault", "vault_receipt": None, "vault_error": None})
    return templates.TemplateResponse(request=request, name="document_vault.html", context=ctx)


@app.post("/document-vault", response_class=HTMLResponse, tags=["UI Pages"])
async def document_vault_upload(
    request: Request,
    accept_terms: Optional[str] = Form(default=None),
    medical_doc: Optional[UploadFile] = File(default=None),
):
    """Processes a medical document in the standalone vault view and proves immediate deletion."""
    ctx = _build_common_ui_context(selected_country="USA")
    ctx["active_page"] = "vault"

    terms_ok = str(accept_terms).lower() in ("true", "on", "1", "yes")
    if not terms_ok:
        ctx.update(
            {
                "vault_receipt": None,
                "vault_error": "You must accept the Terms & Conditions before uploading a medical document.",
            }
        )
        return templates.TemplateResponse(request=request, name="document_vault.html", context=ctx)

    if not medical_doc or not medical_doc.filename:
        ctx.update({"vault_receipt": None, "vault_error": "Please select a medical document file to scan."})
        return templates.TemplateResponse(request=request, name="document_vault.html", context=ctx)

    raw_bytes = await medical_doc.read()
    receipt = doc_processor.process_and_purge(
        filename=medical_doc.filename,
        raw_bytes=raw_bytes,
        accept_terms=True,
    )
    ctx.update(
        {
            "vault_receipt": receipt,
            "vault_error": None,
            "audit_log": doc_processor.get_audit_log(limit=12),
        }
    )
    return templates.TemplateResponse(request=request, name="document_vault.html", context=ctx)


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
