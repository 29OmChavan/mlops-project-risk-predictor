# Changelog

All notable changes to **Project 01: End-to-End MLOps Risk Predictor** are documented in this file.

## [1.0.0] - 2026-10-19
### Added
- **Project Scaffolding**: Created modular package layout (`src/risk_predictor`) via `template.py` and `setup.py`.
- **Observability Layer**: Implemented dual-handler structured logger (`logger.py`) and traceback-aware `CustomException` (`exception.py`).
- **Stage 01 — Data Ingestion**: Added Kaggle API v1 downloader (`mirichoi0218/insurance`) with mirror URL fallback, deduplication, and smoker-stratified train/test split.
- **Stage 02 — Data Validation & EDA**: Added schema validation, domain constraint checks, automated `eda_summary.json`, and Kolmogorov-Smirnov (`ks_2samp`) covariate drift detection (`drift_report.yaml`).
- **Stage 03 — Data Transformation**: Implemented `RiskDomainFeatureGenerator` (`is_obese`, `obese_smoker`, `bmi_age_interaction`) and `ColumnTransformer` serialized to `preprocessor.pkl`.
- **Stage 04 — Model Trainer**: Implemented 5-fold `GridSearchCV` across 6 regression families (`GradientBoostingRegressor`, `XGBRegressor`, `RandomForestRegressor`, `DecisionTreeRegressor`, `Ridge`, `AdaBoostRegressor`) with live experiment tracking on DagsHub MLflow.
- **Stage 05 — Model Evaluation**: Added test set evaluation ($R^2$, Adjusted $R^2$, RMSE, MAE, MAPE) and MLflow Model Registry versioning (`InsuranceRiskPredictorModel`).
- **FastAPI Service & UI**: Built `/predict`, `/predict-form`, `/health`, `/metrics`, `/train`, and dark-mode UI (`templates/index.html`).
- **Containerization & CI/CD**: Added `Dockerfile`, `docker-compose.yml`, `pytest` suite (`tests/test_pipeline.py`), and GitHub Actions workflow (`.github/workflows/main.yaml`).
- **Documentation**: Added `docs/IMPLEMENTATION_GUIDE.md`, `docs/01_DATA_SPEC.md`, `docs/02_EXPERIMENT_LOG_AND_MODEL_CARD.md`, and `docs/03_DEPLOYMENT_RUNBOOK.md`.
