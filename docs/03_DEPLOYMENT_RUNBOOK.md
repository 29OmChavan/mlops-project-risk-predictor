# Role Simulation: ML & DevOps Engineer — Deployment Runbook & On-Call Playbook

**Project**: Project 01 · End-to-End ML Pipeline (Medical Insurance & Financial Risk Predictor)  
**Owner Roles**: ML Engineer & DevOps/MLOps Engineer  
**Container Image**: `29omchavan/mlops-project-1-risk-predictor:latest`

---

## 1. Local & Container Reproduction Runbook (Zero to Production)

### 1.1 Run Locally via Python Virtual Environment
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies & editable package
pip install -r requirements.txt

# 3. Execute 5-stage training pipeline (logs to DagsHub MLflow if credentials set)
python main.py

# 4. Run unit & integration test suite
pytest tests/ -v

# 5. Launch FastAPI production server on port 8080
python app.py
```

### 1.2 Build & Run via Docker / Docker Compose
```bash
# Build image
docker build -t 29omchavan/mlops-project-1-risk-predictor:latest .

# Run container with port mapping and healthcheck
docker run -d -p 8080:8080 --name risk-predictor-api 29omchavan/mlops-project-1-risk-predictor:latest

# Verify health probe & prediction API
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 34, "sex": "male", "bmi": 31.5, "children": 2, "smoker": "yes", "region": "southeast"}'
```

---

## 2. AWS Cloud Deployment Architecture (ECR + EC2 Self-Hosted Runner)

Following Krish Naik's AWS CI/CD reference workflow:
1. **IAM User Setup**: Create IAM user with `AmazonEC2ContainerRegistryFullAccess` and `AmazonEC2FullAccess`.
2. **ECR Repository**: Create private repository `mlops-project-1-risk-predictor`.
3. **EC2 Instance**: Launch Ubuntu 22.04 LTS (`t2.medium` or `t3.small`), allow inbound TCP port `8080`, install Docker:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu && newgrp docker
   ```
4. **GitHub Self-Hosted Runner**: Register EC2 under Repository `Settings -> Actions -> Runners -> New self-hosted runner`.
5. **GitHub Actions Secrets**:
   - `DOCKER_USERNAME`: `29omchavan`
   - `DOCKER_PAT`: Docker Hub Personal Access Token
   - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ECR_LOGIN_URI`, `ECR_REPOSITORY_NAME`

---

## 3. Incident Post-Mortem & On-Call Playbook

### Scenario A: `CustomTransformer` Pickling / Module Path Mismatch During Inference
* **Symptom**: `AttributeError: Can't get attribute 'RiskDomainFeatureGenerator' on <module '__main__'>` when loading `preprocessor.pkl` inside `app.py`.
* **Root Cause**: Defining a custom scikit-learn `BaseEstimator` in `__main__` instead of an importable package module causes `joblib.load()` to fail when invoked from `uvicorn`.
* **Resolution**: `RiskDomainFeatureGenerator` is defined inside the installable package `src.risk_predictor.components.data_transformation` (`setup.py` with `-e .`), ensuring consistent module path resolution across training and serving.

### Scenario B: Data Drift Alert Triggered (`drift_detected: True` in `status.txt`)
* **Symptom**: Two-sample KS test reports $p < 0.05$ on incoming batch data.
* **Action**: Inspect `artifacts/data_validation/drift_report.yaml`, verify upstream data source changes, and trigger `POST /train` or re-run `python main.py` to recalibrate preprocessor scaling statistics and model weights.
