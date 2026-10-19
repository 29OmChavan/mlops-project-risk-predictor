# Role Simulation: ML Researcher — Experiment Log & Model Card

**Project**: Project 01 · End-to-End ML Pipeline (Medical Insurance & Financial Risk Predictor)  
**Owner Role**: ML Researcher  
**MLflow Tracking Server**: [`https://dagshub.com/29omchavan/mlops-project-1-risk-predictor.mlflow`](https://dagshub.com/29omchavan/mlops-project-1-risk-predictor.mlflow)

---

## 1. Research Hypotheses & Feature Engineering

### Hypothesis 1: Synergistic Effect of Obesity ($\text{BMI} \ge 30$) and Smoking
* **Observation during EDA**:
  * Non-smokers have a relatively linear cost progression with age regardless of BMI (mean $\approx \$8,434$).
  * Smokers with $\text{BMI} < 30$ average $\approx \$21,363$, whereas smokers with $\text{BMI} \ge 30$ jump non-linearly to $\approx \$41,558$.
* **Engineered Features (`RiskDomainFeatureGenerator`)**:
  1. `is_obese` $= \mathbb{I}(\text{bmi} \ge 30.0)$
  2. `obese_smoker` $= \mathbb{I}(\text{bmi} \ge 30.0) \times \mathbb{I}(\text{smoker} == \text{'yes'})$
  3. `bmi_age_interaction` $= \frac{\text{bmi} \times \text{age}}{100.0}$
* **Result**: Adding `RiskDomainFeatureGenerator` inside the scikit-learn `Pipeline` boosted linear model ($R^2$ of `Ridge`) from `0.74` to `>0.84` and reduced tree ensemble RMSE by ~7%.

---

## 2. Multi-Model Experiment Log (5-Fold GridSearchCV + MLflow)

Six model families were evaluated across hyperparameter grids defined in `params.yaml`:

| Model Family | Best Hyperparameters (`GridSearchCV`) | CV $R^2$ (5-Fold) | Train $R^2$ | Test $R^2$ | Test RMSE (USD) | Test MAE (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GradientBoostingRegressor (Champion)** | `lr: 0.05, max_depth: 3, n_estimators: 100` | `0.8422` | `0.8797` | **`0.9194`** | **`$3,408.05`** | **`$2,086.07`** |
| **XGBRegressor** | `lr: 0.05, max_depth: 3, n_estimators: 100` | `0.8436` | `0.8742` | `0.9186` | `$3,425.61` | `$2,120.54` |
| **Ridge Regression** | `alpha: 10.0` | `0.8437` | `0.8498` | `0.9177` | `$3,445.29` | `$2,163.42` |
| **RandomForestRegressor** | `max_depth: 6, min_samples_split: 2, n_estimators: 200` | `0.8367` | `0.8884` | `0.9154` | `$3,491.39` | `$2,121.84` |
| **DecisionTreeRegressor** | `max_depth: 4, min_samples_leaf: 5` | `0.8310` | `0.8524` | `0.9087` | `$3,628.82` | `$2,335.75` |
| **AdaBoostRegressor** | `lr: 0.05, n_estimators: 50` | `0.8161` | `0.8296` | `0.8763` | `$4,222.23` | `$3,364.26` |

### Key Feature Importance Discovery
In the champion `GradientBoostingRegressor`, our engineered interaction feature `obese_smoker` (index 4 in the transformed feature matrix) accounts for **`72.62%` (`0.7262`)** of total impurity reduction, followed by `age` (`8.38%`), `smoker` (`12.50%` combined one-hot), and `bmi_age_interaction` (`3.82%`).

*(Exact live run metrics are persisted automatically in `artifacts/model_trainer/model_comparison.json` and `artifacts/model_evaluation/metrics.json`.)*

---

## 3. Model Card (`InsuranceRiskPredictorModel`)

### Intended Use
* **Primary Use**: Real-time actuarial cost estimation and risk tiering (`Low Risk < $8k`, `Moderate Risk $8k–$20k`, `High Risk >= $20k`) via FastAPI REST service.
* **Out-of-Scope Use**: Clinical diagnosis or automated denial of healthcare coverage without human actuarial review.

### Limitations & Residual Analysis
* Unexplained variance (~10% of variance, $1 - R^2 \approx 0.10$) arises from sporadic acute medical events (e.g., emergency surgeries or accidents among young non-smokers) that are not captured by baseline demographic features (`age`, `bmi`, `smoker`, `children`, `region`, `sex`).
