# Role Simulation: Data Engineer — Data Requirement & Pipeline Specification

**Project**: Project 01 · End-to-End ML Pipeline (Medical Insurance & Financial Risk Predictor)  
**Owner Role**: Data Engineer  
**Dataset**: Kaggle `mirichoi0218/insurance` (US Medical Cost Personal Dataset)

---

## 1. Objective & Business Context
Health and financial insurance underwriters require an accurate, automated system to estimate annual medical claim risk (`charges` in USD) from demographic and clinical attributes. Because medical claims exhibit severe right-skewness driven by non-linear interactions (particularly smoking combined with obesity, $\text{BMI} \ge 30$), a production data pipeline must validate schema integrity, detect covariate shift, and engineer domain-specific interaction features reproducibly.

---

## 2. Data Schema & Domain Constraints (`schema.yaml`)

| Column Name | Data Type | Role | Domain Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `age` | `int64` | Numerical Feature | $[18, 100]$ | Age of primary insurance beneficiary |
| `sex` | `object` | Categorical Feature | `{"female", "male"}` | Biological sex of contractor |
| `bmi` | `float64` | Numerical Feature | $[10.0, 70.0]$ | Body Mass Index ($\text{kg/m}^2$); ideal $18.5\text{–}24.9$, obese $\ge 30.0$ |
| `children` | `int64` | Numerical Feature | $[0, 10]$ | Number of dependents covered by health insurance |
| `smoker` | `object` | Categorical Feature | `{"yes", "no"}` | Tobacco smoking status |
| `region` | `object` | Categorical Feature | `{"southwest", "southeast", "northwest", "northeast"}` | Beneficiary's residential area in the US |
| `charges` | `float64` | Target ($y$) | $> 0$ | Individual annual medical risk costs billed by health insurance (USD) |

---

## 3. Data Ingestion Architecture (`src/risk_predictor/components/data_ingestion.py`)

### 3.1 Multi-Tier Source Resolution
To guarantee zero-downtime reproducibility across developer laptops, GitHub Actions CI runners, and air-gapped Docker builds, `DataIngestion.download_data()` resolves data in 3 prioritized tiers:
1. **Primary**: Kaggle API v1 (`https://www.kaggle.com/api/v1/datasets/download/mirichoi0218/insurance`) authenticated via `~/.kaggle/access_token` or `KAGGLE_API_TOKEN`.
2. **Secondary Mirror**: Verified raw CSV mirror (`stedy/Machine-Learning-with-R-datasets`).
3. **Tertiary Air-Gapped Fallback**: Deterministic actuarial data generator (`_generate_fallback_dataset`) seeded with `random_state=42`.

### 3.2 Deduplication & Stratified Splitting
* Raw dataset contains `1,338` rows with `1` exact duplicate record (`1,337` unique rows).
* Because `smoker == "yes"` represents only ~20.5% of records yet accounts for the upper mode of the bimodal cost distribution, we perform a **stratified 80/20 train-test split** on `smoker`:

```python
stratify_col = df["smoker"] if "smoker" in df.columns else None
train_set, test_set = train_test_split(
    df,
    test_size=self.config.test_size,       # 0.20 (268 test rows, 1069 train rows)
    random_state=self.config.random_state, # 42
    stratify=stratify_col,
)
```

---

## 4. Data Validation & Statistical Drift Detection (`src/risk_predictor/components/data_validation.py`)

Every pipeline execution produces three auditable validation artifacts in `artifacts/data_validation/`:
1. `status.txt`: Gate boolean (`Validation status: True/False`) halting training if schema or range rules fail.
2. `drift_report.yaml`: Two-sample **Kolmogorov-Smirnov (`ks_2samp`)** test comparing empirical cumulative distribution functions $F_{\text{train}}(x)$ and $F_{\text{test}}(x)$ across all numerical features:
   $$D = \sup_x |F_{\text{train}}(x) - F_{\text{test}}(x)|$$
   If $p\text{-value} < \alpha = 0.05$, covariate drift is flagged.
3. `eda_summary.json`: Complete summary statistics (quantiles, mean, std, null counts, category frequencies).
