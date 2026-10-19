"""
Stage 02: Data Validation & EDA Component
Performs schema validation, domain constraint checks, automated EDA statistics,
and Kolmogorov-Smirnov (KS) statistical data drift detection between train and test splits.
"""
import sys
from typing import Dict, Any, Tuple
import pandas as pd
from scipy.stats import ks_2samp

from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException
from src.risk_predictor.utils.common import write_yaml, save_json
from src.risk_predictor.entity.config_entity import DataValidationConfig
from src.risk_predictor.entity.artifact_entity import DataValidationArtifact


class DataValidation:
    """Validates dataset schema, constraints, and statistical distribution drift."""

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns_and_constraints(
        self, df: pd.DataFrame, split_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks column presence, null counts, and domain constraints defined in schema.yaml.
        """
        try:
            validation_status = True
            anomalies = []
            expected_cols = set(self.config.all_schema.keys())
            actual_cols = set(df.columns)

            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols

            if missing_cols:
                validation_status = False
                anomalies.append(f"[{split_name}] Missing columns: {sorted(missing_cols)}")

            if extra_cols:
                logger.warning(f"[{split_name}] Unexpected extra columns: {sorted(extra_cols)}")

            # Check missing values
            null_counts = df.isnull().sum().to_dict()
            total_nulls = sum(null_counts.values())
            if total_nulls > 0:
                logger.warning(f"[{split_name}] Found {total_nulls} missing values: {null_counts}")

            # Check domain constraints
            for col, rules in self.config.constraints.items():
                if col not in df.columns:
                    continue
                if "min" in rules and (df[col] < rules["min"]).any():
                    validation_status = False
                    anomalies.append(
                        f"[{split_name}] Column '{col}' has values below min={rules['min']}"
                    )
                if "max" in rules and (df[col] > rules["max"]).any():
                    validation_status = False
                    anomalies.append(
                        f"[{split_name}] Column '{col}' has values above max={rules['max']}"
                    )
                if "allowed" in rules:
                    invalid_vals = set(df[col].dropna().unique()) - set(rules["allowed"])
                    if invalid_vals:
                        validation_status = False
                        anomalies.append(
                            f"[{split_name}] Column '{col}' has invalid categories: {invalid_vals}"
                        )

            for anomaly in anomalies:
                logger.error(f"Data Validation Anomaly: {anomaly}")

            return validation_status, {
                "split": split_name,
                "row_count": int(len(df)),
                "null_counts": {k: int(v) for k, v in null_counts.items()},
                "anomalies": anomalies,
                "passed": validation_status,
            }

        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float = 0.05
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Runs two-sample Kolmogorov-Smirnov (KS) test on numerical columns to detect covariate shift.
        If p-value < alpha (0.05), drift is flagged for that feature.
        """
        try:
            drift_report: Dict[str, Any] = {}
            any_drift_detected = False

            numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
            for col in numeric_cols:
                stat_res = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
                p_value = float(stat_res.pvalue)
                ks_stat = float(stat_res.statistic)
                is_drifted = bool(p_value < alpha)

                if is_drifted:
                    any_drift_detected = True
                    logger.warning(
                        f"Drift detected in column '{col}': p_value={p_value:.5f}, ks_stat={ks_stat:.5f}"
                    )

                drift_report[col] = {
                    "ks_statistic": round(ks_stat, 5),
                    "p_value": round(p_value, 5),
                    "drift_detected": is_drifted,
                }

            write_yaml(self.config.drift_report_file, drift_report)
            return any_drift_detected, drift_report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Executes full Data Validation & EDA stage and writes status, EDA summary, and drift reports.
        """
        logger.info("Entered DataValidation.initiate_data_validation()")
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            train_ok, train_report = self.validate_all_columns_and_constraints(train_df, "train")
            test_ok, test_report = self.validate_all_columns_and_constraints(test_df, "test")
            validation_status = bool(train_ok and test_ok)

            drift_detected, drift_report = self.detect_dataset_drift(train_df, test_df)

            # Automated EDA summary
            eda_summary = {
                "validation_status": validation_status,
                "drift_detected": drift_detected,
                "train_validation": train_report,
                "test_validation": test_report,
                "numeric_summary_train": train_df.describe().round(3).to_dict(),
                "categorical_counts_train": {
                    col: train_df[col].value_counts().to_dict()
                    for col in train_df.select_dtypes(include=["object"]).columns
                },
            }
            save_json(self.config.eda_summary_file, eda_summary)

            with open(self.config.status_file, "w", encoding="utf-8") as f:
                f.write(f"Validation status: {validation_status}\n")
                f.write(f"Drift detected: {drift_detected}\n")

            logger.info(
                f"Data validation completed. Status={validation_status}, Drift={drift_detected}"
            )

            return DataValidationArtifact(
                validation_status=validation_status,
                status_file_path=self.config.status_file,
                drift_report_file_path=self.config.drift_report_file,
                eda_summary_file_path=self.config.eda_summary_file,
                drift_detected=drift_detected,
            )

        except Exception as e:
            raise CustomException(e, sys)
