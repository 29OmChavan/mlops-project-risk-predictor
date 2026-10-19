"""
Stage 01: Data Ingestion Component
Pulls raw data via Kaggle API / HTTP source URL (with deterministic fallback for offline CI tests),
deduplicates records, performs train/test split, and saves ingestion artifacts.
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException
from src.risk_predictor.entity.config_entity import DataIngestionConfig
from src.risk_predictor.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    """Handles downloading, deduplicating, and splitting the raw dataset."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _generate_fallback_dataset(self, n_samples: int = 1338) -> pd.DataFrame:
        """
        Generates a realistic, deterministic Medical Insurance Risk dataset
        if neither Kaggle API nor network URL is reachable (e.g., air-gapped CI).
        """
        rng = np.random.default_rng(self.config.random_state)
        age = rng.integers(18, 65, size=n_samples)
        sex = rng.choice(["female", "male"], size=n_samples)
        bmi = np.round(rng.normal(30.6, 6.1, size=n_samples).clip(15.9, 53.1), 2)
        children = rng.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.43, 0.24, 0.18, 0.11, 0.02, 0.02])
        smoker = rng.choice(["yes", "no"], size=n_samples, p=[0.205, 0.795])
        region = rng.choice(
            ["southeast", "southwest", "northwest", "northeast"], size=n_samples
        )

        # Realistic non-linear actuarial risk formula matching the medical insurance dataset
        base_charge = 250.0 * age + 320.0 * np.maximum(bmi - 25.0, 0) + 475.0 * children
        smoker_multiplier = np.where(smoker == "yes", 15000.0 + (bmi > 30) * 14500.0, 0.0)
        noise = rng.normal(0.0, 1200.0, size=n_samples)
        charges = np.round(np.maximum(1120.0, base_charge + smoker_multiplier + noise), 2)

        return pd.DataFrame(
            {
                "age": age.astype("int64"),
                "sex": sex,
                "bmi": bmi.astype("float64"),
                "children": children.astype("int64"),
                "smoker": smoker,
                "region": region,
                "charges": charges.astype("float64"),
            }
        )

    def download_data(self) -> Path:
        """
        Downloads dataset using Kaggle API token (if available) or public mirror URL.
        Returns the path to the raw CSV file.
        """
        try:
            raw_path = self.config.raw_data_file
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)

            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                logger.info(f"Raw data file already exists at {raw_path}")
                return raw_path

            # 1. Attempt Kaggle API v1 download using KAGGLE_API_TOKEN or ~/.kaggle/access_token
            kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
            token_file = Path.home() / ".kaggle" / "access_token"
            if not kaggle_token and token_file.exists():
                kaggle_token = token_file.read_text(encoding="utf-8").strip()
                os.environ["KAGGLE_API_TOKEN"] = kaggle_token

            downloaded = False
            if kaggle_token:
                try:
                    logger.info(
                        f"Attempting Kaggle download for dataset: {self.config.kaggle_dataset}"
                    )
                    kaggle_url = f"https://www.kaggle.com/api/v1/datasets/download/{self.config.kaggle_dataset}"
                    req = urllib.request.Request(
                        kaggle_url,
                        headers={"Authorization": f"Bearer {kaggle_token}"},
                    )
                    zip_path = self.config.root_dir / "dataset.zip"
                    with urllib.request.urlopen(req, timeout=20) as resp, open(
                        zip_path, "wb"
                    ) as out_file:
                        out_file.write(resp.read())

                    with zipfile.ZipFile(zip_path, "r") as zf:
                        csv_members = [m for m in zf.namelist() if m.endswith(".csv")]
                        if csv_members:
                            zf.extract(csv_members[0], path=self.config.root_dir)
                            extracted_csv = self.config.root_dir / csv_members[0]
                            if extracted_csv != raw_path:
                                os.replace(extracted_csv, raw_path)
                            downloaded = True
                            logger.info(
                                f"Successfully downloaded dataset from Kaggle API to {raw_path}"
                            )
                    if zip_path.exists():
                        os.remove(zip_path)
                except Exception as kaggle_err:
                    logger.warning(
                        f"Kaggle API download skipped/fell back ({kaggle_err}). Trying mirror URL..."
                    )

            # 2. Fallback to GitHub raw dataset mirror URL
            if not downloaded:
                try:
                    logger.info(f"Downloading dataset from source URL: {self.config.source_url}")
                    urllib.request.urlretrieve(self.config.source_url, raw_path)
                    downloaded = True
                    logger.info(f"Downloaded raw dataset to {raw_path}")
                except Exception as url_err:
                    logger.warning(
                        f"Mirror URL unreachable ({url_err}). Generating deterministic dataset..."
                    )

            # 3. Deterministic offline fallback
            if not downloaded:
                df_fallback = self._generate_fallback_dataset()
                df_fallback.to_csv(raw_path, index=False)
                logger.info(f"Generated deterministic offline dataset at {raw_path}")

            return raw_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Executes the full Data Ingestion stage:
        1. Downloads raw CSV
        2. Removes duplicate rows
        3. Splits into train and test sets
        4. Returns DataIngestionArtifact
        """
        logger.info("Entered DataIngestion.initiate_data_ingestion()")
        try:
            raw_path = self.download_data()
            df = pd.read_csv(raw_path)
            logger.info(f"Loaded raw dataset with shape: {df.shape}")

            # Deduplicate records
            initial_rows = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            dropped = initial_rows - len(df)
            if dropped > 0:
                logger.info(f"Removed {dropped} duplicate row(s). Clean shape: {df.shape}")

            # Save cleaned raw dataset
            df.to_csv(raw_path, index=False)

            # Stratified split based on smoker status to preserve target skew distribution
            stratify_col = df["smoker"] if "smoker" in df.columns else None
            train_set, test_set = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_col,
            )

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logger.info(
                f"Train/Test split complete. Train shape: {train_set.shape}, Test shape: {test_set.shape}"
            )

            return DataIngestionArtifact(
                raw_file_path=raw_path,
                train_file_path=self.config.train_data_path,
                test_file_path=self.config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
