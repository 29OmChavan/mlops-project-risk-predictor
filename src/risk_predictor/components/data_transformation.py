"""
Stage 03: Data Transformation Component
Builds an sklearn ColumnTransformer & Pipeline with custom domain feature engineering,
imputation, encoding, and scaling. Saves preprocessor.pkl and transformed numpy arrays.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException
from src.risk_predictor.utils.common import save_object
from src.risk_predictor.entity.config_entity import DataTransformationConfig
from src.risk_predictor.entity.artifact_entity import DataTransformationArtifact


class RiskDomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that engineers clinically & actuarially meaningful
    interaction features prior to encoding and scaling:
    - `is_obese`: Binary indicator (1 if BMI >= 30.0 else 0)
    - `obese_smoker`: High-risk synergistic interaction (1 if BMI >= 30.0 and smoker == 'yes')
    - `bmi_age_interaction`: Continuous interaction (BMI * age) / 100.0
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        is_obese = (X_out["bmi"].astype(float) >= 30.0).astype(float)
        is_smoker = (X_out["smoker"].astype(str).str.lower() == "yes").astype(float)
        X_out["is_obese"] = is_obese
        X_out["obese_smoker"] = is_obese * is_smoker
        X_out["bmi_age_interaction"] = (
            X_out["bmi"].astype(float) * X_out["age"].astype(float)
        ) / 100.0
        return X_out


class DataTransformation:
    """Constructs and executes the feature engineering and preprocessing pipeline."""

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self) -> Pipeline:
        """
        Builds the complete scikit-learn Pipeline combining:
        1. RiskDomainFeatureGenerator (domain interactions)
        2. ColumnTransformer (median imputation + StandardScaler for numeric;
           mode imputation + OneHotEncoder + StandardScaler for categoricals)
        """
        try:
            engineered_num_cols = list(self.config.numerical_columns) + [
                "is_obese",
                "obese_smoker",
                "bmi_age_interaction",
            ]
            categorical_cols = list(self.config.categorical_columns)

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logger.info(f"Numerical columns (with engineered features): {engineered_num_cols}")
            logger.info(f"Categorical columns: {categorical_cols}")

            column_transformer = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, engineered_num_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols),
                ]
            )

            full_preprocessor = Pipeline(
                steps=[
                    ("feature_generator", RiskDomainFeatureGenerator()),
                    ("column_transformer", column_transformer),
                ]
            )

            return full_preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Reads train and test CSVs, fits preprocessor on train features, transforms both sets,
        and saves preprocessor.pkl + numpy arrays.
        """
        logger.info("Entered DataTransformation.initiate_data_transformation()")
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            target_col = self.config.target_column
            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            np.save(self.config.transformed_train_path, train_arr)
            np.save(self.config.transformed_test_path, test_arr)

            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessing_obj,
            )

            logger.info(
                f"Data transformation complete. Train array shape: {train_arr.shape}, "
                f"Test array shape: {test_arr.shape}"
            )

            return DataTransformationArtifact(
                preprocessor_object_file_path=self.config.preprocessor_path,
                transformed_train_file_path=self.config.transformed_train_path,
                transformed_test_file_path=self.config.transformed_test_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
