"""
Real-Time Inference Pipeline for FastAPI Service.
Loads serialized preprocessor.pkl and model.pkl and produces risk predictions + risk tier classification.
"""
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException
from src.risk_predictor.utils.common import load_object


class CustomData:
    """Maps incoming API/UI feature values to a structured pandas DataFrame."""

    def __init__(
        self,
        age: int,
        sex: str,
        bmi: float,
        children: int,
        smoker: str,
        region: str,
    ):
        self.age = int(age)
        self.sex = str(sex).lower().strip()
        self.bmi = float(bmi)
        self.children = int(children)
        self.smoker = str(smoker).lower().strip()
        self.region = str(region).lower().strip()

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)


class PredictionPipeline:
    """Loads saved artifacts and runs low-latency risk inference."""

    def __init__(
        self,
        model_path: Path = Path("artifacts/model_trainer/model.pkl"),
        preprocessor_path: Path = Path("artifacts/data_transformation/preprocessor.pkl"),
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            pred_value = float(model.predict(data_scaled)[0])
            pred_value = max(0.0, round(pred_value, 2))

            if pred_value < 8000:
                risk_tier = "Low Risk"
            elif pred_value < 20000:
                risk_tier = "Moderate Risk"
            else:
                risk_tier = "High Risk"

            logger.info(
                f"Prediction completed: estimated_annual_risk_cost=${pred_value:,.2f} ({risk_tier})"
            )

            return {
                "predicted_charges_usd": pred_value,
                "risk_tier": risk_tier,
                "model_used": type(model).__name__,
            }
        except Exception as e:
            raise CustomException(e, sys)
