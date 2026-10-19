"""
Package initialization for risk_predictor.
Exposes logger and CustomException for convenient top-level imports.
"""
from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException

__all__ = ["logger", "CustomException"]
