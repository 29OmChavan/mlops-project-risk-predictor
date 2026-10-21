"""
Common Utility Functions for YAML, JSON, and Joblib Artifact I/O.
"""
import os
import sys
import json
import yaml
import joblib
from pathlib import Path
from typing import Any, Dict, List
from src.risk_predictor.logger import logger
from src.risk_predictor.exception import CustomException


def read_yaml(path_to_yaml: Path) -> Dict[str, Any]:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        path_to_yaml (Path): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed YAML dictionary.
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                return {}
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return content
    except Exception as e:
        raise CustomException(e, sys)


def write_yaml(path_to_yaml: Path, data: Dict[str, Any]) -> None:
    """
    Writes a dictionary to a YAML file, creating parent directories if needed.

    Args:
        path_to_yaml (Path): Target path for YAML output.
        data (Dict[str, Any]): Dictionary content to serialize.
    """
    try:
        os.makedirs(os.path.dirname(path_to_yaml), exist_ok=True)
        with open(path_to_yaml, "w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(data, yaml_file, sort_keys=False)
        logger.info(f"YAML file saved at: {path_to_yaml}")
    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_to_directories: List[Any], verbose: bool = True) -> None:
    """
    Creates a list of directories on disk.

    Args:
        path_to_directories (List[Any]): List of directory paths to create.
        verbose (bool): Whether to log directory creation.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Saves a dictionary as a formatted JSON file.

    Args:
        path (Path): Target file path.
        data (Dict[str, Any]): Data dictionary to save.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Loads JSON data from a file.

    Args:
        path (Path): Path to JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON content.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return content
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: Path, obj: Any) -> None:
    """
    Serializes and saves a Python object (e.g., sklearn Pipeline or Model) using joblib.

    Args:
        file_path (Path): Destination file path.
        obj (Any): Python object to serialize.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Binary object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: Path) -> Any:
    """
    Loads a binary object serialized with joblib.

    Args:
        file_path (Path): Path to the serialized object.

    Returns:
        Any: Deserialized Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact file not found at: {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Binary object loaded from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)
