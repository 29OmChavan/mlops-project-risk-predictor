"""
Custom Exception Handling Module for Production MLOps Pipeline.
Extracts the exact Python script filename, line number, and error message from sys.exc_info().
"""
import sys
from src.risk_predictor.logger import logger


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Constructs a detailed error message including file name and line number.

    Args:
        error (Exception): The caught exception instance.
        error_detail (sys): The sys module providing exc_info() traceback.

    Returns:
        str: Formatted diagnostic error string.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = 0

    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """Custom exception class that logs detailed file/line diagnostics on raise."""

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
