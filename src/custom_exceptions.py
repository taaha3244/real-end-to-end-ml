import sys
# Directly set the project root directory
project_root = "/workspaces/real-end-to-end-ml"
# Ensure the project root is at the top of sys.path
sys.path.insert(0, project_root)
from src.logging.custom_logger import logger  # Ensure the custom logger is imported

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        "Error occurred in python script name [{0}] line number [{1}] error message [{2}]"
        .format(file_name, exc_tb.tb_lineno, str(error))
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logger.error(self.error_message)  # Log the error message

    def __str__(self):
        return self.error_message
