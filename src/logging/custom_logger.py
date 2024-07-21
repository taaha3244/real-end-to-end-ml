import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Directory for log files
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Log file name with timestamp
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path = os.path.join(logs_dir, log_file)

# Create logger
logger = logging.getLogger("ml_project_logger")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)  # 10 MB per file
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

if __name__ == "__main__":
    logger.info("Logging has started")
