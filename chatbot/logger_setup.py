import logging
import os
from logging.handlers import RotatingFileHandler

# Create a logs directory in the current directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

LOG_FILE = os.path.join(LOGS_DIR, "chatbot.log")

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the specified name that logs to both the console and a file.
    """
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't already have handlers (to prevent duplicate logs)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler (max 5MB per file, keep 3 backups)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger
