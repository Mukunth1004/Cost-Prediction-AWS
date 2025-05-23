# src/utils/logger.py
import logging
import sys
from src.config import config

def get_logger(name):
    """Configure and return a logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger