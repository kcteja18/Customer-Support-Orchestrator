"""Logging configuration and utilities."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("customer_support")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def log_query_metrics(logger: logging.Logger, query: str, response: str, 
                      retrieval_time: float, generation_time: float, 
                      num_docs: int, confidence: float):
    """Log query processing metrics."""
    logger.info(
        f"Query processed | "
        f"retrieval={retrieval_time:.3f}s | "
        f"generation={generation_time:.3f}s | "
        f"docs={num_docs} | "
        f"confidence={confidence:.2f}"
    )
