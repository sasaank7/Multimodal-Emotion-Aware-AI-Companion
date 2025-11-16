"""
Logging utilities for the Emotion-Aware AI Companion.
Provides consistent logging across all modules.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from .config_loader import get_config


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    # Get configuration
    config = get_config()

    # Determine log level
    if level is None:
        level = config.get("logging.level", "INFO")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    log_format = config.get(
        "logging.format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter = logging.Formatter(log_format)

    # Console handler
    if config.get("logging.console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_file = config.get("logging.file", "logs/app.log")

    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for a module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    if name not in logging.Logger.manager.loggerDict:
        return setup_logger(name)
    return logging.getLogger(name)
