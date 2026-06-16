import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        file_handler = RotatingFileHandler(
            "logs/pipeline.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt))
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger
