import logging
from logging.handlers import RotatingFileHandler

from src.paths import project_path

LOG_PATH = project_path("logs", "pipeline.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 7


def get_logger(name: str) -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler = RotatingFileHandler(
            LOG_PATH,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger
