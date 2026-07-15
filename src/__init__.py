from dotenv import load_dotenv

from src.logging_config import get_logger

# Loads OPENAI_API_KEY / FMP_API_KEY etc. from a local .env file (gitignored)
# into the environment, if one exists. Real env vars already set take
# precedence over .env (load_dotenv default).
load_dotenv()

__all__ = ["get_logger"]
