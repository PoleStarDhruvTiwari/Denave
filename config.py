"""All configuration in one place"""
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Create directories
for dir_path in [LOG_DIR, DATA_DIR, INPUT_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

# Model Settings
LLM_CONFIG = {
    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "temperature": 0.1,
    "max_tokens": 2000,
}

# Quality Thresholds
QUALITY = {
    "min_overall_score": 75.0,
    "min_dimension_score": 70.0,
    "balance_diff_threshold": 5.0,  # 5% tolerance for A = L + E
}

# Logging
logger.remove()
logger.add(
    LOG_DIR / "extractor_{time:YYYY-MM-DD}.log",
    rotation="50 MB",
    retention="30 days",
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} | {message}",
)
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
)