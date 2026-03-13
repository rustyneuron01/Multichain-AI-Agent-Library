import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
validator_dir = Path(__file__).parent
env_path = validator_dir / ".env"
load_dotenv(env_path)

# Network configuration
NETUID = int(os.getenv("NETUID", "261"))
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "test")
SUBTENSOR_ADDRESS = os.getenv("SUBTENSOR_ADDRESS", "127.0.0.1:9944")

# Validator configuration
HOTKEY_NAME = os.getenv("HOTKEY_NAME", "default")
WALLET_NAME = os.getenv("WALLET_NAME", "default")
MIN_STAKE_THRESHOLD = float(os.getenv("MIN_STAKE_THRESHOLD", "2"))
VALIDATOR_PORT = int(os.getenv("VALIDATOR_PORT", "8000"))
VALIDATOR_HOST = os.getenv("VALIDATOR_HOST", "0.0.0.0")

# Default configuration values
MIN_MINERS = 1
MAX_MINERS = 60
SCORE_THRESHOLD = 0.7
FRAMES_TO_VALIDATE = 2
ALPHA_SCORING_MULTIPLICATOR = 4
SCORE_VISION_API = "https://api.scorevision.io"
MAX_PROCESSING_TIME = 15.0
#SCORE_VISION_API = "http://localhost:8000"

VERSION_KEY = 2027

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# Additional settings needed for operation
CHALLENGE_INTERVAL = timedelta(minutes=60)
CHALLENGE_TIMEOUT = timedelta(minutes=4)

WEIGHTS_INTERVAL = timedelta(minutes=30)
VALIDATION_DELAY = timedelta(minutes=5)

# Database configuration with environment variable support
DB_PATH = Path(os.getenv("DB_PATH", "validator.db"))

# Log initial configuration
import logging
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

logger.info("Validator Configuration:")
logger.info(f"Network: {SUBTENSOR_NETWORK}")
logger.info(f"Netuid: {NETUID}")
logger.info(f"Min miners: {MIN_MINERS}")
logger.info(f"Max miners: {MAX_MINERS}")
logger.info(f"Min stake threshold: {MIN_STAKE_THRESHOLD}")
logger.info(f"Score threshold: {SCORE_THRESHOLD}")
logger.info(f"Frames to validate: {FRAMES_TO_VALIDATE}")
logger.info(f"Challenge interval: {CHALLENGE_INTERVAL}")
logger.info(f"Challenge timeout: {CHALLENGE_TIMEOUT}")
logger.info(f"Weights interval: {WEIGHTS_INTERVAL}")
logger.info(f"DB path: {DB_PATH}")
logger.info(f"Log level: {LOG_LEVEL}")
logger.info(f"Validation delay: {VALIDATION_DELAY}")
logger.info(f"Alpha multiplicator for scoring function : {ALPHA_SCORING_MULTIPLICATOR}")
