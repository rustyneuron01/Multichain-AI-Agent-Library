import os
from fastapi import FastAPI
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config
from miner.endpoints.soccer import router as soccer_router
from miner.endpoints.availability import router as availability_router

# Setup logging
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI()

# Add dependencies
app.dependency_overrides[Config] = get_config

# Include routers with their prefixes and tags
app.include_router(
    soccer_router,
    prefix="/soccer",
    tags=["soccer"]
)
app.include_router(
    availability_router,
    tags=["availability"]
) 