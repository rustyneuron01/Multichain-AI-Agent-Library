from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from miner.utils.shared import miner_lock

class AvailabilityResponse(BaseModel):
    available: bool

# Create router instance
router = APIRouter()

@router.get("/availability", response_model=AvailabilityResponse)
async def check_availability():
    """Check if the miner is available to process a challenge."""
    is_available = not miner_lock.locked()
    logger.info(f"Miner availability checked: {is_available}")
    return AvailabilityResponse(available=is_available)