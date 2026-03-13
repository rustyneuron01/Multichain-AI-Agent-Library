from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from ...core.utils import get_health_check, get_system_status
from ...core.models.utility_models import HealthCheck, SystemStatus
from ...core.constants import VERSION
from ..dependencies import get_db_session

router = APIRouter(prefix="/health", tags=["health"])

start_time = datetime.utcnow()

@router.get("/status")
async def get_status(db: Session = Depends(get_db_session)) -> SystemStatus:
    """Get current system status."""
    status = get_system_status()
    
    # Add database connection status
    try:
        db.execute("SELECT 1")
        status.components["database"] = True
    except Exception:
        status.components["database"] = False
    
    return status

@router.get("/check")
async def health_check(db: Session = Depends(get_db_session)) -> HealthCheck:
    """Perform a health check."""
    health = get_health_check(VERSION, start_time)
    
    # Add database health check
    try:
        db.execute("SELECT 1")
        health.components["database"] = True
    except Exception:
        health.components["database"] = False
    
    # Add storage health check
    try:
        from ..dependencies import get_challenge_dir, get_results_dir
        challenge_dir = get_challenge_dir()
        results_dir = get_results_dir()
        health.components["storage"] = challenge_dir.exists() and results_dir.exists()
    except Exception:
        health.components["storage"] = False
    
    return health
