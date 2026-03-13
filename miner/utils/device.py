import torch
import platform
from loguru import logger

def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available on macOS."""
    try:
        if platform.system() == "Darwin":
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                return True
    except:
        pass
    return False

def get_optimal_device(requested_device: str = None) -> str:
    """
    Determine the optimal device based on availability and request.
    
    Args:
        requested_device: Optional device request ('cuda', 'mps', 'cpu')
        
    Returns:
        str: The actual device to use ('cuda', 'mps', or 'cpu')
    """
    if requested_device == "cuda":
        if torch.cuda.is_available():
            logger.info("Using CUDA device as requested")
            return "cuda"
        logger.warning("CUDA requested but not available, falling back to CPU")
        return "cpu"
    
    if requested_device == "mps":
        if is_mps_available():
            logger.info("Using MPS device as requested")
            return "mps"
        logger.warning("MPS requested but not available, falling back to CPU")
        return "cpu"
    
    if requested_device == "cpu":
        logger.info("Using CPU device as requested")
        return "cpu"
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        logger.info("No device specified, CUDA available - using CUDA")
        return "cuda"
    if is_mps_available():
        logger.info("No device specified, MPS available - using MPS")
        return "mps"
    
    logger.info("No device specified or no accelerator available - using CPU")
    return "cpu" 