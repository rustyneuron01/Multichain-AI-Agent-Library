from typing import Optional, Dict, Any
import httpx
from datetime import datetime
import json
from fiber.logging_utils import get_logger
from validator.config import SCORE_VISION_API
import asyncio
import os
import time
from fiber.chain.signatures import sign_message
from fiber.chain.chain_utils import load_hotkey_keypair

logger = get_logger(__name__)

NETUID = os.getenv("NETUID", "44")
WALLET_NAME = os.getenv("WALLET_NAME", "default")
HOTKEY_NAME = os.getenv("HOTKEY_NAME", "default")

def optimize_bbox_coordinates(bbox):
    """Round bbox coordinates to 2 decimal places to reduce payload size."""
    return [round(float(x), 2) for x in bbox]

def optimize_keypoints(keypoints):
    """Round keypoint coordinates to 2 decimal places."""
    return [round(float(x), 2) for x in keypoints]

def optimize_response_data(response_data: dict) -> dict:
    """
    Optimize response data to reduce payload size.
    - Rounds coordinates to 2 decimal places
    - Removes unnecessary metadata
    - Optimizes data structure
    """
    optimized_data = {}
    
    for frame_id, frame_data in response_data.get("frames", {}).items():
        optimized_frame = {}
        
        # Optimize players data
        if "players" in frame_data:
            optimized_frame["players"] = []
            for player in frame_data["players"]:
                optimized_player = {
                    "bbox": optimize_bbox_coordinates(player["bbox"]),
                    "class_id": player.get("class_id", 2)  # Default to regular player
                }
                optimized_frame["players"].append(optimized_player)
        
        # Optimize ball data
        if "ball" in frame_data:
            optimized_frame["ball"] = []
            for ball in frame_data["ball"]:
                optimized_ball = {
                    "bbox": optimize_bbox_coordinates(ball["bbox"])
                }
                optimized_frame["ball"].append(optimized_ball)
        
        # Optimize keypoints
        if "keypoints" in frame_data:
            optimized_frame["keypoints"] = [
                optimize_keypoints(point) for point in frame_data["keypoints"]
            ]
        
        optimized_data[frame_id] = optimized_frame
    
    return {
        "frames": optimized_data,
        "challenge_id": response_data.get("challenge_id"),
        "processing_time": round(float(response_data.get("processing_time", 0)), 2)
    }

def log_data_size(data: Dict, prefix: str = "") -> None:
    """Log the size of data and its components."""
    try:
        # Convert to JSON string to get actual payload size
        data_json = json.dumps(data)
        total_size = len(data_json)
        
        logger.info(f"{prefix}Total payload size: {total_size / 1024:.2f} KB ({total_size:,} bytes)")
        
        # Log sizes of main components
        if isinstance(data, dict):
            for key, value in data.items():
                component_size = len(json.dumps(value))
                if component_size > 1024:  # Only log components larger than 1KB
                    logger.info(f"{prefix}{key}: {component_size / 1024:.2f} KB")
                    
                    # For frames data, provide more detailed breakdown
                    if key == "frames" and isinstance(value, dict):
                        total_keypoints = 0
                        total_players = 0
                        total_balls = 0
                        
                        for frame_data in value.values():
                            if isinstance(frame_data, dict):
                                keypoints = frame_data.get("keypoints", [])
                                players = frame_data.get("players", [])
                                balls = frame_data.get("ball", [])
                                
                                total_keypoints += len(keypoints)
                                total_players += len(players)
                                total_balls += len(balls)
                        
                        logger.info(f"{prefix}Frame stats:")
                        logger.info(f"{prefix}- Total frames: {len(value)}")
                        logger.info(f"{prefix}- Total keypoints: {total_keypoints}")
                        logger.info(f"{prefix}- Total players detected: {total_players}")
                        logger.info(f"{prefix}- Total balls detected: {total_balls}")
        
        # Warn if approaching size limit
        if total_size > 900000:  # 900KB warning threshold
            logger.warning(f"Payload size ({total_size / 1024:.2f} KB) is approaching the 1MB limit!")
            
    except Exception as e:
        logger.error(f"Error logging data size: {str(e)}")

async def get_next_challenge(validator_hotkey: str) -> dict:
    """
    Fetch the next challenge from the API.
    
    Args:
        validator_hotkey (str): The validator's hotkey.
    
    Returns:
        dict: Challenge data if successful, None otherwise.
    """
    try:
        # Load the hotkey
        keypair = load_hotkey_keypair(WALLET_NAME, HOTKEY_NAME)
        
        # Generate nonce (current time in nanoseconds)
        nonce = str(int(time.time() * 1e9))
        
        # Sign the nonce
        signature = sign_message(keypair, nonce)
        
        # Prepare query parameters
        params = {
            "validator_hotkey": validator_hotkey,
            "signature": signature,
            "nonce": nonce,
            "netuid": NETUID
        }
        
        # Use the API URL from config
        url = f"{SCORE_VISION_API}/api/tasks/next/v2"
        
        # Log the full URL that will be constructed
        from urllib.parse import urlencode
        full_url = f"{url}?{urlencode(params)}"
        logger.debug(f"Making request to: {full_url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            challenge_data = response.json()
            
            if challenge_data:
                # Rename 'id' to 'task_id' in the response
                if 'id' in challenge_data:
                    challenge_data['task_id'] = challenge_data.pop('id')
                logger.info(f"Fetched challenge: {challenge_data}")
                return challenge_data
            else:
                logger.warning("No challenge available from API")
                return None
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while fetching the challenge: {str(e)}")
        return None

async def update_task_scores(
    validator_address: str,
    task_id: str,
    challenge_id: str,
    miner_id: str,
    miner_hotkey: str,
    response_data: str,
    evaluation_score: float,
    speed_score: float,
    total_score: float,
    processing_time: float,
    started_at: Optional[str],
    completed_at: Optional[str]
) -> bool:
    """
    Update task scores via API.
    
    Args:
        validator_address: Validator's public key hex
        task_id: Task ID
        challenge_id: Challenge ID
        miner_id: Miner's node ID
        miner_hotkey: Miner's public key hex
        response_data: Response data as JSON string
        evaluation_score: Quality evaluation score
        speed_score: Speed score
        total_score: Total weighted score
        processing_time: Processing time in seconds
        started_at: ISO format timestamp string when task started
        completed_at: ISO format timestamp string when task completed
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the hotkey for authentication
        keypair = load_hotkey_keypair(WALLET_NAME, HOTKEY_NAME)
        
        # Generate nonce (current time in nanoseconds)
        nonce = str(int(time.time() * 1e9))
        
        # Sign the nonce
        signature = sign_message(keypair, nonce)
        
        # Parse response data
        response_data_dict = json.loads(response_data)
        
        # Prepare request data
        data = {
            "challenge_id": int(challenge_id),
            "miner_id": int(miner_id),
            "miner_hotkey": miner_hotkey,
            "response_data": response_data_dict,
            "evaluation_score": float(evaluation_score),
            "speed_score": float(speed_score),
            "total_score": float(total_score),
            "availability_score": 1.0,
            "processing_time": float(processing_time),
            "started_at": started_at,
            "completed_at": completed_at
        }
        
        # Enhanced logging with challenge context
        logger.info(f"Preparing API update for challenge {challenge_id}:")
        logger.info(f"  Endpoint: {SCORE_VISION_API}/api/tasks/update")
        logger.info(f"  Validator: {validator_address}")
        logger.info(f"  Miner: {miner_id} ({miner_hotkey})")
        logger.info(f"  Scores:")
        logger.info(f"    - Evaluation: {evaluation_score:.3f}")
        logger.info(f"    - Speed: {speed_score:.3f}")
        logger.info(f"    - Total: {total_score:.3f}")
        logger.info(f"  Timing:")
        logger.info(f"    - Processing time: {processing_time:.2f}s")
        logger.info(f"    - Started: {started_at}")
        logger.info(f"    - Completed: {completed_at}")
        
        # Make request with retries
        url = f"{SCORE_VISION_API}/api/tasks/update"
        
        # Prepare authentication parameters
        params = {
            "validator_hotkey": validator_address,
            "signature": signature,
            "nonce": nonce,
            "netuid": NETUID
        }
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    logger.info(f"Sending API request (attempt {retry_count + 1}/{max_retries})")
                    response = await client.post(url, params=params, json=data)
                    if not response.is_success:
                        logger.error(f"API Error Response")
                    response.raise_for_status()
                    logger.info(f"Successfully updated scores for challenge {challenge_id}")
                    return True
                    
            except httpx.HTTPError as e:
                retry_count += 1
                logger.error(f"API request failed (Attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count == max_retries:
                    logger.error(f"Failed to update scores for challenge {challenge_id} after {max_retries} attempts")
                    return False
                logger.info(f"Retrying in 1 second...")
                await asyncio.sleep(1)
                
    except Exception as e:
        logger.error(f"Error preparing task score update for challenge {challenge_id}: {str(e)}")
        return False
