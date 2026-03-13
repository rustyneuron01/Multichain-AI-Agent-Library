import asyncio
import json
from datetime import datetime, timezone
import httpx
from fiber.logging_utils import get_logger
from fiber.validator import client as validator
from fiber import Keypair
from validator.challenge.challenge_types import GSRChallenge, GSRResponse
from validator.config import CHALLENGE_TIMEOUT
from typing import List, Dict
import uuid
from validator.utils.async_utils import AsyncBarrier

logger = get_logger(__name__)

def optimize_coordinates(coords: List[float]) -> List[float]:
    """Round coordinates to 2 decimal places to reduce data size."""
    return [round(float(x), 2) for x in coords]

def optimize_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """Round keypoint coordinates to 2 decimal places."""
    return [optimize_coordinates(kp) for kp in keypoints]

def optimize_response_data(response_data: Dict) -> Dict:
    """Optimize response data by rounding coordinates while preserving all keypoints."""
    optimized_data = {}
    
    # Handle frames data
    if "frames" in response_data and isinstance(response_data["frames"], list):
        frames = {}
        for frame in response_data["frames"]:
            frame_num = str(frame["frame_number"])
            frame_data = {}
            
            # Optimize keypoints if present - preserve all keypoints including (0,0)
            if "keypoints" in frame and isinstance(frame["keypoints"], list):
                frame_data["keypoints"] = optimize_keypoints(frame["keypoints"])
                
            # Optimize bounding boxes if present
            if "objects" in frame and isinstance(frame["objects"], list):
                optimized_objects = []
                for obj in frame["objects"]:
                    optimized_obj = obj.copy()
                    if "bbox" in obj:
                        optimized_obj["bbox"] = optimize_coordinates(obj["bbox"])
                    optimized_objects.append(optimized_obj)
                frame_data["objects"] = optimized_objects
                
            frames[frame_num] = frame_data
            
        optimized_data = frames
        
    return optimized_data

async def send_challenge(
    challenge: GSRChallenge,
    server_address: str,
    hotkey: str,
    keypair: Keypair,
    node_id: int,
    barrier: AsyncBarrier,
    db_manager=None,
    client: httpx.AsyncClient = None,
    timeout: float = CHALLENGE_TIMEOUT.total_seconds()  # Use config timeout in seconds
) -> httpx.Response:
    """Send a challenge to a miner node using fiber 2.0.0 protocol."""
    endpoint = "/soccer/challenge"
    payload = challenge.to_dict()

    logger.info(f"Preparing to send challenge to node {node_id}")
    logger.info(f"  Server address: {server_address}")
    logger.info(f"  Endpoint: {endpoint}")
    logger.info(f"  Hotkey: {hotkey}")
    logger.info(f"  Challenge ID: {challenge.challenge_id}")
    logger.info(f"  Video URL: {challenge.video_url}")

    remaining_barriers = 2
    response = None
    
    try:
        # First, store the challenge in the challenges table
        if db_manager:
            logger.debug(f"Storing challenge {challenge.challenge_id} in database")
            db_manager.store_challenge(
                challenge_id=challenge.challenge_id,
                challenge_type=str(challenge.type),  # Convert enum to string
                video_url=challenge.video_url,
                task_name="soccer"
            )

        # Record the assignment
        if db_manager:
            logger.debug(f"Recording challenge assignment in database")
            db_manager.assign_challenge(challenge.challenge_id, hotkey, node_id)

        # Create client if not provided
        should_close_client = False
        if client is None:
            logger.debug("Creating new HTTP client")
            client = httpx.AsyncClient(timeout=timeout)
            should_close_client = True

        if db_manager:
            logger.debug("Marking challenge as sent in database")
            db_manager.mark_challenge_sent(challenge.challenge_id, hotkey)

        if remaining_barriers: 
            await barrier.wait()
            remaining_barriers -= 1
        
        try:
            sent_time = datetime.now(timezone.utc)

            logger.debug("Sending challenge request...")
            
            # Send the challenge using fiber validator client with long timeout
            try:
                response = await validator.make_non_streamed_post(
                    httpx_client=client,
                    server_address=server_address,
                    validator_ss58_address=keypair.ss58_address,
                    miner_ss58_address=hotkey,
                    keypair=keypair,
                    endpoint=endpoint,
                    payload=payload,
                    timeout=timeout
                )
                response.raise_for_status()
            except Exception as e:
                logger.warning(f"Error sending challenge to {hotkey} (node {node_id}): {e}. Returning empty response.")
                response =  httpx.Response(
                    status_code=200,
                    json={"frames": []},
                )
            received_time = datetime.now(timezone.utc)
            processing_time = (received_time - sent_time).total_seconds()

            if remaining_barriers: 
                await barrier.wait()
                remaining_barriers -= 1
            
            logger.debug(f"Got response with status code: {response.status_code}")
            
            try:
                response_data = response.json()

                if response_data is None:
                    response_data={"frames": {}, "processing_time": 60.0}
                # Log essential information about the response
                logger.info(f"Received response for challenge {challenge.challenge_id}:")
                logger.info(f"  Processing time: {processing_time:.2f} seconds")
                logger.info(f"  Number of frames: {len(response_data.get('frames', []))}")
                
                # Optimize response data
                optimized_response = optimize_response_data(response_data)
                
                # Create GSRResponse with parsed data
                gsr_response = GSRResponse(
                    challenge_id=challenge.challenge_id,
                    miner_hotkey=hotkey,
                    node_id=node_id,
                    frames=optimized_response,
                    processing_time=processing_time,
                    received_at=sent_time
                )
                
                # Store response in responses table
                if db_manager:
                    logger.debug("Storing response in database")
                    response_id = db_manager.store_response(
                        challenge_id=challenge.challenge_id,
                        miner_hotkey=hotkey,
                        response=gsr_response,
                        node_id=node_id,
                        processing_time=processing_time,
                        received_at=sent_time,
                        completed_at=received_time
                    )
                    
                    logger.info(f"Stored response {response_id} in database")
                
                logger.info(f"Challenge {challenge.challenge_id} sent successfully to {hotkey} (node {node_id})")

            except Exception as e:
                logger.error("Failed to process response")
                logger.error(e)
                logger.error("Full error traceback:", exc_info=True)
                raise

            finally:
                return response
            
        except Exception as e:
            if remaining_barriers: 
                await barrier.wait()
                remaining_barriers -= 1
            logger.error(f"Response error: {str(e)}")
            logger.error(f"Response status code: {response.status_code}")
            logger.error(f"Response headers: {response.headers}")
            error_msg = f"Failed to send challenge {challenge.challenge_id} to {hotkey} (node {node_id}): {str(e)}"
            logger.error(error_msg)
            logger.error("Full error traceback:", exc_info=True)
            raise ValueError(error_msg)
            
        finally:
            if should_close_client:
                logger.debug("Closing HTTP client")
                await client.aclose()
                
    except Exception as e:
        if remaining_barriers: 
            await barrier.wait()
            remaining_barriers -= 1
        if remaining_barriers: 
            await barrier.wait()
            remaining_barriers -= 1
        error_msg = f"Failed to send challenge {challenge.challenge_id} to {hotkey} (node {node_id}): {str(e)}"
        logger.error(error_msg)
        logger.error("Full error traceback:", exc_info=True)
        raise ValueError(error_msg)
        
