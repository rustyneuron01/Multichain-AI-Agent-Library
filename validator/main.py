import asyncio
import json
import os
import sys
import time
import random

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List

import aiohttp
import httpx
from dotenv import load_dotenv
from fiber.chain import signatures, fetch_nodes
from fiber.chain.models import Node
from fiber.chain.interface import get_substrate
from fiber.chain.chain_utils import load_hotkey_keypair, load_coldkeypub_keypair
from loguru import logger
from substrateinterface import Keypair

from multiprocessing import Process
from validator.evaluation.evaluation_process import start_evaluation
from validator.challenge.challenge_process import start_challenge_sender

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

from validator.db.operations import DatabaseManager
from validator.config import (
    NETUID, SUBTENSOR_NETWORK, SUBTENSOR_ADDRESS,
    WALLET_NAME, HOTKEY_NAME,
    MIN_MINERS, MAX_MINERS, MIN_STAKE_THRESHOLD,
    CHALLENGE_INTERVAL, CHALLENGE_TIMEOUT, DB_PATH,
    SCORE_THRESHOLD, WEIGHTS_INTERVAL, OPENAI_API_KEY
)
from validator.challenge.send_challenge import send_challenge
from validator.challenge.challenge_types import (
    ChallengeType, GSRChallenge, GSRResponse, ValidationResult, ChallengeTask
)
from validator.evaluation.evaluation import GSRValidator
from validator.evaluation.set_weights import set_weights
from validator.evaluation.calculate_score import calculate_score
from validator.db.schema import init_db
from validator.evaluation.evaluation_loop import run_evaluation_loop
from validator.utils.api import get_next_challenge
from validator.utils.async_utils import AsyncBarrier

# TODO check why stopped working, only doing availablity check, but not logging anything.
# Load environment variables
validator_dir = Path(__file__).parent
env_path = validator_dir / ".env"
load_dotenv(env_path)


                    
class ChallengeTask:
    def __init__(self, node_id: int, task: asyncio.Task, timestamp: datetime, challenge: GSRChallenge, miner_hotkey: str):
        self.node_id = node_id
        self.task = task
        self.timestamp = timestamp
        self.challenge = challenge
        self.miner_hotkey = miner_hotkey
        self.frames_to_validate = None  # Will be set after all responses are received

def get_active_nodes_with_stake() -> list[Node]:
    """Get list of active nodes that meet the stake requirement (less than 100 TAO)."""
    try:
        # Get nodes from chain
        substrate = get_substrate(
            subtensor_network=SUBTENSOR_NETWORK,
            subtensor_address=SUBTENSOR_ADDRESS
        )
        
        nodes = fetch_nodes.get_nodes_for_netuid(substrate, NETUID)
        logger.info(f"Found {len(nodes)} total nodes on chain")
        
        # Filter for active nodes with less than 100 TAO stake
        MAX_STAKE = 999  # 999 TAO maximum stake
        active_nodes = [
            node for node in nodes
            if node.stake < MAX_STAKE  # Changed from >= MIN_STAKE_THRESHOLD to < MAX_STAKE
        ]
        
        # Log details about active nodes
        logger.info(f"Found {len(active_nodes)} nodes with stake less than {MAX_STAKE} TAO")
        for node in active_nodes:
            logger.info(f"Active node id: {node.node_id} hotkey: {node.hotkey}, ip: {node.ip}, port: {node.port}, last_updated: {node.last_updated}")
        
        # Return all active nodes without MAX_MINERS filtering
        return active_nodes
        
    except Exception as e:
        logger.error(f"Failed to get active nodes: {str(e)}")
        return []

async def process_challenge_results(
    challenge_tasks: List[ChallengeTask],
    db_manager: DatabaseManager,
    validator: GSRValidator,
    keypair: Keypair,
    substrate: Any
) -> None:
    """Process challenge results without blocking."""
    logger.info(f"Processing {len(challenge_tasks)} challenge results")
    
    # Wait for all tasks to complete with timeout
    pending = [task.task for task in challenge_tasks]
    timeout = 3600  # 1 hour timeout
    
    while pending:
        # Wait for the next task to complete, with timeout
        done, pending = await asyncio.wait(
            pending,
            timeout=60,  # Check every minute for completed tasks
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Process completed tasks
        for task in done:
            try:
                response = await task
                # Process the response...
                logger.debug(f"Processed challenge response: {response}")
            except Exception as e:
                logger.error(f"Error processing challenge result: {str(e)}")
        
        # Log status of remaining tasks
        if pending:
            logger.info(f"Still waiting for {len(pending)} challenges to complete")
    
    logger.info("All challenge results processed")

def construct_server_address(node: Node) -> str:
    """Construct server address for a node.
    
    For local development:
    - Nodes register as 0.0.0.1 on the chain (since 127.0.0.1 is not allowed)
    - But we connect to them via 127.0.0.1 locally
    """
    if node.ip == "0.0.0.1":
        # For local development, connect via localhost
        return f"http://127.0.0.1:{node.port}"
    return f"http://{node.ip}:{node.port}"
    
async def weights_update_loop(db_manager: DatabaseManager) -> None:
    """Run the weights update loop on WEIGHTS_INTERVAL."""
    logger.info("Starting weights update loop")
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        try:
            await set_weights(db_manager)
            consecutive_failures = 0  # Reset failure counter on success
            logger.info(f"Weights updated successfully, sleeping for {WEIGHTS_INTERVAL}")
            await asyncio.sleep(WEIGHTS_INTERVAL.total_seconds())
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Error in weights update loop (attempt {consecutive_failures}/{max_consecutive_failures}): {str(e)}")
            
            if consecutive_failures >= max_consecutive_failures:
                logger.error("Too many consecutive failures in weights update loop, waiting for longer period")
                await asyncio.sleep(WEIGHTS_INTERVAL.total_seconds() * 2)  # Wait twice as long before retrying
                consecutive_failures = 0  # Reset counter after long wait
            else:
                # Wait normal interval before retry
                await asyncio.sleep(WEIGHTS_INTERVAL.total_seconds())

async def periodic_cleanup(db_manager: DatabaseManager, interval_hours: int = 24):
    """
    Periodically clean up old data from the database and debug frames.
    
    Args:
        db_manager: DatabaseManager instance
        interval_hours: Number of hours between cleanup operations
    """
    while True:
        try:
            logger.info("Starting periodic database and debug frames cleanup")
            
            # Clean up database
            db_manager.cleanup_old_data()
            logger.info("Database cleanup completed")
            
            # Clean up debug frames older than 7 days
            debug_frames_dir = Path("debug_frames")
            if debug_frames_dir.exists():
                current_time = datetime.now()
                deleted_count = 0
                
                # Iterate through all date subdirectories
                for date_dir in debug_frames_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                        
                    # Check all files in the date directory
                    for frame_file in date_dir.iterdir():
                        if not frame_file.is_file():
                            continue
                            
                        file_age = current_time - datetime.fromtimestamp(frame_file.stat().st_mtime)
                        if file_age.days > 7:
                            try:
                                frame_file.unlink()
                                deleted_count += 1
                            except Exception as e:
                                logger.error(f"Error deleting old debug frame {frame_file}: {str(e)}")
                    
                    # Try to remove empty date directories
                    try:
                        if not any(date_dir.iterdir()):
                            date_dir.rmdir()
                    except Exception as e:
                        logger.error(f"Error removing empty directory {date_dir}: {str(e)}")
                
                logger.info(f"Removed {deleted_count} debug frames older than 7 days")
            
            logger.info("Periodic cleanup completed")
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {str(e)}")
        
        # Wait for the next cleanup interval
        await asyncio.sleep(interval_hours * 3600)

async def get_next_challenge_with_retry(hotkey: str, max_retries: int = 2, initial_delay: float = 5.0) -> Optional[dict]:
    """
    Attempt to fetch the next challenge from the API with retries.
    
    Args:
        hotkey (str): The validator's hotkey.
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay in seconds between retries.
    
    Returns:
        Optional[dict]: Challenge data if successful, None otherwise.
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Fetching next challenge from API (Attempt {attempt + 1}/{max_retries + 1})...")
            challenge_data = await get_next_challenge(hotkey)
            if challenge_data:
                logger.info(f"Successfully fetched challenge: task_id={challenge_data['task_id']}")
                return challenge_data
            else:
                logger.warning("No challenge available from API")
        except Exception as e:
            logger.error(f"Error fetching challenge: {str(e)}")
        
        if attempt < max_retries:
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    
    logger.warning("Failed to fetch challenge after all retry attempts")
    return None



async def main():
    """Main validator loop."""
    # Load configuration
    load_dotenv()
    
    logger.info(f"Using database path: {DB_PATH}")
    
    # Get environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Load validator keys
    try:
        hotkey = load_hotkey_keypair(WALLET_NAME, HOTKEY_NAME)
        coldkey = load_coldkeypub_keypair(WALLET_NAME)
    except Exception as e:
        logger.error(f"Failed to load keys: {str(e)}")
        return

    # Initialize database manager and validator
    logger.info(f"Initializing database manager with path: {DB_PATH}")
    db_manager = DatabaseManager(DB_PATH)
    validator = GSRValidator(openai_api_key=OPENAI_API_KEY, validator_hotkey=hotkey.ss58_address)
    


    # Initialize substrate connection
    substrate = get_substrate(
        subtensor_network=SUBTENSOR_NETWORK,
        subtensor_address=SUBTENSOR_ADDRESS
    )

    # Initialize HTTP client with long timeout
    async with httpx.AsyncClient(timeout=CHALLENGE_TIMEOUT.total_seconds()) as client:
        os.environ["DB_PATH"] = str(DB_PATH)
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["VALIDATOR_HOTKEY"] = hotkey.ss58_address
        # Start evaluation loop as a separate task
        logger.info("Starting evaluation loop task...")
        
        evaluation_proc = Process(target=start_evaluation)
        evaluation_proc.start()

        challenge_proc = Process(target=start_challenge_sender)
        challenge_proc.start()
        
        # Start weights update loop as a separate task
        logger.info("Starting weights update task...")
        weights_task = asyncio.create_task(weights_update_loop(db_manager))
        weights_task.add_done_callback(
            lambda t: logger.error(f"Weights task ended unexpectedly: {t.exception()}")
            if t.exception() else None
        )
    
        # Start the periodic cleanup task
        logger.info("Starting cleanup task...")
        cleanup_task = asyncio.create_task(periodic_cleanup(db_manager))
        cleanup_task.add_done_callback(
            lambda t: logger.error(f"Cleanup task ended unexpectedly: {t.exception()}")
            if t.exception() else None
        )
        
        try:
            # Main challenge loop
            iteration = 0
            while True:
                try:
                    iteration += 1
                    logger.info(f"Main loop iteration {iteration}")
                    
                    # Check if any background tasks failed
                    for task in [weights_task, cleanup_task]:
                        if task.done() and not task.cancelled():
                            exc = task.exception()
                            if exc:
                                logger.error(f"Background task failed: {exc}")
                                # Restart the failed task
                                if task == weights_task:
                                    logger.info("Restarting weights update loop...")
                                    weights_task = asyncio.create_task(weights_update_loop(db_manager))
                                elif task == cleanup_task:
                                    logger.info("Restarting cleanup task...")
                                    cleanup_task = asyncio.create_task(periodic_cleanup(db_manager))

                    # Log background task status
                    logger.info("Background task status:")
                    logger.info(f"  - Weights task running: {not weights_task.done()}")
                    logger.info(f"  - Cleanup task running: {not cleanup_task.done()}")
                    
                    # Sleep until next challenge interval
                    await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
        finally:
            # Cancel evaluation and weights loops
            weights_task.cancel()
            cleanup_task.cancel()
            try:
                await asyncio.gather(weights_task, cleanup_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
                
            if evaluation_proc.is_alive():
                logger.info("Terminating evaluation subprocess...")
                evaluation_proc.terminate()
                evaluation_proc.join()
                
            if challenge_proc.is_alive():
                logger.info("Terminating challenge subprocess...")
                challenge_proc.terminate()
                challenge_proc.join()

    # Cleanup
    if db_manager:
        db_manager.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
