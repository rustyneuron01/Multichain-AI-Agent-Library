import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Set
from pathlib import Path
import time
import random

import httpx
from fiber.logging_utils import get_logger
from validator.db.operations import DatabaseManager
from validator.evaluation.evaluation import GSRValidator
from validator.challenge.challenge_types import GSRResponse, ValidationResult, GSRChallenge, ChallengeType
from validator.evaluation.calculate_score import calculate_score
from validator.utils.api import update_task_scores
from validator.config import VALIDATION_DELAY, FRAMES_TO_VALIDATE
from loguru import logger
import cv2
import tempfile
from dataclasses import dataclass
from validator.utils.frame_filter import (detect_pitch, batch_clip_verification, init_clip_model)
from validator.evaluation.keypoint_scoring import detect_pitch_lines_tophat

# New constant for minimum number of players
MIN_PLAYERS_PER_FRAME = 4
INITIAL_FRAME_SAMPLE_SIZE = 20
RESPONSES_TO_CHECK = 10

# Constants for batching and workers
BATCH_SIZE = 5  # Number of frames to process in one batch
MAX_WORKERS = 5  # Number of concurrent worker tasks
MAX_RETRIES = 3  # Maximum retry attempts for failed batches
WORKER_TIMEOUT = 300  # 5 minutes timeout for worker tasks

logger = get_logger(__name__)

async def _evaluate_single_response(
    validator: GSRValidator,
    db_manager: DatabaseManager,
    video_path: Path,
    row: sqlite3.Row,
    frame_cache: Dict = None,
    frames_to_validate: List[int] = None
) -> Dict[str, Any]:
    """Evaluate a single response."""
    try:
        # Log row information (excluding response_data)
        row_dict = dict(row)
        if 'response_data' in row_dict:
            row_dict['response_data'] = '<omitted>'
        #logger.info(f"Processing row: {row_dict}")
        
        # Get response data
        response_data = json.loads(row["response_data"] if row["response_data"] else "{}")
        #logger.info(f"Response metadata - challenge_id: {row['challenge_id']}, miner_hotkey: {row['miner_hotkey']}")
        
        # Create GSRResponse object with node_id
        response = GSRResponse(
            challenge_id=row["challenge_id"],
            miner_hotkey=row["miner_hotkey"],
            frames=response_data.get("frames", {}),
            processing_time=row["processing_time"],
            response_id=row["response_id"],
            node_id=row["node_id"]
        )
        #logger.info(f"Created GSRResponse object for response_id: {response.response_id}")

        # Get challenge data
        challenge = GSRChallenge(
            challenge_id=row["challenge_id"],
            type=ChallengeType.GSR,
            created_at=row["created_at"] if "created_at" in row.keys() else None,
            video_url=row["video_url"] if "video_url" in row.keys() else ""
        )
        #logger.info(f"Processing challenge_id: {challenge.challenge_id}")

        # Get timing information from database
        started_at = db_manager.get_challenge_assignment_sent_at(challenge.challenge_id, response.miner_hotkey)
        completed_at = row["completed_at"] if "completed_at" in row.keys() else None
        #logger.info(f"Timing info - started_at: {started_at}, completed_at: {completed_at}")

        # Evaluate response using cached frames if available
        result = await validator.evaluate_response(
            response, 
            challenge, 
            video_path, 
            frame_cache=frame_cache,
            frames_to_validate=frames_to_validate
        )
        logger.info(f"Evaluation complete - score: {result.score}")
        
        # Mark response as evaluated
        db_manager.mark_response_as_evaluated(response.response_id)

        return {
            "challenge_id": challenge.challenge_id,
            "miner_hotkey": response.miner_hotkey,
            "node_id": response.node_id,
            "response_id": response.response_id,
            "score": result.score,
            "processing_time": row["processing_time"],
            "validation_result": result,
            "task_returned_data": response_data,
            "started_at": started_at,
            "completed_at": completed_at,
            "received_at": row["received_at"] if "received_at" in row.keys() else None
        }
    except Exception as e:
        logger.error(f"Error evaluating response: {str(e)}")
        logger.error(f"Row object type: {type(row)}")
        if hasattr(row, 'keys'):
            logger.error(f"Available row keys: {row.keys()}")
        raise


async def evaluate_pending_responses(
    validator: GSRValidator,
    db_manager,
    challenge: Dict[str, Any]
) -> None:
    """Evaluate all pending responses for a challenge using the worker pool."""
    eval_queue = None
    try:

        # Get video path
        video_path = await validator.download_video(challenge['video_url'])
        if video_path is None:
            logger.warning(f"Skipping challenge {challenge['challenge_id']} due to missing video (404).")
            db_manager.mark_responses_failed(challenge['challenge_id'])
            return
        if not video_path or not video_path.exists():
            logger.error(f"Failed to download video for challenge {challenge['challenge_id']} skipping")
            db_manager.mark_responses_failed(challenge['challenge_id'])
            return

        # Get pending responses
        responses = await db_manager.get_pending_responses(challenge['challenge_id'])
        if not responses:
            logger.info(f"No pending responses for challenge {challenge['challenge_id']}")
            return

        # Select frames for this challenge
        frame_paths = []
        frame_indices = []
        pitch_lines_by_frame = {}
        
        video_cap = cv2.VideoCapture(str(video_path))
        for idx in range(int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video_cap.read()
            if not success:
                continue
            _, _, _, _, _, final_kept = detect_pitch_lines_tophat(frame)
            pitch_lines_by_frame[str(idx)] = final_kept
            tmp_path = Path(tempfile.gettempdir()) / f"frame_{challenge['challenge_id']}_{idx}.jpg"
            cv2.imwrite(str(tmp_path), frame)
            frame_paths.append(str(tmp_path))
            frame_indices.append(idx)


        clip_scores = batch_clip_verification(frame_paths)

        frames = []
        for i, path in enumerate(frame_paths):
            score = detect_pitch(path, clip_scores=clip_scores)
            if score == 1:
                frames.append(frame_indices[i])
                
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_cap.release()
        
        if len(frames)<75:
            logger.info(f"Skipping challenge {challenge['challenge_id']} having not enough valid frames ({len(frames)})")
            db_manager.mark_responses_failed(challenge['challenge_id'])
            return
        logger.info(f'Accepting challenge with {len(frames)} valid frames for evaluation')

        selected_frames_id_bbox = random.sample(frames, min(100, len(frames)))
        
        # Create and queue tasks for each response
        evaluation_results = []

        for response in responses:
            logger.info(f"Processing response {response.response_id}")
            try:
                result = await validator.evaluate_response(
                    response=response,
                    challenge=GSRChallenge(
                        challenge_id=challenge["challenge_id"],
                        type=ChallengeType.GSR,
                        created_at=challenge.get("created_at"),
                        video_url=challenge["video_url"]
                    ),
                    video_path=video_path,
                    video_width=video_width,
                    video_height=video_height,
                    frames_to_validate=frames,
                    selected_frames_id_bbox=selected_frames_id_bbox,
                    pitch_lines_by_frame=pitch_lines_by_frame
                )
                if result:
                    started_at=db_manager.get_challenge_assignment_sent_at(challenge['challenge_id'], response.miner_hotkey)
                    evaluation_results.append({
                        "challenge_id": challenge["challenge_id"],
                        "miner_hotkey": response.miner_hotkey,
                        "node_id": response.node_id,
                        "response_id": response.response_id,
                        "score": result.score,
                        "processing_time": response.processing_time,
                        "validation_result": result,
                        "task_returned_data": response.frames,
                        "started_at": started_at,
                        "completed_at": None,
                        "received_at": None
                    })
                else: 
                    logger.error(f"No results for response {response.response_id}")
                    
            except Exception as e:
                logger.error(f'Error processing response {response.response_id}')
                db_manager.mark_response_failed(response.response_id)
                continue

        # Calculate final scores and update DB/API
        async with httpx.AsyncClient() as client:
            scores = await calculate_score(evaluation_results, client, validator_hotkey=validator.validator_hotkey, db_manager=db_manager)

            # Trier les réponses par quality_score descendant
            sorted_responses = sorted(scores.items(), key=lambda item: item[1]["quality_score"], reverse=True)

            # Récupérer les response_ids des deux meilleurs
            top_5_response_ids = {response_id for response_id, _ in sorted_responses[:5]}
            
            # Log all scores being processed
            logger.info(f"Processing scores for {len(scores)} responses")
            
            # Update DB and external API for each response
            for response_id, score_data in scores.items():
                node_id = score_data['node_id']
                miner_hotkey = score_data['miner_hotkey']
                
                logger.info(f"Processing response {response_id} for node {node_id}")
                
                # Log detailed scoring information
                logger.info(f"Response {response_id} scoring details:")
                logger.info(f"  - Quality score: {score_data['quality_score']:.3f}")
                logger.info(f"  - Speed score: {score_data['speed_score']:.3f}")
                logger.info(f"  - Final score: {score_data['final_score']:.3f}")
                
                # Update response with score and evaluation status
                db_manager.update_response(
                    response_id=response_id,
                    score=score_data['final_score'],
                    evaluated=True,
                    evaluated_at=datetime.utcnow()
                )
                
                db_manager.store_response_score(
                    response_id=response_id,
                    challenge_id=challenge["challenge_id"],
                    validation_result=score_data['validation_result'],
                    validator_hotkey=validator.validator_hotkey,
                    miner_hotkey=miner_hotkey,
                    node_id=int(node_id),
                    speed_score=score_data['speed_score'],
                    total_score=score_data['final_score']
                )
                # Préparer le bon contenu pour l'API
                if response_id in top_5_response_ids:
                    data_to_send = score_data['task_returned_data']
                else:
                    data_to_send = {}
                # Update external API for each response
                update_success = await update_task_scores(
                    validator_address=validator.validator_hotkey,
                    task_id=challenge["challenge_id"],
                    challenge_id=challenge["challenge_id"],
                    miner_id=node_id,
                    miner_hotkey=score_data['miner_hotkey'],
                    response_data=json.dumps(data_to_send),
                    evaluation_score=score_data['quality_score'],
                    speed_score=score_data['speed_score'],
                    total_score=score_data['final_score'],
                    processing_time=score_data['processing_time'],
                    started_at=(score_data['started_at']),
                    completed_at=datetime.now(timezone.utc).isoformat()
                )
                
                if update_success:
                    logger.info(f"Successfully updated API with scores for response {response_id}")
                else:
                    logger.warning(f"Failed to update API with scores for response {response_id}")
                
                # Add a small delay between API calls to prevent rate limiting
                await asyncio.sleep(0.5)
            
            logger.info(f"Completed processing all {len(scores)} responses for challenge {challenge['challenge_id']}")

        # Cleanup
        logger.info("Continuing with next iteration...")

    except Exception as e:
        logger.error(f"Error in evaluate_pending_responses: {str(e)}")

async def run_evaluation_loop(
    db_path: str,
    openai_api_key: str,
    validator_hotkey: str,
    batch_size: int = 10,
    sleep_interval: int = 60
) -> None:
    """Entrypoint that sets up the DB, validator, and runs the loop."""
    try:
        logger.info("Initializing evaluation loop...")
        init_clip_model()
        db_manager = DatabaseManager(db_path)
        validator = GSRValidator(openai_api_key=openai_api_key, validator_hotkey=validator_hotkey)
        validator.db_manager = db_manager  # Let the validator store frame-level evaluations
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logger.info(f"Starting evaluation loop iteration {iteration}")
                logger.info("Getting database connection...")
                
                # Get pending challenges
                conn = db_manager.get_connection()
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                try:
                    logger.info("Checking for challenges ready for evaluation...")
                    # Get challenges ready for evaluation
                    cursor.execute("""
                        SELECT DISTINCT c.challenge_id, c.video_url, c.type AS challenge_type, 
                               COUNT(r.response_id) as pending_count,
                               MIN(r.received_at) as earliest_received
                        FROM responses r
                        JOIN challenges c ON r.challenge_id = c.challenge_id
                        WHERE r.evaluated = FALSE
                          AND datetime(r.received_at) <= datetime('now', '-' || ? || ' minutes')
                        GROUP BY c.challenge_id, c.video_url, c.type
                        LIMIT 1
                    """, (VALIDATION_DELAY.total_seconds() / 60,))
                    
                    challenge = cursor.fetchone()
                    
                except Exception as e:
                    logger.error(f"Database error while checking for challenges: {str(e)}")
                    raise
                finally:
                    logger.info("Closing database cursor and connection...")
                    cursor.close()
                    conn.close()
                
                if not challenge:
                    logger.info(f"No challenges ready for evaluation (iteration {iteration})")
                    logger.info(f"Preparing to sleep for {sleep_interval} seconds...")
                    sleep_start = time.time()
                    await asyncio.sleep(sleep_interval)
                    sleep_duration = time.time() - sleep_start
                    logger.info(f"Waking up after sleeping for {sleep_duration:.1f} seconds (iteration {iteration})")
                    continue
                    
                logger.info(f"Processing challenge {challenge['challenge_id']} with {challenge['pending_count']} responses (iteration {iteration})")
                
                try:
                    # Process the challenge
                    logger.info("Starting evaluate_pending_responses...")
                    await evaluate_pending_responses(
                        validator=validator,
                        db_manager=db_manager,
                        challenge=dict(challenge)
                    )
                    logger.info(f"Successfully completed challenge processing (iteration {iteration})")
                except Exception as e:
                    logger.error(f"Error processing challenge {challenge['challenge_id']} (iteration {iteration}): {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
                    # Continue to next iteration even if this challenge failed
                
                logger.info(f"Completed evaluation iteration {iteration}, preparing to sleep for {sleep_interval} seconds...")
                sleep_start = time.time()
                await asyncio.sleep(sleep_interval)
                sleep_duration = time.time() - sleep_start
                logger.info(f"Waking up after sleeping for {sleep_duration:.1f} seconds (iteration {iteration})")
                
            except Exception as e:
                logger.error(f"Error in evaluation loop iteration {iteration}: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                logger.info(f"Preparing to sleep for {sleep_interval} seconds before retry...")
                sleep_start = time.time()
                await asyncio.sleep(sleep_interval)
                sleep_duration = time.time() - sleep_start
                logger.info(f"Waking up after sleeping for {sleep_duration:.1f} seconds to retry after error")
                continue  # Ensure we continue the loop after any error
                
    except Exception as e:
        logger.error(f"Fatal error in run_evaluation_loop: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise  # Re-raise the exception to trigger the task's error callback
