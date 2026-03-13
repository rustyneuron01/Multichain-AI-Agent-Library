import os
import json
import base64
import httpx
import tempfile
from typing import Dict, List, Tuple, Optional
import random
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import asyncio
from openai import OpenAI
from fiber.logging_utils import get_logger
from validator.challenge.challenge_types import (
    GSRResponse,
    GSRChallenge,
    ValidationResult
)
from validator.config import FRAMES_TO_VALIDATE
from validator.evaluation.prompts import VALIDATION_PROMPT
from validator.utils.vlm_api import VLMProcessor
from validator.evaluation.bbox_clip import (evaluate_frame, evaluate_bboxes)
from validator.evaluation.keypoint_scoring import (process_input_file, calculate_final_score_keypoints)

FRAME_TIMEOUT = 180.0  # seconds

logger = get_logger(__name__)

# Class IDs
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Colors
COLORS = {
    "player": (0, 255, 0),
    "goalkeeper": (0, 0, 255),
    "referee": (255, 0, 0),
    "ball": (0, 255, 255),
    "keypoint": (255, 0, 255)
}

def optimize_coordinates(coords: List[float]) -> List[float]:
    """Round coordinates to 2 decimals."""
    return [round(float(x), 2) for x in coords]

def filter_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """Remove zero-coord keypoints; round others to 2 decimals."""
    return [optimize_coordinates(kp) for kp in keypoints if not (kp[0] == 0 and kp[1] == 0)]

class GSRValidator:
    def __init__(self, openai_api_key: str, validator_hotkey: str):
        self.openai_api_key = openai_api_key
        self.validator_hotkey = validator_hotkey
        self.db_manager = None
        self._video_cache = {}
        self.vlm_processor = VLMProcessor(openai_api_key)
        self.frame_reference_counts = {}

    def encode_image(self, image):
        """Base64-encode an image."""
        ok, buf = cv2.imencode('.jpg', image)
        return base64.b64encode(buf).decode('utf-8') if ok else ""

    async def download_video(self, video_url: str) -> Path:
        """
        Download video or return from cache if possible. Handles direct URLs or Google Drive.
        """
        if video_url in self._video_cache:
            cached_path = self._video_cache[video_url]
            if cached_path.exists():
                logger.info(f"Using cached video at: {cached_path}")
                return cached_path
            else:
                del self._video_cache[video_url]

        logger.info(f"Downloading video from: {video_url}")
        if 'drive.google.com' in video_url:
            file_id = None
            if 'id=' in video_url:
                file_id = video_url.split('id=')[1].split('&')[0]
            elif '/d/' in video_url:
                file_id = video_url.split('/d/')[1].split('/')[0]
            if not file_id:
                raise ValueError("Failed to extract Google Drive file ID from URL")
            video_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

        max_retries, retry_delay, timeout = 3, 5, 60.0
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.get(video_url)
                    if resp.status_code==404:
                        logger.error(f"❌ Video not found (404): {video_url}. Skipping challenge.")
                        return None
                    resp.raise_for_status()
                    temp_dir = Path(tempfile.gettempdir())
                    path = temp_dir / f"video_{datetime.now().timestamp()}.mp4"
                    path.write_bytes(resp.content)
                    if not path.exists() or path.stat().st_size == 0:
                        raise ValueError("Video is empty/missing")

                    cap = cv2.VideoCapture(str(path))
                    if not cap.isOpened():
                        cap.release()
                        path.unlink(missing_ok=True)
                        raise ValueError("Not a valid video")
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    logger.info(f"Video stats: {frame_count} frames, {fps} FPS")
                    self._video_cache[video_url] = path
                    return path
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All download attempts failed: {str(e)}")
                    if 'path' in locals() and path.exists():
                        path.unlink()
                    logger.warning(f"Failed to download: {str(e)}")
                    return None


    async def validate_keypoints(self, frames: dict,video_path, video_width: int, video_height: int, frames_to_validate, pitch_lines_by_frame: dict) -> dict:
        """
        Uses the advanced scoring system for keypoints.
        Returns per-frame keypoint scores and stats.
        """
        try:
            (
                results,
                valid_frames,
                avg_inlier_ratio,
                avg_reprojection_error,
                avg_keypoint_score,
                player_score,
                avg_jump,
                total_jumps,
                biggest_jump,
                large_jumps,
                transitions,
                avg_keypoint_stability,
                avg_player_plausibility,
                mean_on_line, 
                mean_inside, 
                mean_scale, 
                scale_valid,
                scene_scores, 
                kpts_stability_weighted_avg
            ) = process_input_file(frames,video_path, video_width, video_height, frames_to_validate, pitch_lines_by_frame)
    
            final_score = calculate_final_score_keypoints(
                avg_keypoint_score,
                player_score,
                avg_keypoint_stability,
                avg_player_plausibility,
                mean_on_line, 
                mean_inside, 
                mean_scale, 
                scale_valid, 
                kpts_stability_weighted_avg
            )
    
            return {
                "per_frame_scores": results,
                "final_score": final_score,
                "components": {
                    "avg_keypoint_score": avg_keypoint_score,
                    "player_score": player_score,
                    "scale": (mean_scale+scale_valid)/2 ,
                    "mean_inside": mean_inside,
                    "mean_on_line": mean_on_line
                }
            }
    
        except Exception as e:
            logger.error(f"Error in new keypoint validation: {str(e)}")
            return {
                "per_frame_scores": {},
                "final_score": 0.0,
                "components": {}
            }

    async def validate_bbox_clip(self, frame_idx: int, frame, detections: dict) -> float:
        try:
            objects = detections.get("objects", [])
            if not objects:
                return 0.0
            return evaluate_frame(frame_idx, frame.copy(), objects)
        except Exception as e:
            logger.error(f"[Frame {frame_idx}] BBox CLIP validation failed: {e}")
            return 0.0
        
    async def evaluate_response(
        self,
        response: GSRResponse,
        challenge: GSRChallenge,
        video_path: Path,
        video_width: int,
        video_height: int,
        frames_to_validate: List[int] = None,
        selected_frames_id_bbox: List[int] = None,
        pitch_lines_by_frame: dict = None
    ) -> ValidationResult:

        temp_frames=response.frames

        if isinstance(temp_frames, list):
            logger.warning("Legacy formatting detected. Updating...")
            temp_frames = {
                frame.get('frame_number',str(i)):frame
                for i, frame in enumerate(temp_frames)
                }

        filtered_frames = {
            str(k): v for k, v in temp_frames.items() if int(k) in frames_to_validate
        }
        # Analyse keypoints (globale)
        scoring_result = await self.validate_keypoints(filtered_frames, video_path, video_width, video_height,frames_to_validate, pitch_lines_by_frame)
        per_frame_keypoints = scoring_result.get("per_frame_scores", {})
        keypoints_final_score = scoring_result.get("final_score", 0.0)/100

        frame_evals = []
        total_bbox_score = 0.0
        frame_scores = {}
        frame_details = []
        
        selected_frames_bbox = {str(i): v for i, v in temp_frames.items() if int(i) in selected_frames_id_bbox}
        
        logger.info(f'Starting to evaluate {len(selected_frames_id_bbox)} frames')
        avg_bbox_score = await evaluate_bboxes(
            prediction=selected_frames_bbox,
            path_video=video_path,
            n_frames=750,
            n_valid=len(selected_frames_id_bbox)
        )

        logger.info(f'avg bbox score : {avg_bbox_score}')
        return ValidationResult(
            score=avg_bbox_score,
            frame_scores=frame_scores,
            feedback={
                "frame_details": frame_details,
                "keypoints_final_score": keypoints_final_score
            }
        )


    def calculate_bbox_confidence_score(self, results: dict) -> float:
        """
        Weighted average of all object validation scores (0..1).
        Different classes get different weighting.
        """
        objs = results.get("objects", [])
        if not objs:
            return 0.0

        total, weight_sum = 0.0, 0.0
        weights = {
            "soccer ball": 0.7,
            "goalkeeper": 0.3,
            "referee": 0.2,
            "soccer player": 1.0
        }
        for o in objs:
            cls_name = o["class"]
            prob = o["probability"]
            w = weights.get(cls_name, 0.5)
            total += prob * w
            weight_sum += w
        return total / weight_sum if weight_sum else 0.0

    def calculate_final_score(self, keypoint_score: float, bbox_score: float) -> float:
        """
        Combine keypoints, bboxes, and object counts into final 0..1.
        """
        KEY_W, BOX_W = 0.5, 0.5
        return (
            (keypoint_score * KEY_W) +
            (bbox_score * BOX_W) 
        )
    

    def select_random_frames(self, video_path: Path, num_frames: int = None) -> List[int]:
        """
        Randomly pick frames from a video, skipping start/end buffer.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to select. If None, uses FRAMES_TO_VALIDATE from config
            
        Returns:
            List of selected frame numbers
        """
        num_frames = num_frames or FRAMES_TO_VALIDATE
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Calculate buffer size - either 30 frames or 10% of total, whichever is smaller
        buffer = min(5, total // 2)
        
        # Ensure we have enough frames to sample from
        if total <= (2 * buffer):
            logger.warning(f"Video too short ({total} frames) for buffer size {buffer}")
            buffer = total // 4  # Use 25% of total as buffer if video is very short
            
        # Calculate valid frame range
        start_frame = buffer
        end_frame = max(buffer, total - buffer)
        valid_range = range(start_frame, end_frame)
        
        if len(valid_range) < num_frames:
            logger.warning(f"Not enough frames ({len(valid_range)}) to select {num_frames} samples")
            num_frames = len(valid_range)
            
        frames = random.sample(valid_range, num_frames) if valid_range else []
        logger.info(f"Selected {len(frames)} frames from {total}")
        return sorted(frames)

    def draw_annotations(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Draw bounding boxes and keypoints onto the frame."""
        out = frame.copy()
        for obj in detections.get("objects", []):
            (x1, y1, x2, y2) = obj["bbox"]
            cid = obj["class_id"]
            if cid == BALL_CLASS_ID:
                color = COLORS["ball"]
            elif cid == GOALKEEPER_CLASS_ID:
                color = COLORS["goalkeeper"]
            elif cid == REFEREE_CLASS_ID:
                color = COLORS["referee"]
            else:
                color = COLORS["player"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        for (x, y) in detections.get("keypoints", []):
            if x != 0 and y != 0:
                cv2.circle(out, (int(x), int(y)), 5, COLORS["keypoint"], -1)
        return out

    def get_class_name(self, class_id: int) -> str:
        """Map class_id to string. (Legacy usage retained.)"""
        names = {0: "soccer ball", 1: "goalkeeper", 2: "player", 3: "referee"}
        return names.get(class_id, "unknown")

    def validate_bbox_coordinates(
        self,
        bbox: List[float],
        frame_shape: Tuple[int, int],
        class_id: int
    ) -> Optional[List[int]]:
        """
        Clamp bbox to frame bounds. Discard invalid or tiny ones.
        """
        try:
            h, w = frame_shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            if x2 <= x1 or y2 <= y1:
                return None
            x1, x2 = sorted([max(0, min(x1, w)), max(0, min(x2, w))])
            y1, y2 = sorted([max(0, min(y1, h)), max(0, min(y2, h))])
            if class_id == 0:  # ball can be small
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    return None
            else:
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    return None
            return [x1, y1, x2, y2]
        except Exception as e:
            logger.error(f"BBox validation error: {str(e)}")
            return None

    def resize_frame(self, frame: np.ndarray, target_width: int = 400) -> np.ndarray:
        """Keep aspect ratio on resize."""
        h, w = frame.shape[:2]
        aspect = w / h
        return cv2.resize(frame, (target_width, int(target_width / aspect)))

    def filter_detections(self, detections: Dict, shape: Tuple[int, int]) -> Dict:
        """Clamp bboxes, keep valid ones, preserve keypoints."""
        valid = {"objects": [], "keypoints": detections.get("keypoints", [])}
        for obj in detections.get("objects", []):
            bbox = self.validate_bbox_coordinates(obj["bbox"], shape, obj["class_id"])
            if bbox:
                valid["objects"].append({**obj, "bbox": bbox})
        return valid
