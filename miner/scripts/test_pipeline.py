#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from pathlib import Path
import time
from loguru import logger
from typing import List, Dict, Union

miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager
from utils.video_downloader import download_video
from endpoints.soccer import process_soccer_video
from utils.device import get_optimal_device
from scripts.download_models import download_models

TEST_VIDEO_URL = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/test_10secs.mov"

def optimize_coordinates(coords: List[float]) -> List[float]:
    return [round(float(x), 2) for x in coords]

def filter_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    return [optimize_coordinates(kp) for kp in keypoints if not (kp[0] == 0 and kp[1] == 0)]

def optimize_frame_data(frame_data: Dict) -> Dict:
    optimized_data = {}
    
    if "objects" in frame_data:
        optimized_data["objects"] = []
        for obj in frame_data["objects"]:
            optimized_obj = obj.copy()
            if "bbox" in obj:
                optimized_obj["bbox"] = optimize_coordinates(obj["bbox"])
            optimized_data["objects"].append(optimized_obj)
    
    if "keypoints" in frame_data:
        optimized_data["keypoints"] = filter_keypoints(frame_data["keypoints"])
    
    return optimized_data

def optimize_result_data(result: Dict[str, Union[Dict, List, float, str]]) -> Dict[str, Union[Dict, List, float, str]]:
    optimized_result = result.copy()
    
    if "frames" in result:
        frames = result["frames"]
        
        if isinstance(frames, list):
            optimized_frames = {}
            for i, frame_data in enumerate(frames):
                if frame_data:
                    optimized_frames[str(i)] = optimize_frame_data(frame_data)
        elif isinstance(frames, dict):
            optimized_frames = {}
            for frame_num, frame_data in frames.items():
                if frame_data:
                    optimized_frames[str(frame_num)] = optimize_frame_data(frame_data)
        else:
            logger.warning(f"Unexpected frames data type: {type(frames)}")
            optimized_frames = frames
            
        optimized_result["frames"] = optimized_frames
    
    if "processing_time" in result:
        optimized_result["processing_time"] = round(float(result["processing_time"]), 2)
    
    return optimized_result

async def main():
    try:
        logger.info("Starting video processing test")
        start_time = time.time()
        
        logger.info("Checking for required models...")
        download_models()
        
        logger.info(f"Downloading test video from {TEST_VIDEO_URL}")
        video_path = await download_video(TEST_VIDEO_URL)
        logger.info(f"Video downloaded to {video_path}")
        
        try:
            device = get_optimal_device()
            logger.info(f"Using device: {device}")
            
            model_manager = ModelManager(device=device)
            
            logger.info("Loading models...")
            model_manager.load_all_models()
            logger.info("Models loaded successfully")
            
            logger.info("Starting video processing...")
            result = await process_soccer_video(
                video_path=str(video_path),
                model_manager=model_manager
            )
            
            logger.info("Optimizing frame data...")
            optimized_result = optimize_result_data(result)
            
            output_dir = Path(__file__).parent.parent / "test_outputs"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"pipeline_test_results_{int(time.time())}.json"
            
            result_json = json.dumps(optimized_result)
            data_size = len(result_json) / 1024
            logger.info(f"Result data size: {data_size:.2f} KB")
            
            with open(output_file, "w") as f:
                f.write(result_json)
            
            total_time = time.time() - start_time
            frames = len(optimized_result["frames"])
            fps = frames / optimized_result["processing_time"]
            
            logger.info("Processing completed successfully!")
            logger.info(f"Total frames processed: {frames}")
            logger.info(f"Processing time: {optimized_result['processing_time']:.2f} seconds")
            logger.info(f"Average FPS: {fps:.2f}")
            logger.info(f"Total time (including download): {total_time:.2f} seconds")
            logger.info(f"Results saved to: {output_file}")
            
        finally:
            model_manager.clear_cache()
            
    finally:
        try:
            video_path.unlink()
            logger.info("Cleaned up temporary video file")
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 
