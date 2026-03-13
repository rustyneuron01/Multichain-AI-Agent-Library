#!/usr/bin/env python3
import sys
from pathlib import Path
# Add miner directory to Python path
miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

import cv2
import math
import numpy as np
import json
import argparse
from miner.sports.configs.soccer import SoccerPitchConfiguration
from collections import defaultdict

def preprocess_keypoints(keypoints):
    """
    Preprocess keypoints from the ground truth format to the expected format.
    
    Parameters:
    keypoints (list): Flat list of keypoints [x1, y1, x2, y2, ...]
    
    Returns:
    np.ndarray: Array of keypoint coordinates in the format [[x1, y1], [x2, y2], ...]
    """
    return np.array(keypoints).reshape(-1, 2)

def get_valid_keypoints(keypoints, image_width=None, image_height=None, min_dist=25):
    """
    Get valid keypoints (non-zero and optionally inside image bounds), with original indices.
    
    Parameters:
    keypoints (np.ndarray): Array of keypoint coordinates, shape (N, 2)
    image_width (int or None): Optional image width to clip to
    image_height (int or None): Optional image height to clip to
    
    Returns:
    valid_kps (np.ndarray): Filtered keypoints
    valid_indices (np.ndarray): Indices of valid keypoints in original array
    """
    mask = ~(np.all(keypoints == 0, axis=1))

    if image_width is not None and image_height is not None:
        in_bounds = (
            (keypoints[:,0] >= 0) & (keypoints[:,0] < image_width) &
            (keypoints[:,1] >= 0) & (keypoints[:,1] < image_height)
        )
        mask &= in_bounds

    valid_indices = np.where(mask)[0]
    valid_kps = keypoints[valid_indices]

    if min_dist is not None and valid_kps.shape[0] > 0:
        filtered_kps = []
        filtered_idx = []
        for kp, idx in zip(valid_kps, valid_indices):
            if all(np.linalg.norm(kp - prev) >= min_dist for prev in filtered_kps):
                filtered_kps.append(kp)
                filtered_idx.append(idx)
        valid_kps = np.array(filtered_kps, dtype=keypoints.dtype)
        valid_indices = np.array(filtered_idx, dtype=int)

    return valid_kps, valid_indices

def normalize_points(points):
    """
    Normalize points by subtracting the mean and scaling to unit standard deviation.
    
    Parameters:
    points (np.ndarray): Array of point coordinates.
    
    Returns:
    np.ndarray: Normalized points.
    np.ndarray: Normalization matrix.
    """
    mean = np.mean(points, axis=0)
    std = np.std(points)
    norm_points = (points - mean) / std
    norm_matrix = np.array([[1/std, 0, -mean[0]/std],
                            [0, 1/std, -mean[1]/std],
                            [0, 0, 1]])
    return norm_points, norm_matrix

def estimate_homography_ransac(keypoints, video_width, video_height, ransac_thresh_ratio=0.01):
    """
    Estimate a homography from keypoints in image coordinates to pitch vertices scaled to pixel coordinates using RANSAC.
    
    This function uses the vertices defined in the SoccerPitchConfiguration and scales them to match the image dimensions.
    
    Parameters:
      keypoints (np.ndarray): An (N, 2) array of detected keypoints in pixel coordinates.
      video_width (int): Width of the video frame in pixels.
      video_height (int): Height of the video frame in pixels.
      ransac_thresh_ratio (float): The reprojection threshold for RANSAC as a ratio of video width.
    
    Returns:
      H (np.ndarray or None): The 3x3 homography matrix if computed, else None.
      inlier_ratio (float): The ratio of inliers to total valid keypoints.
      avg_reprojection_error (float): The average reprojection error for all points in pixels.
    """
    # Load the soccer pitch configuration and extract vertices.
    pitch_config = SoccerPitchConfiguration()
    pitch_vertices = np.array(pitch_config.vertices, dtype=np.float32)

    # Scale pitch vertices to match image dimensions
    pitch_width = pitch_config.width
    pitch_height = pitch_config.length
    scale_x = video_width / pitch_width
    scale_y = video_height / pitch_height
    scaled_pitch_vertices = pitch_vertices * np.array([scale_x, scale_y])

    # Get valid keypoints and their indices
    valid_keypoints, valid_indices = get_valid_keypoints(keypoints, video_width, video_height)

    # Ensure at least 4 valid points are available.
    if valid_keypoints.shape[0] < 4:
        return None, 0.0, float('inf')

    # Match the scaled pitch vertices to the valid keypoints
    valid_pitch_vertices = scaled_pitch_vertices[valid_indices]

    # Calculate RANSAC threshold based on video width
    ransac_thresh = video_width * ransac_thresh_ratio
    ransac_thresh = max(ransac_thresh, 12)  # Ensure a minimum threshold of 12 pixels

    # Use OpenCV's RANSAC implementation to estimate the homography
    H, mask = cv2.findHomography(valid_keypoints, valid_pitch_vertices, cv2.RANSAC, ransac_thresh)
    
    if H is None or mask is None:
        return None, 0.0, float('inf')
    
    inlier_ratio = float(np.sum(mask)) / mask.size

    # Calculate reprojection errors in pixel space
    reprojected_points = cv2.perspectiveTransform(valid_keypoints.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(reprojected_points - valid_pitch_vertices, axis=1)
    avg_reprojection_error = np.mean(errors)

    return H, inlier_ratio, avg_reprojection_error

def detect_scene_transitions(frames, large_jump_threshold=20, transition_threshold=0.5):
    """
    Detect scene transitions (camera cuts, replays, close-ups) in the video.
    
    Parameters:
    frames (dict): Dictionary of frame data.
    large_jump_threshold (float): Threshold for considering a player jump as large.
    transition_threshold (float): Threshold ratio of players with large jumps to consider a scene transition.
    
    Returns:
    list: List of frame IDs where scene transitions occur.
    dict: Dictionary mapping frame IDs to segment IDs.
    """
    frame_ids = sorted([int(fid) for fid in frames.keys()])
    transitions = []
    frame_to_segment = {}
    current_segment = 0
    
    # Track player positions across frames
    player_positions = {}
    
    for i, frame_id in enumerate(frame_ids):
        frame_id_str = str(frame_id)
        frame_to_segment[frame_id_str] = current_segment
        
        # Skip first frame as we need a previous frame for comparison
        if i == 0:
            # Initialize player positions for first frame
            for obj in frames[frame_id_str].get("objects", []):
                if obj.get("class_id") == 2:  # Player
                    pid = obj["id"]
                    bbox = obj["bbox"]
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    player_positions[pid] = center
            continue
        
        # Count large jumps in current frame
        large_jumps = 0
        total_players = 0
        current_positions = {}
        
        for obj in frames[frame_id_str].get("objects", []):
            if obj.get("class_id") == 2:  # Player
                pid = obj["id"]
                bbox = obj["bbox"]
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                current_positions[pid] = center
                
                if pid in player_positions:
                    prev_center = player_positions[pid]
                    jump = np.linalg.norm(np.array(center) - np.array(prev_center))
                    
                    if jump > large_jump_threshold:
                        large_jumps += 1
                
                total_players += 1
        
        # Update player positions for next frame
        player_positions = current_positions
        
        # Check if this is a scene transition
        if total_players > 0 and large_jumps / total_players >= transition_threshold:
            transitions.append(frame_id)
            current_segment += 1
    
    return transitions, frame_to_segment

def check_keypoint_stability(frames, frame_to_segment, video_width, video_height):
    """
    Check if keypoints maintain stable relative positions on the pitch within segments.
    
    Parameters:
    frames (dict): Dictionary of frame data.
    frame_to_segment (dict): Dictionary mapping frame IDs to segment IDs.
    video_width (int): Width of the video frame in pixels.
    video_height (int): Height of the video frame in pixels.
    
    Returns:
    dict: Dictionary with segment stability scores.
    """
    # Group frames by segment
    segments = defaultdict(list)
    for frame_id, segment_id in frame_to_segment.items():
        segments[segment_id].append(frame_id)
    
    stability_scores = {}
    
    for segment_id, segment_frames in segments.items():
        if len(segment_frames) < 2:
            stability_scores[segment_id] = {
                'keypoint_stability': 1.0,  # Default score for single-frame segments
                'homography_stability': 1.0,
                'frames_analyzed': len(segment_frames)
            }
            continue
        
        # Calculate homography for each frame in the segment
        homographies = {}
        valid_keypoints_map = {}
        
        for frame_id in segment_frames:
            if 'keypoints' not in frames[frame_id]:
                continue
                
            keypoints = preprocess_keypoints(frames[frame_id]['keypoints'])
            H, inlier_ratio, avg_reprojection_error = estimate_homography_ransac(
                keypoints, video_width, video_height
            )
            
            if H is not None:
                homographies[frame_id] = H
                valid_keypoints, valid_indices = get_valid_keypoints(keypoints)
                valid_keypoints_map[frame_id] = (valid_keypoints, valid_indices)
        
        # Skip segments with fewer than 2 valid homographies
        if len(homographies) < 2:
            stability_scores[segment_id] = {
                'keypoint_stability': 0.5,  # Neutral score for segments with insufficient data
                'homography_stability': 0.5,
                'frames_analyzed': len(homographies)
            }
            continue
        
        # Calculate homography stability
        homography_diffs = []
        frame_ids = sorted(homographies.keys())
        
        for i in range(1, len(frame_ids)):
            prev_H = homographies[frame_ids[i-1]]
            curr_H = homographies[frame_ids[i]]
            
            # Calculate Frobenius norm of difference between normalized homographies
            norm_prev_H = prev_H / np.linalg.norm(prev_H)
            norm_curr_H = curr_H / np.linalg.norm(curr_H)
            diff = np.linalg.norm(norm_prev_H - norm_curr_H)
            homography_diffs.append(diff)
        
        avg_homography_diff = np.mean(homography_diffs)
        homography_stability = np.exp(-avg_homography_diff)  # Convert to 0-1 score (higher is better)
        
        # Calculate keypoint stability (relative positions)
        keypoint_diffs = []
        
        for i in range(1, len(frame_ids)):
            prev_frame_id = frame_ids[i-1]
            curr_frame_id = frame_ids[i]
            
            if prev_frame_id not in valid_keypoints_map or curr_frame_id not in valid_keypoints_map:
                continue
                
            prev_keypoints, prev_indices = valid_keypoints_map[prev_frame_id]
            curr_keypoints, curr_indices = valid_keypoints_map[curr_frame_id]
            
            # Find common keypoints
            common_indices = np.intersect1d(prev_indices, curr_indices)
            
            if len(common_indices) < 2:
                continue
                
            # Get positions of common keypoints
            prev_common = np.array([prev_keypoints[np.where(prev_indices == idx)[0][0]] for idx in common_indices])
            curr_common = np.array([curr_keypoints[np.where(curr_indices == idx)[0][0]] for idx in common_indices])
            
            # Calculate pairwise distances within each frame
            prev_distances = []
            curr_distances = []
            
            for i in range(len(common_indices)):
                for j in range(i+1, len(common_indices)):
                    prev_dist = np.linalg.norm(prev_common[i] - prev_common[j])
                    curr_dist = np.linalg.norm(curr_common[i] - curr_common[j])
                    prev_distances.append(prev_dist)
                    curr_distances.append(curr_dist)
            
            # Calculate relative difference in distances
            if len(prev_distances) > 0:
                prev_distances = np.array(prev_distances)
                curr_distances = np.array(curr_distances)
                rel_diffs = np.abs(prev_distances - curr_distances) / (prev_distances + 1e-6)
                avg_rel_diff = np.mean(rel_diffs)
                keypoint_diffs.append(avg_rel_diff)
        
        if len(keypoint_diffs) > 0:
            avg_keypoint_diff = np.mean(keypoint_diffs)
            keypoint_stability = np.exp(-10 * avg_keypoint_diff)  # Convert to 0-1 score (higher is better)
        else:
            keypoint_stability = 0.5  # Neutral score if no common keypoints
        
        stability_scores[segment_id] = {
            'keypoint_stability': keypoint_stability,
            'homography_stability': homography_stability,
            'frames_analyzed': len(homographies)
        }
    
    return stability_scores

def check_player_plausibility(frames, frame_to_segment, max_speed_pixels=30):
    """
    Check if player movements are physically plausible within continuous segments.
    
    Parameters:
    frames (dict): Dictionary of frame data.
    frame_to_segment (dict): Dictionary mapping frame IDs to segment IDs.
    max_speed_pixels (float): Maximum plausible player movement in pixels per frame.
    
    Returns:
    dict: Dictionary with segment plausibility scores.
    """
    # Group frames by segment
    segments = defaultdict(list)
    for frame_id, segment_id in frame_to_segment.items():
        segments[segment_id].append(frame_id)
    
    plausibility_scores = {}
    
    for segment_id, segment_frames in segments.items():
        # Convert to integers and sort
        segment_frames = sorted([int(fid) for fid in segment_frames])
        segment_frames = [str(fid) for fid in segment_frames]  # Convert back to strings
        
        if len(segment_frames) < 2:
            plausibility_scores[segment_id] = {
                'plausibility_score': 1.0,  # Default score for single-frame segments
                'implausible_movements': 0,
                'total_movements': 0
            }
            continue
        
        # Track player positions across frames
        player_positions = {}
        implausible_movements = 0
        total_movements = 0
        
        # Process frames in order
        for i in range(len(segment_frames)):
            frame_id = segment_frames[i]
            current_positions = {}
            
            # Get player positions in current frame
            for obj in frames[frame_id].get("objects", []):
                if obj.get("class_id") == 2:  # Player
                    pid = obj["id"]
                    bbox = obj["bbox"]
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    current_positions[pid] = center
                    
                    # Check movement if player was in previous frame
                    if i > 0 and pid in player_positions:
                        prev_center = player_positions[pid]
                        movement = np.linalg.norm(np.array(center) - np.array(prev_center))
                        
                        # Only count consecutive frames
                        prev_frame_id = segment_frames[i-1]
                        if int(frame_id) - int(prev_frame_id) <= 2:  # Allow for 1 frame gap
                            total_movements += 1
                            
                            # Lower the threshold to catch more implausible movements
                            if movement > max_speed_pixels:
                                implausible_movements += 1
                                print(f"Implausible movement detected: Player {pid} moved {movement:.2f} pixels between frames {prev_frame_id} and {frame_id}")
            
            # Update player positions for next frame
            player_positions = current_positions
        
        # Calculate plausibility score
        if total_movements > 0:
            plausibility_score = max(0, 1.0 - (implausible_movements / total_movements))
            print(f"Segment {segment_id}: {implausible_movements} implausible movements out of {total_movements} total movements")
        else:
            plausibility_score = 0.5  # Neutral score if no movements to analyze
            print(f"Segment {segment_id}: No player movements to analyze")
        
        plausibility_scores[segment_id] = {
            'plausibility_score': plausibility_score,
            'implausible_movements': implausible_movements,
            'total_movements': total_movements
        }
    
    return plausibility_scores

def score_player_positions(frames, small_jump_threshold=2, medium_jump_threshold=10):
    """
    Compute a consistency score based on the relative movement of players.
    Assumes each frame has an "objects" key, and each object has an "id", "bbox", and "class_id".
    Only considers objects with class_id == 2 (players).

    Returns:
        final_score (float): A score from 0 to 100 (higher is better).
        avg_jump (float): Average jump (in pixels) across all tracked players.
        total_jumps (int): Total number of jumps computed.
        biggest_jump (float): The largest jump observed.
        large_jumps (int): Number of large jumps (potential substitutions or view changes).
    """
    player_tracks = {}
    for frame_id, frame_data in frames.items():
        for obj in frame_data.get("objects", []):
            if obj.get("class_id") == 2:
                bbox = obj["bbox"]
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                pid = obj["id"]
                if pid not in player_tracks:
                    player_tracks[pid] = []
                player_tracks[pid].append((int(frame_id), center))
    
    for pid in player_tracks:
        player_tracks[pid].sort(key=lambda x: x[0])
    
    total_jump = 0
    jump_count = 0
    biggest_jump = 0
    large_jumps = 0
    small_jumps = 0
    medium_jumps = 0
    
    total_score = 0
    max_possible_score = 0

    for pid, track in player_tracks.items():
        for i in range(1, len(track)):
            _, center_prev = track[i-1]
            _, center_curr = track[i]
            jump = np.linalg.norm(np.array(center_curr) - np.array(center_prev))
            
            total_jump += jump
            jump_count += 1
            biggest_jump = max(biggest_jump, jump)
            
            if jump <= small_jump_threshold:
                small_jumps += 1
                total_score += 1  # Full score for small jumps
            elif jump <= medium_jump_threshold:
                medium_jumps += 1
                total_score += 0.5  # Half score for medium jumps (slight penalty)
            else:
                large_jumps += 1
                total_score += 1  # Full score for large jumps (flagged but not penalized)
            
            max_possible_score += 1  # Maximum score possible for each jump

    avg_jump = total_jump / jump_count if jump_count > 0 else 0
    final_score = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    print(f"Small jumps: {small_jumps}, Medium jumps: {medium_jumps}, Large jumps: {large_jumps}")
    
    return final_score, avg_jump, jump_count, biggest_jump, large_jumps

def calculate_keypoint_score(inlier_ratio, avg_reprojection_error):
    inlier_score = min(inlier_ratio * 100, 100)  # Scale to 0-100
    reprojection_score = max(0, 100 - avg_reprojection_error)  # Inverse scale, lower is better
    return (inlier_score + reprojection_score) / 2

def is_bbox_large_enough(bbox_dict, min_width=15, min_height=40) -> bool:
    x1, y1, x2, y2 = bbox_dict["bbox"]
    w, h = x2 - x1, y2 - y1
    if bbox_dict.get("class_id") == 0:  # FOOTBALL
        return True
    return w >= min_width and h >= min_height

def is_touching_scoreboard_zone(bbox_dict, frame_width=1280, frame_height=720) -> bool:
    x1, y1, x2, y2 = bbox_dict["bbox"]
    return not (x2 < 0 or x1 > frame_width or y2 < 0 or y1 > 150)

def filter_valid_bboxes_for_keypoints(objects: list, frame_width=1280, frame_height=720) -> list:
    """
    Filters bboxes based on minimum size and scoreboard exclusion zone.
    Same logic as in the bbox_clip evaluation pipeline.
    """
    return [
        obj for obj in objects
        if is_bbox_large_enough(obj) and not is_touching_scoreboard_zone(obj, frame_width, frame_height)
    ]

def keypoint_on_line(u, v, line_mask, tol=3):
    h, w = line_mask.shape
    x0, y0 = int(round(u)), int(round(v))
    x1, y1 = max(0, x0 - tol), max(0, y0 - tol)
    x2, y2 = min(w-1, x0 + tol), min(h-1, y0 + tol)
    return np.any(line_mask[y1:y2+1, x1:x2+1] > 0)


def filter_by_density(kept, radius=80, max_neighbors=80):
    mids = [((x1+x2)/2, (y1+y2)/2) for x1,y1,x2,y2 in kept]
    filtered = []
    for i, (x1,y1,x2,y2) in enumerate(kept):
        mx, my = mids[i]
        cnt = sum(1 for ux,uy in mids if abs(ux-mx)<radius and abs(uy-my)<radius)
        if cnt <= max_neighbors:
            filtered.append((x1,y1,x2,y2))
    return filtered

def line_on_mask(x1, y1, x2, y2, mask, samples=20):
    for t in np.linspace(0, 1, samples):
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        if mask[y, x] == 0:
            return False
    return True

def detect_pitch_lines_tophat(frame, border_ignore=3):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) basic grass mask
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35,40,40), (85,255,255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

    # 3) restriction to grass
    masked = cv2.bitwise_and(white_lines, grass)

    # 4) Canny + HoughP
    blur  = cv2.GaussianBlur(masked, (5,5), 0)
    edges = cv2.Canny(blur, 30, 100)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi/360,
                            threshold=30,
                            minLineLength=10,
                            maxLineGap=15)

    # 5) only grass segments
    line_mask  = np.zeros_like(gray)
    kept_lines = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if line_on_mask(x1,y1,x2,y2, grass, samples=30):
                cv2.line(line_mask, (x1,y1), (x2,y2), 255, 2)
                kept_lines.append((x1,y1,x2,y2))

    # 6) get rid of borders
    line_mask[:border_ignore, :] = 0
    line_mask[-border_ignore:, :] = 0
    line_mask[:, :border_ignore] = 0
    line_mask[:, -border_ignore:] = 0


    # 7) final list reconstruction
    final_kept = []
    for x1,y1,x2,y2 in kept_lines:
        mx, my = (x1+x2)//2, (y1+y2)//2
        if 0 <= mx < w and 0 <= my < h and line_mask[my,mx] > 0:
            final_kept.append((x1,y1,x2,y2))

    final_kept = filter_by_density(final_kept,
                                   radius=80,
                                   max_neighbors=80)

    return grass, white_lines, masked, edges, line_mask, final_kept


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    # vectors
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    # scalar projection w·v / |v|²
    t = (wx*vx + wy*vy) / float(vx*vx + vy*vy)
    t = max(0, min(1, t)) 
    # projected point
    projx = x1 + t*vx
    projy = y1 + t*vy
    # euclidian distance
    return np.hypot(px - projx, py - projy)

def keypoint_on_line_segments(u, v, segments, tol=3):
    for x1,y1,x2,y2 in segments:
        if point_to_segment_dist(u, v, x1, y1, x2, y2) <= tol:
            return True
    return False

def mean_keypoint_to_line_distance_score(keypoints, line_segments, video_width, video_height):
    """
    Gives a score between 0 and 1 based on average proximity to pitch lines
    """
    valid_kps, _ = get_valid_keypoints(keypoints, video_width, video_height)
    if len(valid_kps) == 0:
        return 0.5 

    if not line_segments:
        return 0
    norm_kps = valid_kps / np.array([video_width, video_height])
    norm_segs = [(
        x1 / video_width, y1 / video_height,
        x2 / video_width, y2 / video_height
    ) for (x1, y1, x2, y2) in line_segments]

    distances = []
    for (u, v) in norm_kps:
        min_dist = min(point_to_segment_dist(u, v, x1, y1, x2, y2) for (x1, y1, x2, y2) in norm_segs)
        distances.append(min_dist)

    mean_dist = np.mean(distances)
    k  = 0.0037
    x0 = 0.02
    score = 1.0 / (1.0 + np.exp((mean_dist - x0) / k))
    return score

def calculate_keypoint_to_player_scale_ratio(keypoints, frame_data, video_width, video_height):
    """
    Compare keypoints distance to object distance to see if it is logical

    Args:
        keypoints (np.ndarray): (N, 2)
        frame_data (dict): frame data

    Returns:
        float: ratio keypoint_scale / player_dispersion
    """
    valid_kps, _ = get_valid_keypoints(keypoints, video_width, video_height)
    if len(valid_kps) < 2:
        return 1.0 

    # avg distance btw keypoints
    dists_kps = []
    for i in range(len(valid_kps)):
        for j in range(i + 1, len(valid_kps)):
            dists_kps.append(np.linalg.norm(valid_kps[i] - valid_kps[j]))
    mean_kp_dist = np.mean(dists_kps)

    # get players centers
    player_centers = []
    valid_objects = filter_valid_bboxes_for_keypoints(
        frame_data.get("objects", []), video_width, video_height
    )
    for obj in valid_objects:
        bbox = obj["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        player_centers.append([cx, cy])

    player_centers = np.array(player_centers)
    if len(player_centers) < 2:
        return 1.0 

    # players dispertion
    dists_players = []
    for i in range(len(player_centers)):
        for j in range(i + 1, len(player_centers)):
            dists_players.append(np.linalg.norm(player_centers[i] - player_centers[j]))
    mean_player_dist = np.mean(dists_players)

    # Ratio
    scale_ratio = mean_kp_dist / (mean_player_dist + 1e-6)

    return scale_ratio

def fraction_objects_inside(frame_data, H, config: SoccerPitchConfiguration,video_width,video_height):

    # 1) exterior lines construction
    verts = np.array(config.vertices, dtype=np.float32)
    outer_idx = [0, 5, 29, 24] 
    contour = verts[outer_idx].reshape(-1,1,2).astype(np.int32)

    # 2) project and test
    inside, total = 0, 0
    valid_objects = filter_valid_bboxes_for_keypoints(
        frame_data.get("objects", []), video_width, video_height
    )
    for obj in valid_objects:
        x0,y0,x1,y1 = obj["bbox"]
        cx, cy = (x0+x1)/2, (y0+y1)/2
        pt_cm = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float32),
            H
        ).reshape(-1)
        if cv2.pointPolygonTest(contour, (pt_cm[0], pt_cm[1]), False) >= 0:
            inside += 1
        total += 1

    return inside / total if total else 0.0

def compute_scene_keypoint_consistency(frames, valid_frame_ids, video_width, video_height):
    """
    Compute per-frame keypoint consistency (normalized) in each scene.
    Returns scene_scores plus a length-weighted overall average.
    """
    if not valid_frame_ids:
        return {}, 1.0  # neutral if nothing to score

    # 1) normalize IDs to strings that actually exist in frames
    valid_ids_str = sorted(
        {str(fid) for fid in valid_frame_ids},
        key=lambda s: int(s)
    )
    if not valid_ids_str:
        return {}, 1.0

    # 2) break into consecutive scenes (work with ints for adjacency, keep str for indexing)
    valid_ids_int = [int(s) for s in valid_ids_str]
    scenes = []
    current = [valid_ids_int[0]]
    for prev, curr in zip(valid_ids_int, valid_ids_int[1:]):
        if curr == prev + 1:
            current.append(curr)
        else:
            scenes.append(current)
            current = [curr]
    scenes.append(current)

    scene_scores = {}
    total_len = 0
    weighted_sum = 0.0
    diag = math.hypot(video_width, video_height)

    for si, scene in enumerate(scenes):
        frame_scores = []
        frame_data = []

        for idx, fid_int in enumerate(scene):
            fid = str(fid_int)

            if idx == 0:
                frame_scores.append(1.0)
                frame_data.append((fid, 0, 1))
                continue

            prev_id = str(scene[idx - 1])
            if prev_id not in frames or fid not in frames:
                frame_scores.append(0.0)  
                frame_data.append((fid, 1, 0))
                continue
            
            previous_kpts = frames[prev_id]['keypoints']
            current_kpts = frames[fid]['keypoints']

            if (len(previous_kpts) != 32) or ((len(current_kpts) != 32)):
                frame_scores.append(0.0)
                frame_data.append((fid, 1, 0))
                continue

            # get valid keypoints + their original indices
            kp_prev, idx_prev = get_valid_keypoints(
                preprocess_keypoints(previous_kpts),
                video_width, video_height
            )
            kp_curr, idx_curr = get_valid_keypoints(
                preprocess_keypoints(current_kpts),
                video_width, video_height
            )

            # catch the case where no keyoints detected consecutively
            prev_all_zero = all(p == [0.0, 0.0] for p in previous_kpts)
            current_all_zero = all(p == [0.0, 0.0] for p in current_kpts)

            if prev_all_zero and current_all_zero:
                frame_scores.append(1)
                frame_data.append((fid, 0, 1))
                continue

            # if not all zero consecutively, then compare keypoints which are similar
            common = np.intersect1d(idx_prev, idx_curr)
            if (len(common) == 0):
                frame_scores.append(0)
                frame_data.append((fid, 1, 0))
                continue

            pts_prev = np.array([kp_prev[np.where(idx_prev == i)[0][0]] for i in common])
            pts_curr = np.array([kp_curr[np.where(idx_curr == i)[0][0]] for i in common])
            dists = np.linalg.norm(pts_curr - pts_prev, axis=1)
            avg_dist = float(np.mean(dists))

            norm_dist = avg_dist / diag
            k, x0 = 0.0037, 0.02
            frame_score = 1.0 / (1.0 + np.exp((norm_dist - x0) / k))
            frame_scores.append(frame_score)
            frame_data.append((fid, norm_dist, frame_score))

        avg_scene = float(np.mean(frame_scores)) if frame_scores else 1.0
        L = len(frame_scores)
        scene_scores[si] = {
            'frame_scores': frame_scores,
            'avg_score': avg_scene,
            'length': L,
            'frame_data': frame_data
        }
        total_len += L
        weighted_sum += avg_scene * L

    overall_weighted_avg = weighted_sum / total_len if total_len else 1.0
    return scene_scores, overall_weighted_avg

def process_input_file(input_file, video_path, video_width, video_height, frames_to_validate, pitch_lines_by_frame):
    if isinstance(input_file, str): 
        with open(input_file, 'r') as f:
            data = json.load(f)
            
    elif isinstance(input_file, dict):
        if "frames" not in input_file:
            data = {"frames": input_file}
        else:
            data = input_file
            
    elif isinstance(input_file, list):
        frames_dict = {}
        for i, frame_data in enumerate(input_file):
            if frame_data:
                # Use frame_number if available, otherwise use index
                frame_id = str(frame_data.get('frame_number', i))
                frames_dict[frame_id] = frame_data
        data={"frames": frames_dict}
        
    # Detect scene transitions and segment the video
    transitions, frame_to_segment = detect_scene_transitions(data['frames'])
    print(f"Detected {len(transitions)} scene transitions")
    print(f"Video segmented into {len(set(frame_to_segment.values()))} continuous segments")

    # Check keypoints consistency within valid pitch frames
    scene_scores, kpts_stability_weighted_avg = compute_scene_keypoint_consistency(data['frames'], frames_to_validate, video_width, video_height)
    # Check keypoint stability within segments
    stability_scores = check_keypoint_stability(data['frames'], frame_to_segment, video_width, video_height)
    
    # Check player movement plausibility within segments
    plausibility_scores = check_player_plausibility(data['frames'], frame_to_segment)
    
    # Process keypoints for each frame
    results = {}
    valid_frames = 0
    total_inlier_ratio = 0
    total_reprojection_error = 0
    total_keypoint_score = 0
    mean_list=[]
    mean_list_fracinside=[]
    mean_list_scale_ratio=[]
    valid_counter=0

    for frame_id, frame_data in data['frames'].items():
        keypoints = preprocess_keypoints(frame_data['keypoints'])
        valid_kps, _ = get_valid_keypoints(keypoints, video_width, video_height)

        H, inlier_ratio, avg_reprojection_error = estimate_homography_ransac(keypoints, video_width, video_height)
        
        if H is not None:
            valid_counter+=1

            bboxes = [obj['bbox'] for obj in frame_data.get("objects",[])]
            
            keypoint_score = calculate_keypoint_score(inlier_ratio, avg_reprojection_error)

            pitch_config = SoccerPitchConfiguration()
            final_kept = pitch_lines_by_frame.get(frame_id, [])
            frac_on_line = mean_keypoint_to_line_distance_score(valid_kps, final_kept, video_width, video_height)
            frac_inside = fraction_objects_inside(frame_data, H, pitch_config,video_width,video_height)
            scale_ratio = min(calculate_keypoint_to_player_scale_ratio(keypoints, frame_data, video_width, video_height),1.5)
            # Get segment-specific scores
            segment_id = frame_to_segment.get(frame_id, 0)
            segment_stability = stability_scores.get(segment_id, {'keypoint_stability': 1.0, 'homography_stability': 1.0})
            segment_plausibility = plausibility_scores.get(segment_id, {'plausibility_score': 1.0})
            results[frame_id] = {
                'inlier_ratio': float(inlier_ratio),
                'avg_reprojection_error': float(avg_reprojection_error),
                'keypoint_score': keypoint_score,
                'segment_id': segment_id,
                'frac_inside': frac_inside, 
                'frac_on_line': frac_on_line,
                'player_plausibility': segment_plausibility['plausibility_score'],
                'keypoint_stability': segment_stability['keypoint_stability']
            }
            valid_frames += 1
            total_inlier_ratio += inlier_ratio
            total_reprojection_error += avg_reprojection_error
            total_keypoint_score += keypoint_score
            mean_list.append(frac_on_line)
            mean_list_fracinside.append(frac_inside)
            mean_list_scale_ratio.append(scale_ratio)


    mean_on_line=sum(mean_list)/len(mean_list) if len(mean_list) else 0
    mean_inside = sum(mean_list_fracinside)/len(mean_list_fracinside) if len(mean_list_fracinside) else 0
    mean_scale = sum(mean_list_scale_ratio)/len(mean_list_scale_ratio) if len(mean_list_scale_ratio) else 0
    mean_scale = min(mean_scale,1)
    scale_valid =  np.clip(valid_counter / len(frames_to_validate) / 0.4, 0, 1)

    avg_inlier_ratio = total_inlier_ratio / valid_frames if valid_frames > 0 else 0
    avg_reprojection_error = total_reprojection_error / valid_frames if valid_frames > 0 else float('inf')
    avg_keypoint_score = total_keypoint_score / valid_frames if valid_frames > 0 else 0
    
    # Calculate player position consistency score
    player_score, avg_jump, total_jumps, biggest_jump, large_jumps = score_player_positions(data['frames'])
    
    # Calculate average stability and plausibility scores
    avg_keypoint_stability = np.mean([r['keypoint_stability'] for r in results.values()]) if results else 0
    avg_player_plausibility = np.mean([r['player_plausibility'] for r in results.values()]) if results else 0
    
    return results, valid_frames, avg_inlier_ratio, avg_reprojection_error, avg_keypoint_score, player_score, avg_jump, total_jumps, biggest_jump, large_jumps, transitions, avg_keypoint_stability, avg_player_plausibility, mean_on_line, mean_inside, mean_scale, scale_valid, scene_scores, kpts_stability_weighted_avg

def summarize_scores(results):
    inlier_ranges = {
        '0-10': 0, '11-20': 0, '21-30': 0, '31-40': 0, '41-50': 0,
        '51-60': 0, '61-70': 0, '71-80': 0, '81-90': 0, '91-100': 0
    }
    error_ranges = {
        '0-5': 0, '5-10': 0, '10-15': 0, '15-20': 0, '20-25': 0,
        '25-30': 0, '30-35': 0, '35-40': 0, '40-45': 0, '45+': 0
    }
    
    for frame_data in results.values():
        inlier_score = frame_data['inlier_ratio'] * 100
        error_score = frame_data['avg_reprojection_error']
        
        for range_key in inlier_ranges:
            low, high = map(int, range_key.split('-'))
            if low <= inlier_score <= high:
                inlier_ranges[range_key] += 1
                break
        
        for range_key in error_ranges:
            if range_key == '45+':
                if error_score >= 45:
                    error_ranges[range_key] += 1
            else:
                low, high = map(int, range_key.split('-'))
                if low <= error_score < high:
                    error_ranges[range_key] += 1
                    break
    
    return {'inlier_ratio': inlier_ranges, 'avg_reprojection_error': error_ranges}

def calculate_final_score_keypoints(keypoint_score, player_score, keypoint_stability, player_plausibility, mean_on_line , mean_inside, mean_scale,  scale_valid, kpts_stability_weighted_avg):

    if round(mean_on_line, 2) <= 0.2:
        return 0.0
    
    # Weight the different components
    weights = {
        'keypoint_score': 0.25,
        'mean_on_line': 0.4,
        'keypoint_stability': 0.05,
        'player_final_score': 0.1,
        'mean_inside': 0.2
    }
    
    # Scale all scores to 0-100
    
    keypoint_stability_score = keypoint_stability * 100
    player_plausibility_score = player_plausibility * 100
    point_on_line_score = mean_on_line * 100
    mean_inside = mean_inside * 100
    player_final_score = (player_plausibility_score + player_score) / 2
    # Print component scores for debugging
    print(f"\nComponent Scores:")
    print(f"Keypoint Score: {keypoint_score:.2f} (weight: {weights['keypoint_score']})")
    print(f"Point on line Score: {point_on_line_score:.2f} (weight: {weights['mean_on_line']})")
    print(f"Mean inside Score: {mean_inside:.2f} (weight: {weights['mean_inside']})")
    print(f"Keypoint Stability: {keypoint_stability_score:.2f} (weight: {weights['keypoint_stability']})")
    print(f"Player final score: {player_final_score:.2f} (weight: {weights['player_final_score']})")
        
        # Calculate weighted score
    final_score = (
        weights['keypoint_score'] * keypoint_score +
        weights['mean_on_line'] * point_on_line_score +
        weights['mean_inside'] * mean_inside +
        weights['keypoint_stability'] * keypoint_stability_score +
        weights['player_final_score'] * player_final_score
    )
    final_score *= kpts_stability_weighted_avg*((mean_scale + scale_valid)/2)
    print(f"\nFinal Score: {final_score:.2f}")
    return final_score

def main():
    parser = argparse.ArgumentParser(description="Process keypoints and estimate homography for each frame.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--video-width', type=int, default=1280, help="Width of the video frame in pixels")
    parser.add_argument('--video-height', type=int, default=720, help="Height of the video frame in pixels")
    args = parser.parse_args()

    results, valid_frames, avg_inlier_ratio, avg_reprojection_error, avg_keypoint_score, player_score, avg_jump, total_jumps, biggest_jump, large_jumps, transitions, avg_keypoint_stability, avg_homography_stability, avg_player_plausibility = process_input_file(args.input, args.video_width, args.video_height)
    
    # Summarize and print score ranges
    score_summary = summarize_scores(results)
    print("\nInlier Ratio Summary:")
    for range_key, count in score_summary['inlier_ratio'].items():
        print(f"{range_key}: {count} frames")
    
    print("\nAverage Reprojection Error Summary:")
    for range_key, count in score_summary['avg_reprojection_error'].items():
        print(f"{range_key}: {count} frames")
    
    print(f"\nTotal frames processed: {len(results)}")
    print(f"Valid frames (with 4 or more keypoints): {valid_frames}")
    print(f"Average Inlier Ratio: {avg_inlier_ratio:.2f}")
    print(f"Average Reprojection Error: {avg_reprojection_error:.2f} pixels")
    print(f"Keypoint Score: {avg_keypoint_score:.2f}")
    
    print(f"\nPlayer Position Consistency Score: {player_score:.2f}")
    print(f"Average Player Jump: {avg_jump:.2f} pixels")
    print(f"Biggest Player Jump: {biggest_jump:.2f} pixels")
    print(f"Total Player Jumps Analyzed: {total_jumps}")
    print(f"Large Jumps (potential substitutions or view changes): {large_jumps}")
    
    print(f"\nScene Transitions Detected: {len(transitions)}")
    print(f"Keypoint Stability Score: {avg_keypoint_stability:.2f}")
    print(f"Homography Stability Score: {avg_homography_stability:.2f}")
    print(f"Player Movement Plausibility Score: {avg_player_plausibility:.2f}")
    
    final_score = calculate_final_score_keypoints(
        avg_keypoint_score, 
        player_score, 
        avg_keypoint_stability, 
        avg_homography_stability, 
        avg_player_plausibility
    )
    print(f"\nFinal Score: {final_score:.2f}")

if __name__ == '__main__':
    main()
