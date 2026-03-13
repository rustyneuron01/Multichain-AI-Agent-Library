import torch
import open_clip
from PIL import Image
import cv2
import numpy as np

# Global variables
_clip_model = None
_clip_preprocess = None
_text_features = None
_texts = ["a football pitch", "a close-up of a football player"]

def init_clip_model():
    """Clip to be called in the subprocess"""
    global _clip_model, _clip_preprocess, _text_features

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    device = torch.device("cpu")
    clip_model.to(device)
    clip_model.eval()

    with torch.no_grad():
        text_tokens = tokenizer(_texts).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    _clip_model = clip_model
    _clip_preprocess = clip_preprocess
    _text_features = text_features

def batch_clip_verification(image_paths, threshold=0.7):
    global _clip_model, _clip_preprocess, _text_features, _texts

    if _clip_model is None or _clip_preprocess is None or _text_features is None:
        raise RuntimeError("CLIP model not initialized. Call init_clip_model() first.")

    image_features_list = []
    valid_paths = []

    for path in image_paths:
        try:
            image = _clip_preprocess(Image.open(path)).unsqueeze(0)
            image_features_list.append(image)
            valid_paths.append(path)
        except:
            continue

    if not image_features_list:
        return {}

    batch = torch.cat(image_features_list).to("cpu")

    with torch.no_grad():
        image_features = _clip_model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ _text_features.T).softmax(dim=-1)

    clip_scores = {}
    for i, path in enumerate(valid_paths):
        top_prob, top_label = similarity[i].max(0)
        if _texts[top_label] == "a football pitch":
            clip_scores[path] = top_prob.item()
        else:
            clip_scores[path] = 0.0

    return clip_scores

def is_close_plan(mask_green, threshold=0.8, band_ratio=0.025):
    """
    Detects if the image is a close-up based on green coverage on image borders.
    """
    h, w = mask_green.shape
    band_h = int(h * band_ratio)
    band_w = int(w * band_ratio)

    # Extract borders
    top = mask_green[:band_h, :]
    bottom = mask_green[-band_h:, :]
    left = mask_green[:, :band_w]
    right = mask_green[:, -band_w:]

    # Compute green ratios
    green_ratios = [
        np.sum(top > 0) / top.size,
        np.sum(bottom > 0) / bottom.size,
        np.sum(left > 0) / left.size,
        np.sum(right > 0) / right.size
    ]

    if all(ratio > threshold for ratio in green_ratios):
        return True
    return False

def detect_goal_net_by_lines(lines):
    if lines is None or len(lines) < 15:
        return False

    vertical_lines = 0
    horizontal_lines = 0
    line_lengths = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
        length = np.linalg.norm([x2 - x1, y2 - y1])

        if angle < 10:
            horizontal_lines += 1
        elif angle > 80:
            vertical_lines += 1

        line_lengths.append(length)

    avg_length = np.mean(line_lengths)
    std_length = np.std(line_lengths)

    is_grid = vertical_lines > 15 and horizontal_lines > 15
    is_short_lines = avg_length < 50
    is_uniform_length = std_length < 10

    return (is_grid and is_short_lines) or (is_grid and is_uniform_length)

def detect_pitch(image_path, clip_scores=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((10, 10), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            mask_obj = np.zeros_like(mask_green)
            cv2.drawContours(mask_obj, [cnt], -1, 255, thickness=cv2.FILLED)

            green_pixels = np.sum((mask_green > 0) & (mask_obj > 0))
            total_pixels = np.sum(mask_obj > 0)

            green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
            if green_ratio < 0.5:
                cv2.drawContours(mask_cleaned, [cnt], -1, 0, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_cleaned)
    edges = cv2.Canny(masked_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/360, threshold=50, minLineLength=50, maxLineGap=8)

    if detect_goal_net_by_lines(lines):
        lines = None

    green_ratio = np.sum(mask_cleaned > 0) / mask_cleaned.size
    total_line_length = sum(np.linalg.norm([x2 - x1, y2 - y1]) for x1, y1, x2, y2 in lines[:, 0]) if lines is not None else 0

    score = 0.3 * green_ratio + 0.7 * (total_line_length / 4500)
    score = min(1, score)

    if is_close_plan(mask_green, threshold=0.8):
        score -= 0.29

    if 0.7 <= score <= 1.0 and clip_scores is not None:
        clip_score = clip_scores.get(image_path, 0.0)
        return 1 if clip_score >= 0.75 else 0

    return max(score, 0.0)
