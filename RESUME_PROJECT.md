# Score Vision — Resume Project Description

**Use this text for your resume. It avoids mentioning subnet, Bittensor, or blockchain. Focus is on computer vision, models, and engineering.**

---

## Project name (suggested)

**Score Vision — Decentralized Sports Video Analytics**  
*Or:* **Real-Time Football Game State Recognition (GSR) Pipeline**

---

## Project purpose (1–2 lines)

- Built a **decentralized computer vision system** for real-time football video analysis that reduces the cost and time of manual game-state annotation (e.g. player/ball/referee tracking, pitch keypoints) by an order of magnitude, targeting sports analytics, broadcast, and data providers.
- Designed to process multiple HD video streams in parallel via distributed inference nodes, with lightweight validation to keep quality high while minimizing compute overhead.

---

## Technologies used

| Category | Technologies |
|----------|--------------|
| **Detection & tracking** | YOLO (Ultralytics), HRNet, OSNet — custom-trained on football datasets for player, goalkeeper, referee, and ball detection; ByteTrack for multi-object tracking |
| **Pitch / keypoints** | YOLO-based pitch detection; keypoint estimation for field geometry |
| **Backend & API** | Python, FastAPI, Uvicorn; async frame streaming; request verification and concurrency control |
| **Video & inference** | OpenCV, supervision; async video download and frame-by-frame streaming; GPU (CUDA) / Apple Silicon (MPS) support |
| **Validation (optional to mention)** | CLIP and VLM-based checks for bounding-box semantics; keypoint stability and reprojection scoring |
| **Infrastructure** | Hugging Face (model hosting); SQLite for challenge/response storage; PM2/Docker for deployment |

---

## Model choices and advantages

- **YOLO:** Fast, single-stage detection suitable for real-time video; strong accuracy/speed trade-off for players and ball; easy to tune confidence and IoU for production.
- **HRNet:** High-resolution feature maps across scales → better localization and keypoint/pose-style outputs; helps with crowded scenes and small objects (e.g. ball, distant players).
- **OSNet:** Lightweight person re-identification and feature extraction; improves identity consistency across frames when combined with a tracker (e.g. ByteTrack), reducing ID switches in multi-player tracking.
- **Custom training on domain datasets:** All models trained on football-specific data (players, referees, goalkeepers, ball, pitch) to maximize accuracy and robustness to camera angles, lighting, and occlusions typical in match footage.

---

## Inference pipeline (how inference works)

1. **Ingest:** Receive video URL → download (with retries/timeouts) → validate file (OpenCV).
2. **Frame stream:** Stream frames asynchronously (OpenCV `VideoCapture`) with device-specific timeouts to avoid stalls on CPU/GPU.
3. **Per-frame inference:**  
   - **Pitch:** YOLO pitch model → extract pitch keypoints (e.g. field lines, corners) for downstream geometry.  
   - **Players/ball/referee/goalkeeper:** YOLO (and, where used, HRNet/OSNet) detection → confidence threshold and NMS applied (see *Challenge* below) → ByteTrack to assign stable IDs across frames.  
   - **Output:** Per-frame list of objects (bbox, class_id, tracker_id) and keypoints; coordinates rounded for compact JSON.
4. **Response:** Aggregate frames + processing time; optional signing/verification for integration with the distributed system.

---

## Main technical challenge and solution

**Challenge:** During bounding-box model inference, the same player or object was often detected multiple times in overlapping boxes, leading to duplicate detections and noisy tracking.

**Approach:**  
- Tuned the **confidence (score) threshold** so that only high-confidence detections are kept, significantly reducing duplicates while retaining true positives.  
- In production, the **confidence (ROI score) range is typically set between 0.7 and 0.9**; the exact value is chosen per model and dataset based on validation metrics (precision/recall, downstream tracking quality).  
- Threshold choice **depends on model training output** (e.g. confidence distribution and calibration); lower thresholds increase recall but duplicate detections, higher thresholds reduce duplicates but can drop valid detections in occlusion or motion blur.

---

## Short resume bullet ideas (copy-paste style)

- Developed a **real-time football video analysis pipeline** for game-state recognition (player/ball/referee tracking, pitch keypoints) using **YOLO, HRNet, and OSNet** custom-trained on football datasets; deployed in a distributed, API-driven architecture with FastAPI and async video streaming.
- Designed and tuned **confidence thresholds (0.7–0.9)** for detection models to eliminate duplicate bounding boxes while preserving recall; threshold selection driven by per-model validation and training output.
- Implemented **end-to-end inference**: async video download → frame streaming (OpenCV) → multi-model detection (YOLO + ByteTrack) → structured JSON output; supported CUDA and MPS for production and development.
- Built **lightweight validation** (CLIP + keypoint checks) to verify detection quality at scale without running full re-inference, enabling cost-effective quality control across many nodes.

---

*You can trim or merge bullets to fit your resume length. Replace "distributed" / "many nodes" with "microservices" or "multi-instance" if you prefer to avoid any decentralized connotation.*
