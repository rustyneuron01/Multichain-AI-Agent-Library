# Score Vision — Real-Time Football Game State Recognition

**Score Vision** is a computer vision system for real-time football video analysis. It reduces the cost and time of manual game-state annotation (player/ball/referee tracking, pitch keypoints) by an order of magnitude, targeting sports analytics, broadcast, and data providers. The pipeline is designed to process multiple HD video streams in parallel via distributed inference nodes, with lightweight validation to keep quality high while minimizing compute overhead.

---

## Sample output

The pipeline takes a video stream and returns per-frame detections and tracking (players, ball, referee, goalkeeper, pitch keypoints). Example output on a football clip:

<video src="64aacaf72b1948b88a61982970d81c-10M.mp4" controls width="640" muted></video>

_Sample video: [64aacaf72b1948b88a61982970d81c-10M.mp4](64aacaf72b1948b88a61982970d81c-10M.mp4)_

---

## Overview

The framework combines **detection and tracking** (YOLO, HRNet, OSNet, ByteTrack) with **lightweight validation** (CLIP and keypoint checks) so that complex video analysis can run at scale without re-running full inference for verification. The result is fast, cost-effective game-state recognition suitable for live or batch processing.

### Why football?

Football is a demanding domain: high-stakes, real-time decisions, crowded scenes, and varying camera angles. It is an ideal testbed for robust detection and tracking. The same pipeline design extends to other sports and general video analytics.

### Market context

Manual video annotation in sports can cost **$10–55 per minute**, with complex scenarios requiring up to **four hours of human labelling per minute** of footage. Score Vision aims to cut these costs by **10× to 100×** while improving speed and accuracy, serving clubs, broadcasters, betting operators, and analytics providers.

---

## Technologies

| Category                 | Technologies                                                                                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Detection & tracking** | YOLO (Ultralytics), HRNet, OSNet — custom-trained on football datasets for player, goalkeeper, referee, and ball detection; ByteTrack for multi-object tracking |
| **Pitch / keypoints**    | YOLO-based pitch detection; keypoint estimation for field geometry                                                                                              |
| **Backend & API**        | Python, FastAPI, Uvicorn; async frame streaming; request verification and concurrency control                                                                   |
| **Video & inference**    | OpenCV, supervision; async video download and frame-by-frame streaming; GPU (CUDA) / Apple Silicon (MPS) support                                                |
| **Validation**           | CLIP and VLM-based checks for bounding-box semantics; keypoint stability and reprojection scoring                                                               |
| **Infrastructure**       | Hugging Face (model hosting); SQLite for challenge/response storage; PM2/Docker for deployment                                                                  |

---

## Model choices

- **YOLO:** Single-stage detection suited to real-time video; strong accuracy/speed trade-off for players and ball; tunable confidence and IoU for production.
- **HRNet:** High-resolution feature maps across scales → better localization and keypoint/pose-style outputs; helps in crowded scenes and with small objects (e.g. ball, distant players).
- **OSNet:** Lightweight person re-identification; improves identity consistency across frames when combined with ByteTrack, reducing ID switches in multi-player tracking.
- **Custom training:** Models are trained on football-specific data (players, referees, goalkeepers, ball, pitch) to maximize accuracy and robustness to camera angles, lighting, and occlusions typical in match footage.

---

## Inference pipeline

1. **Ingest:** Receive video URL → download (with retries/timeouts) → validate file (OpenCV).
2. **Frame stream:** Stream frames asynchronously (OpenCV `VideoCapture`) with device-specific timeouts to avoid stalls on CPU/GPU.
3. **Per-frame inference:**
   - **Pitch:** YOLO pitch model → extract pitch keypoints (field lines, corners) for downstream geometry.
   - **Players / ball / referee / goalkeeper:** YOLO (and, where used, HRNet/OSNet) detection → confidence threshold and NMS → ByteTrack to assign stable IDs across frames.
   - **Output:** Per-frame list of objects (bbox, class_id, tracker_id) and keypoints; coordinates rounded for compact JSON.
4. **Response:** Aggregate frames + processing time; optional signing/verification for integration with distributed or API-driven systems.

---

## Lightweight validation

Validation ensures output quality without re-running full inference:

1. **Frame filtering & keypoint validation**  
   Frames are filtered using pitch detection. A global scoring system evaluates keypoint accuracy (stability, plausibility, reprojection error).

2. **Semantic bounding-box assessment**  
   Selected frames are checked with CLIP-based object verification (players, ball, referees, goalkeepers). The result is a confidence-weighted quality score.

Scores are combined and normalized (0–1) for quality control and optional use in ranking or routing.

---

## Main technical challenge: duplicate detections

**Problem:** During bounding-box inference, the same player or object was often detected multiple times in overlapping boxes, leading to duplicate detections and noisy tracking.

**Approach:**

- **Confidence (score) threshold:** Only high-confidence detections are kept, reducing duplicates while retaining true positives.
- **Typical range:** Confidence (ROI score) is set between **0.7 and 0.9** in production; the exact value is chosen per model and dataset from validation metrics (precision/recall, tracking quality).
- **Trade-off:** Lower thresholds increase recall but add duplicate detections; higher thresholds reduce duplicates but can drop valid detections under occlusion or motion blur. Threshold choice depends on model training output (e.g. confidence distribution and calibration).

---

## Performance metrics

**Game State Higher Order Tracking Accuracy (GS-HOTA)** is used for quality assessment:

- **GS-HOTA = √(Detection × Association)**
- **Detection:** Object detection accuracy.
- **Association:** Tracking consistency across frames.

Evaluation covers detection and tracking accuracy, consistency over time, and response latency. The system is designed to support quality-weighted scoring and volume-based contribution metrics where the pipeline is run in a distributed or multi-instance setup.

---

## Architecture (high level)

- **Inference nodes:** Receive video streams (or URLs), run the detection-and-tracking pipeline per frame, and return structured JSON (objects, keypoints, IDs).
- **Validation layer:** Optional lightweight checks (CLIP, keypoint scoring) on a subset of frames to verify quality without full re-inference.
- **API & storage:** FastAPI for ingestion and responses; SQLite or similar for challenge/response and state; optional Hugging Face for model hosting.

No blockchain or token terminology is required to run or integrate the pipeline; it can be deployed as a standard microservice or multi-instance API.

---

## Quick start

1. Ensure system requirements (Python 3.10+, GPU recommended for real-time inference; see `min_compute.yaml` for reference specs).
2. Clone the repository and install dependencies (see setup guides in the repo).
3. Configure environment (API keys, model paths, device).
4. Run the inference service and send video URLs or streams to the API.

Detailed setup for **inference nodes** (running the detection pipeline) and **validation / quality-check services** is in the documentation:

- [Inference node setup](miner/README.md)
- [Validation / quality-check setup](validator/README.md)

_(These guides may use internal role names; functionally they describe “run the detector” and “run the validator” services.)_

---

## Roadmap

- **Current:** Game State Recognition pipeline, VLM-based validation, benchmarking.
- **Planned:** Human-in-the-loop validation, additional footage types (e.g. grassroots), dashboard and leaderboard.
- **Future:** Action spotting, match event captioning, advanced player tracking, integration APIs, adaptation to other sports, developer tools and SDKs.

---

## Research

The paper **["Score Vision: Enabling Complex Computer Vision Through Lightweight Validation - A Game State Recognition Framework for Live Football"](https://drive.google.com/file/d/1oADURxxIZK0mTEqJPDudgXypohtFNkON/view)** describes the lightweight validation approach and its role in reducing computational overhead while maintaining accuracy at scale.

---

## Contributing

Contributions are welcome. See [Contributing Guidelines](CONTRIBUTING.md) for code style, pull request process, and testing.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
