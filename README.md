# Padel Game Analysis

A computer vision pipeline for analyzing padel matches. It detects and tracks players and the ball, identifies hit events, classifies the shot type (forehand, backhand, smash), and annotates the output video with that information. Per-frame data is also exported to CSV and JSON.

---

## Demo and Model Weights

- **Demo Video and Model Weights:** [Google Drive Link](https://drive.google.com/drive/folders/1UwzrqMDz7VlF3dVndEmRg2qrG9d5tM-b?usp=sharing)

---

## Pipeline Overview

```
Input Video
    |
    ├── Ball Detection & Tracking      (fine-tuned YOLOv5x)
    ├── Player Detection & Tracking    (YOLO11m + BoT-SORT)
    |
    ├── Hit Event Detection            (acceleration-peak analysis on ball trajectory)
    |
    ├── Pose Extraction at Hit Frames  (YOLO11m-pose, 30-frame window around each hit)
    |
    ├── Shot Classification            (LSTM: backhand / forehand / smash)
    |
    └── Output
            ├── Annotated video        (bboxes, ball, shot log panel)
            ├── frame_data.csv
            └── frame_data.json
```

---

## Techniques and Models Used

### Ball Tracking

- A **YOLOv5x** model fine-tuned on padel ball images detects the ball in each frame.
- After detection, raw positions are post-processed:
  - **Interpolation**: fills short gaps between two nearby detections.
  - **Extrapolation**: extends continuous motion segments forward and backward to cover frames where detection was lost.
- Ball center positions are stored per frame. Missing frames are marked as `None`.

### Player Detection and Tracking

- A **YOLO11m** model combined with **BoT-SORT** tracks players across frames using persistent IDs.
- Players are filtered to the court region using manually annotated court keypoints (perspective-aware x-bounds per y-level).
- To handle tracking ID switches, detections are post-processed:
  - The top-4 most-appearing track IDs are kept.
  - Frames where a valid ID was missed are reassigned to the nearest known player by bounding-box center distance.
- Players are then labelled by team based on their average vertical position relative to the court centerline: `player11`, `player12` (far side) and `player21`, `player22` (near side).

### Hit Event Detection

- The ball's x and y positions are smoothed with a rolling mean, then velocity and acceleration are computed frame-by-frame.
- Peaks in the acceleration magnitude signal (using `scipy.signal.find_peaks`) indicate sudden direction/speed changes — i.e., a hit.
- Each peak is validated by checking whether any player is within an expanded bounding-box radius of the ball position at that frame. Only valid hits are kept.

### Pose Extraction

- For each confirmed hit event, a 30-frame window (14 frames before and after the hit) is extracted from the video.
- The player crop for each frame in the window is fed into **YOLO11m-pose**, which returns 17 body keypoints (x, y) normalized to the crop size.

### Shot Classification (LSTM)

- A 2-layer **LSTM** network with dropout and a fully connected output head classifies each 30-frame keypoint sequence.
- Input: sequence of shape `(30, 34)` — 17 keypoints × 2 coordinates, flattened per frame.
- Output: one of three classes — `backhand`, `forehand`, `smash`.
- The model was trained on labeled padel match data using cross-entropy loss.

### Output Video Annotation

Frames are processed one at a time (streamed) to avoid holding all frames in memory:
- Player bounding boxes are drawn (green for team 1, red for team 2).
- Ball position is marked with a circle.
- A **side panel** (shot log table) is stitched to the right of the video showing Time, Player, and Shot Type for all shots seen so far. The most recent shot is highlighted.
- A brief on-screen label appears on the main video for 2 seconds after each shot.

### Data Export

For each frame, the following is written to `output/frame_data.csv` and `output/frame_data.json`:
- Frame index
- Ball x, y position
- Bounding box coordinates for each player
- Whether a shot was made in that frame, and if so, which player and shot type

---

## Project Structure

```
.
├── main.py                          # Entry point
├── tracking/
│   ├── ball_tracking_yolo.py        # Ball detection, interpolation, extrapolation
│   └── player_tracking.py           # Player detection, filtering, ID assignment
├── shot_classification/
│   ├── shot_classifier.py           # Hit detection, frame window extraction, pose extraction
│   └── lstm_classifier.py           # LSTM model definition and ShotPredictor
├── utilities/
│   ├── video_process.py             # read_video, save_video
│   ├── process_and_save_video.py    # Streaming frame processor (draw + save, one frame at a time)
│   ├── draw_shot_overlay.py         # Side panel and on-screen shot label rendering
│   ├── export_frame_data.py         # CSV and JSON export of per-frame data
│   ├── key_point_collector.py       # Interactive court keypoint annotation tool
│   ├── player_detection_processing.py
│   └── measurements.py
├── TrackNet/                        # Alternative ball tracking approach (TrackNet-based)
├── model_weights/                   # Fine-tuned YOLO and LSTM weights (not in repo, see Drive link)
├── preloaded_detections/            # Cached pickle stubs for player and ball detections
├── input/                           # Input videos
└── output/                          # Output video, CSV, JSON
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place model weights in the `model_weights/` directory. See the Drive link above for weights and demo video.

---

## Running

```bash
python main.py
```

Output is saved to `output/output_video.mp4`, `output/frame_data.csv`, and `output/frame_data.json`.

To run without preloaded stubs (fresh detection), set `read_from_stub=False` in `main.py`.

---

## Court Keypoints

Court keypoints define the perspective boundaries of the court and are used to filter player detections to the valid playing area. They are hardcoded for the current test video. To collect keypoints for a new video, uncomment the `KeypointCollector` line in `main.py`.
