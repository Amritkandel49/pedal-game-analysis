# Approach Explanation

## Methodology

### Problem Understanding

The input video presented unique challenges compared to standard padel footage available online. The camera was positioned at a high angle with a wide-angle lens, capturing multiple courts in a single frame. This made isolating the playing court, detecting the ball, and identifying only the four players on the target court significantly more difficult.

The foundation of the entire system is accurate ball and player detection. Every downstream step — hit detection, player identification at hit time, and shot classification — depends directly on the quality of tracking. This guided all design decisions throughout the project.

### Court Isolation and Keypoint Collection

Since multiple courts appeared in the frame, it was necessary to define the boundaries of the target court. Court keypoints (corner and service line intersections) were used to compute perspective-aware x-boundaries at any given y-level. This allowed player detections to be filtered spatially, keeping only detections that fell within the realistic court region.

Because the wide-angle lens introduced curved lines, automated keypoint detection models trained on standard court footage were not applicable. The camera angle was fixed throughout the video, so keypoints were collected once manually from the first frame of the video using an interactive tool.

### Ball Detection and Tracking

A YOLOv5x model pretrained on general objects was finetuned on a padel ball dataset. Finetuning was necessary because the ball is a small, fast-moving object that the pretrained model failed to detect in almost every frame.

A specific problem in the given video was the presence of spare balls resting beneath the center net. Since the training dataset for padel ball detection contained only one ball per frame (as is typical in match footage), the finetuned model learned to detect a single ball. The stationary spare balls were consistently detected, while the actual playing ball, being erratic due to motion, was not. To address this, detections with no significant positional change between consecutive frames were excluded, and only detections showing movement were retained for tracking.

Following detection, gaps in ball positions were filled using two strategies:
- **Interpolation**: missing positions between two nearby detections were linearly filled if the spatial gap was within a threshold.
- **Extrapolation**: segments of continuous motion were extended forward and backward to recover frames where tracking was lost.

An alternative ball tracking approach, TrackNet (a VGGNet-based model achieving ~90% precision), was explored but could not be used due to hardware constraints.

### Player Detection and Tracking

A YOLO11m model was finetuned for person detection and combined with BoT-SORT for multi-object tracking across frames. Players were filtered to the court region using the court keypoints and then post-processed to handle tracking ID switches:

- The four most consistently appearing track IDs were retained.
- Frames where a valid player was missed were reassigned to the closest known player by bounding-box center distance.

Since padel is played by two teams of two on opposite sides of the net, players were assigned to teams based on their average vertical position relative to the court centerline. Players above the centerline were assigned to team 1 (far side), and those below to team 2 (near side), yielding labels `player11`, `player12`, `player21`, `player22`.

### Hit Detection

Once consistent ball and player positions were available, hit events were detected by analyzing the ball's motion signal. The ball's x and y positions were smoothed using a rolling mean, then velocity and acceleration were computed frame-by-frame.

An initial plan was to detect direction reversals in the ball's trajectory as indicators of a hit. This required smooth, continuous ball tracking across all frames, which was not achievable given the detection gaps. The approach was revised to use peak detection on the acceleration magnitude signal. Sharp spikes in acceleration correspond to sudden changes in ball speed or direction, which occur at the moment of a hit.

Each detected peak was validated by checking whether any player was within an expanded bounding-box radius of the ball. If so, the nearest player was assigned as the hitter.

### Shot Classification

For each confirmed hit event, a 30-frame window centered on the hit frame was extracted from the video. The player crop for each frame in the window was passed through a YOLO11m-pose model to extract 17 body keypoints (x, y), normalized to the crop dimensions.

These 30-frame keypoint sequences (shape: 30 × 34 after flattening) were fed into a 2-layer LSTM classifier with dropout, producing one of three labels: **backhand**, **forehand**, or **smash**.

A single snapshot of the player's pose is insufficient for shot classification because different shots can look similar at any one moment. The temporal context of the swing — before and after contact — is what distinguishes shot types. This is why a sequence model was used rather than a frame-level classifier.

### Output

For each frame, the pipeline writes player bounding boxes, ball position, hit flag, player involved, and shot type to `output/frame_data.csv` and `output/frame_data.json`. The annotated output video includes bounding boxes, a ball marker, an on-screen shot label at the moment of each hit, and a persistent shot log panel alongside the video.

---

## Challenges Faced

**Ball detection with multiple balls in frame.** The spare balls resting under the net were stationary and easy for the detector to pick up, while the fast-moving playing ball was frequently missed. Because the training data had only one ball per frame, the model was not equipped to handle this scenario. Filtering by positional change resolved the issue, but it added complexity to the post-processing.


**Curved court lines from wide-angle lens.** Standard court keypoint detection models trained on conventional footage did not generalize to the curved lines in this video. Manual keypoint collection was used as a practical workaround given the fixed camera position.

**Sparse and incomplete ball tracking.** Despite interpolation and extrapolation, the ball was missing in many frames, which made the initial hit detection approach (based on direction reversal) unreliable. The pivot to acceleration-peak detection was more robust to gaps, but accuracy still depends on the quality of the smoothed trajectory.

**Limited padel-specific pose data for shot classification.** Padel and tennis share similarities but differ in shot types and player body mechanics. The underarm serve and wall-rebound shots in padel produce player poses that do not directly correspond to tennis equivalents. The LSTM model was trained on tennis keypoint data due to the unavailability of a labeled padel shot dataset, which limits classification accuracy.

---

## Improvements

**Shot classification training data.** The most impactful improvement would be to collect a padel-specific keypoint dataset for shot classification. This does not require manual labeling of images; the keypoints can be extracted automatically using YOLO-pose on match footage, and shots can be labeled from match commentary or manual annotation of clip timestamps. Retraining the LSTM on padel-specific data would substantially improve classification accuracy for backhand, forehand, and smash.

**Ball tracking reliability.** Incorporating a dedicated ball tracking model such as TrackNet, which is designed for fast, small ball tracking in racket sports, would improve the underlying trajectory quality. Better ball tracking would reduce the number of missing frames and improve hit detection precision.

**Automated court keypoint detection.** Training a keypoint detection model on wide-angle padel footage with curved court lines would remove the need for manual annotation and make the pipeline fully automated for new input videos.
