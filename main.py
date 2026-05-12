from tracking import BallTracker, PlayerTracker
from utilities import save_video, KeypointCollector, export_frame_data, process_and_save_video
from shot_classification import ShotClassifier, LSTMClassifier, ShotPredictor
import pandas as pd
import cv2
import json

def main():
    POSE_EXTRACTION_MODEL = 'yolo11m-pose.pt'
    VIDEO_PATH = "input/input_video_shortest.mp4"
    OUTPUT_PATH = "output/output_video.mp4"
    # PLAYER_DETECTION_MODEL = "model_weights/person_tracking_best_weight.pt"
    PLAYER_DETECTION_MODEL = "yolo11m.pt"
    BALL_DETECTION_MODEL = "model_weights/yolov5x_best.pt"
    # BALL_DETECTION_MODEL = "yolov5x.pt"
    
    court_keypoints = {
        'p1': (670, 119), 'p2': (1230, 106), 'p3': (626, 161), 'p4': (946, 142), 'p5': (1275, 146), 'p6': (484, 319), 'p7': (1420, 293), 'p8': (228, 714), 'p9': (959, 730), 'p10': (1703, 686), 'p11': (122, 999), 'p12': (1832, 965)
    } # for test only 
    
    # court keypoints collection
    
    # court_keypoints = KeypointCollector(VIDEO_PATH).collect_keypoints()
    # print("Collected Court Keypoints:", court_keypoints)

    frames_output = []
    player_tracker = PlayerTracker(PLAYER_DETECTION_MODEL, court_keypoints=court_keypoints)
    ball_tracker = BallTracker(BALL_DETECTION_MODEL)

    # frames=[] is safe here because read_from_stub=True skips frame iteration
    player_detections = player_tracker.detect_frames([], read_from_stub=True, stub_path="preloaded_detections/player_detections_stub.pkl")
    ball_detections = ball_tracker.detect_frames([], read_from_stub=True, stub_path="preloaded_detections/ball_detections_stub.pkl")
    
    shot_classifier = ShotClassifier(POSE_EXTRACTION_MODEL, VIDEO_PATH, ball_detections, player_detections)

    hit_events = shot_classifier.detect_hits()
    
    # window_size=14 combined with our new logic gives exactly 30 frames
    shot_frame_sequences = shot_classifier.get_hit_frame_sequences_for_player(hit_events, window_size=14)
    
    sequence_keypoints_all_shots = shot_classifier.pose_extraction_for_all_hit_events(shot_frame_sequences)
    print(f"Extracted keypoint sequences for {len(sequence_keypoints_all_shots)} hit events.\n")
    print("Sample extracted sequence (first event):")
    if len(sequence_keypoints_all_shots) > 0:
        print(sequence_keypoints_all_shots[0]['sequence_keypoints'])
    else:
        print("No sequences extracted.")
   
    
    predictor = ShotPredictor(model_weight_path='model_weights/best_lstm_model_fg.pth')
    predictions = predictor.predict_shots(sequence_keypoints_all_shots)
    
    print("\n--- SHOT CLASSIFICATION RESULTS ---")
    for pred in predictions:
        print(f"Frame {pred['hit_frame']} | Player {pred['player_id']} | Type: {pred['shot_type'].upper()}")
    
    export_frame_data(
        ball_detections=ball_detections,
        player_detections=player_detections,
        predictions=predictions,
        output_csv="output/frame_data.csv",
        output_json="output/frame_data.json"
    )

    # Process one frame at a time — no full frame list in memory
    process_and_save_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        player_detections=player_detections,
        ball_detections=ball_detections,
        predictions=predictions
    )

if __name__ == "__main__":
    main()