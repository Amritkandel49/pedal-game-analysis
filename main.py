from tracking import BallTracker, PlayerTracker
from utilities import read_video, save_video

def main():
    VIDEO_PATH = "input/input_video_shortest.mp4"
    OUTPUT_PATH = "output/output_video.mp4"
    PLAYER_DETECTION_MODEL = "model_weights/person_tracking_best_weight.pt"
    BALL_DETECTION_MODEL = "model_weights/yolov5x_best.pt"

    frames = read_video(VIDEO_PATH)
    frames_output = []
    player_tracker = PlayerTracker(PLAYER_DETECTION_MODEL)
    ball_tracker = BallTracker(BALL_DETECTION_MODEL)

    player_detections = player_tracker.detect_frames(frames)
    ball_detections = ball_tracker.detect_frames(frames)

    print("Player Detections:", player_detections)
    print("Ball Detections:", ball_detections) 
    
    frames_output = player_tracker.draw_bounding_boxes(frames, player_detections)
    frames_output = ball_tracker.draw_bboxes(frames_output, ball_detections)
    
    save_video(frames_output, OUTPUT_PATH)

if __name__ == "__main__":
    main()