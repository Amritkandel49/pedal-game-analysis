from tracking import BallTracker, PlayerTracker
from utilities import read_video, save_video, KeypointCollector

def main():
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

    frames = read_video(VIDEO_PATH)
    frames_output = []
    player_tracker = PlayerTracker(PLAYER_DETECTION_MODEL, court_keypoints=court_keypoints)
    ball_tracker = BallTracker(BALL_DETECTION_MODEL)

    player_detections = player_tracker.detect_frames(frames, read_from_stub=True, stub_path="preloaded_detections/player_detections_stub.pkl")
    ball_detections = ball_tracker.detect_frames(frames, read_from_stub=True, stub_path="preloaded_detections/ball_detections_stub.pkl")
    

    # print("Player Detections:", player_detections)
    # print('\n', "-"*20, '\n')
    # print("Ball Detections:", ball_detections) 
    
    frames_output = player_tracker.draw_bounding_boxes(frames, player_detections)
    frames_output = ball_tracker.draw_bboxes(frames_output, ball_detections) 
    # frames_output = ball_tracker.draw_bboxes(frames, ball_detections) 
    
    save_video(frames_output, OUTPUT_PATH)

if __name__ == "__main__":
    main()