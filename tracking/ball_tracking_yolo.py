import torch
from ultralytics import YOLO
import cv2

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_frames(self,frames):
        ball_detections = []
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.track(frame,persist=True, tracker="bytetrack.yaml")[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict
    
    
    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            
            for track_id, (x1, y1, x2,y2) in ball_dict.items():
                cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 10, (0, 0, 255), 1)
                
                cv2.putText(frame, f'ID: {track_id}', (int((x1+x2)/2), int((y1+y2)/2) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
            output_video_frames.append(frame)
        
        return output_video_frames


    