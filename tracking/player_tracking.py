from ultralytics import YOLO
import torch
import cv2 
class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def detect_frames(self, frames):
        detections = []
        for frame in frames:
            player_dict = self.detect_frame(frame)
            detections.append(player_dict)
            
        return detections   
    
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        player_dict = {}
        
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            # print("Track ID:", track_id)
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = results.names[object_cls_id]
            # print(f"Detected class name: '{object_cls_name}'")
            if object_cls_name == "1" or object_cls_name == "player":
                player_dict[track_id] = result
                # print(f"Player detected with ID: {track_id}")
        return player_dict
    
    
    def draw_bounding_boxes(self, frames, player_detections):
        output_frames = []
        
        for frame, players in zip(frames, player_detections):
            for track_id, (x1, y1, x2, y2) in players.items():
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            output_frames.append(frame)
        return output_frames 
            