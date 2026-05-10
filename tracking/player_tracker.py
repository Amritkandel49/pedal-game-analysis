from ultralytics import YOLO
import torch
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
        results = self.model.track(frame, persist=True)[0]
        player_dict = {}
        
        
        
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = results.names[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
                
        return player_dict