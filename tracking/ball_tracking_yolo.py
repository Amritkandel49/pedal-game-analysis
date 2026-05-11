import torch
from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def _extrapolate_continuous_positions(self, df_ball_pos_center, max_extrapolate=4, min_continuous=2):
        """
        Extrapolates missing ball positions if there is continuous movement in 
        preceding or succeeding frames, rather than interpolating between gaps.
        """
        df_ext = df_ball_pos_center.copy()
        n = len(df_ext)
        
        # Forward extrapolation
        continuous_count = 0
        for i in range(n):
            if pd.notna(df_ext.loc[i, 'x_center']):
                continuous_count += 1
            else:
                if continuous_count >= min_continuous:
                    idx1 = i - min_continuous
                    idx2 = i - 1
                    dx = (df_ext.loc[idx2, 'x_center'] - df_ext.loc[idx1, 'x_center']) / (idx2 - idx1)
                    dy = (df_ext.loc[idx2, 'y_center'] - df_ext.loc[idx1, 'y_center']) / (idx2 - idx1)
                    
                    frames_to_fill = 0
                    while (i + frames_to_fill < n) and pd.isna(df_ext.loc[i + frames_to_fill, 'x_center']) and (frames_to_fill < max_extrapolate):
                        frames_to_fill += 1
                        
                    for j in range(frames_to_fill):
                        df_ext.loc[i+j, 'x_center'] = df_ext.loc[idx2, 'x_center'] + dx * (j + 1)
                        df_ext.loc[i+j, 'y_center'] = df_ext.loc[idx2, 'y_center'] + dy * (j + 1)
                continuous_count = 0
                
        # Backward extrapolation
        continuous_count = 0
        for i in range(n-1, -1, -1):
            if pd.notna(df_ext.loc[i, 'x_center']):
                continuous_count += 1
            else:
                if continuous_count >= min_continuous:
                    idx1 = i + min_continuous
                    idx2 = i + 1
                    dx = (df_ext.loc[idx1, 'x_center'] - df_ext.loc[idx2, 'x_center']) / (idx1 - idx2)
                    dy = (df_ext.loc[idx1, 'y_center'] - df_ext.loc[idx2, 'y_center']) / (idx1 - idx2)
                    
                    frames_to_fill = 0
                    while (i - frames_to_fill >= 0) and pd.isna(df_ext.loc[i - frames_to_fill, 'x_center']) and (frames_to_fill < max_extrapolate):
                        frames_to_fill += 1
                        
                    for j in range(frames_to_fill):
                        df_ext.loc[i-j, 'x_center'] = df_ext.loc[idx2, 'x_center'] + dx * (-(j + 1))
                        df_ext.loc[i-j, 'y_center'] = df_ext.loc[idx2, 'y_center'] + dy * (-(j + 1))
                continuous_count = 0
                
        return df_ext

    def _interpolate_ball_positions(self, df_ball_pos_center, max_distance=150):
        # interpolate the missing values only if the gap is between two near points
        df_interpolated = df_ball_pos_center.copy()
        valid_indices = df_interpolated.dropna().index.tolist()
        
        for i in range(len(valid_indices) - 1):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i+1]
            
            if idx2 - idx1 > 1: # Gap exists
                p1 = df_interpolated.loc[idx1, ['x_center', 'y_center']].values
                p2 = df_interpolated.loc[idx2, ['x_center', 'y_center']].values
                dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                
                if dist <= max_distance:
                    df_interpolated.loc[idx1:idx2] = df_interpolated.loc[idx1:idx2].interpolate()
        
        ball_positions = [{1:x} for x in df_interpolated.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        
        def convert_to_center(ball_detections):
            ball_pos = [x.get(1,[]) for x in ball_detections]
            
            p_center = []
            for bp in ball_pos:
                if len(bp) == 4:
                    x1, y1, x2, y2 = bp
                    p_center.append((int((x1+x2)/2.0), int((y1+y2)/2.0)))
                else:
                    p_center.append((None, None))
                    
            df_ball_pos_center = pd.DataFrame(p_center, columns=['x_center','y_center'])   
            
            df_ball_pos_center.to_csv('ball_positions.csv', index=True) 
            
            # Standard interpolation for small gaps if needed
            ball_detections = self._interpolate_ball_positions(df_ball_pos_center, max_distance=350)
            
            ball_detections = self._interpolate_ball_positions(df_ball_pos_center, max_distance=200)
                
            # Extrapolate forward/backward based on continuous motion
            df_ball_pos_center = self._extrapolate_continuous_positions(df_ball_pos_center, max_extrapolate=3, min_continuous=2)
            
            
            
            return ball_detections
        
        
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
                
                ball_detections = convert_to_center(ball_detections)
            
            return ball_detections
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
                
        ball_detections =convert_to_center(ball_detections)
        
        
        return ball_detections

    def detect_frame(self,frame):
        # Using botsort.yaml (BoT-SORT), which is Ultralytics' built-in advanced tracking 
        # alternative to DeepSORT (as DeepSORT is not natively built-in by default)
        results = self.model.track(frame, persist=True,tracker="botsort.yaml")[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict
    
    
    def draw_bboxes(self,video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            
            for track_id, (x_center, y_center) in ball_dict.items():
                if pd.notna(x_center) and pd.notna(y_center):
                    cv2.circle(frame, (int(x_center), int(y_center)), 10, (0, 0, 255), 1)
                    
                    cv2.putText(frame, f'ID: {track_id}', (int(x_center), int(y_center) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
            output_video_frames.append(frame)
        
        return output_video_frames


    