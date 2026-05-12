import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import cv2
import os
from ultralytics import YOLO

class ShotClassifier:
    def __init__(self, MODEL_PATH, input_video_path, ball_detections, player_detections):
        self.input_video_path = input_video_path
        self.ball_detections = ball_detections
        self.player_detections = player_detections
        self.MODEL_PATH = MODEL_PATH
        
    def get_hit_frame_sequences_for_player(self, hit_events, window_size=5, output_dir="output/crops"):
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print(f"Error opening video at {self.input_video_path}")
            return []

        v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        for event in hit_events:
            hit_frame = event['frame']
            player_id = event['player_id']
            # ball_pos = event['ball_pos']
            
            start_frame = max(0, hit_frame - window_size)
            end_frame = hit_frame + window_size
            
            frames_needed = window_size * 2 + 2
            
            sequence_crops_paths = []
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for f in range(start_frame, start_frame + frames_needed):
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame {f}")
                    break
                
                player_bbox = None # if not found in the frame 
                p_dict = self.player_detections[f] if f < len(self.player_detections) else None
                if p_dict is not None and player_id in p_dict:
                    player_bbox = p_dict[player_id]
                
                if player_bbox is not None:
                    (x1, y1, x2, y2) = player_bbox
                    
                    pad_w = int((x2 - x1) * 0.1)
                    pad_h = int((y2 - y1) * 0.1)
                    
                    crop_x1 = max(0, int(x1 - pad_w))
                    crop_y1 = max(0, int(y1 - pad_h))
                    crop_x2 = min(v_width, int(x2 + pad_w))
                    crop_y2 = min(v_height, int(y2 + pad_h))
                    
                    player_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if player_crop.size > 0:
                        # Append a copy of the crop to avoid holding the whole frame in memory
                        sequence_crops_paths.append(player_crop.copy())
                    else:
                        sequence_crops_paths.append(None)
                else:
                    # If player was completely missed by detector in this frame, append None or a blank frame
                    sequence_crops_paths.append(None)
                
            yield {
                'hit_frame': hit_frame,
                'player_id': player_id,
                'sequence_crops': sequence_crops_paths
            }
            
        cap.release()
        
    def detect_hits(self, player_vicinity_threshold=100):
        ball_data = []
        hit_events = []
        for frame_idx, b_dict in enumerate(self.ball_detections):
            if 1 in b_dict and pd.notna(b_dict[1][0]) and pd.notna(b_dict[1][1]):
                ball_data.append({'frame': frame_idx, 'x': b_dict[1][0], 'y': b_dict[1][1]})
            else:
                ball_data.append({'frame': frame_idx, 'x': np.nan, 'y': np.nan})
                
        df = pd.DataFrame(ball_data)
        
        # Using rolling windows to smooth out minor tracking jitters
        df['x_smooth'] = df['x'].rolling(window=8, min_periods=1, center=True).mean()
        df['y_smooth'] = df['y'].rolling(window=8, min_periods=1, center=True).mean()
        
        df['vx'] = df['x_smooth'].diff()
        df['vy'] = df['y_smooth'].diff()
        
        df['ax'] = df['vx'].diff()
        df['ay'] = df['vy'].diff()
        
        df['a_mag'] = np.sqrt(df['ax']**2 + df['ay']**2)
        df['a_mag'] = df['a_mag'].fillna(0)
        
        peaks, _ = find_peaks(df['a_mag'], distance=15, prominence=8)
        
        for peak_frame in peaks:
            ball_x = df.loc[peak_frame, 'x']
            ball_y = df.loc[peak_frame, 'y']
            
            if pd.isna(ball_x) or pd.isna(ball_y):
                continue
                
            closest_player = None
            min_dist = float('inf')
            
            p_dict = self.player_detections[peak_frame]
            for p_id, p_bbox in p_dict.items():
                px1, py1, px2, py2 = p_bbox
                
                box_width = px2 - px1
                box_height = py2 - py1
                expanded_dist = max(box_width, box_height) * 0.4
                
                if (px1 - expanded_dist <= ball_x <= px2 + expanded_dist) and \
                (py1 - expanded_dist <= ball_y <= py2 + expanded_dist):
                    closest_player = p_id
                    min_dist = 0 # It's inside the bbox
                    break 
                
                px_center = (px1 + px2) / 2
                py_center = (py1 + py2) / 2
                dist = np.hypot(ball_x - px_center, ball_y - py_center)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_player = p_id
                    
            if closest_player is not None and min_dist <= player_vicinity_threshold:
                hit_events.append({
                    'frame': int(peak_frame),
                    'player_id': closest_player,
                    'ball_pos': (int(ball_x), int(ball_y))
                })
                
        df_hit_events = pd.DataFrame(hit_events)
        print(df_hit_events.head(10))
        return hit_events
    
    
    def pose_extraction_for_all_hit_events(self, frame_sequences_for_all_shots):
        
        # Load model once to avoid reloading for every crop
        model = YOLO(self.MODEL_PATH)
        
        sequence_keypoints_all_shots = []
        for i, event in enumerate(frame_sequences_for_all_shots):
            
            hit_frame = event['hit_frame']
            player_id = event['player_id']
            sequence_crops = event['sequence_crops']
            
            valid_crops = []
            valid_indices = []
            
            for idx, crop in enumerate(sequence_crops):
                if crop is not None and crop.size > 0:
                    valid_crops.append(crop)
                    valid_indices.append(idx)
                    
            sequence_keypoints_each_shot = [np.zeros((17, 2))] * len(sequence_crops)
            
            if valid_crops:
                for res_idx, crop in enumerate(valid_crops):
                    orig_idx = valid_indices[res_idx]
                    
                    results = next(model.predict(crop, verbose=False, stream=True))
                    
                    if results.keypoints is not None and len(results.keypoints) > 0:
                        kpts = results.keypoints.xy[0].cpu().numpy() # Shape: (17, 2)
                        
                        h, w = crop.shape[:2]
                        if w > 0 and h > 0:
                           kpts[:, 0] = kpts[:, 0] / w
                           kpts[:, 1] = kpts[:, 1] / h
                        
                        sequence_keypoints_each_shot[orig_idx] = kpts
            
            del valid_crops
            del sequence_crops
            
            sequence_keypoints_all_shots.append({
                'hit_frame': hit_frame,
                'player_id': player_id,
                'sequence_keypoints': sequence_keypoints_each_shot
            })
            
        return sequence_keypoints_all_shots

    