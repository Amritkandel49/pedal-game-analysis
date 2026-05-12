# from pyglm import row
from ultralytics import YOLO
import torch
import cv2 
import pickle
import pandas as pd
from utilities import convert_to_df, get_euclidean_dist, convert_df_to_detections


class PlayerTracker:
    def __init__(self, model_path, court_keypoints=None, expected_players=4):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.court_keypoints = court_keypoints
        self.expected_players = expected_players

        ## x is horizontal and y is vertical components
        if court_keypoints is not None:
            self.far_end_y_avg = (court_keypoints['p1'][1] + court_keypoints['p2'][1]) / 2   ## Upper limit where the player can maximum be located
            self.near_end_y_avg = (court_keypoints['p11'][1] + court_keypoints['p12'][1]) / 2 ## lower limit where the player can maximum be located
            
            self.p1 = court_keypoints['p1']
            self.p2 = court_keypoints['p2']
            self.p11 = court_keypoints['p11']
            self.p12 = court_keypoints['p12']
        
    def get_x_for_y(self, p_top, p_bottom, y):
        """Calculate the x coordinate on a line given y using linear interpolation."""
        x1, y1 = p_top
        x2, y2 = p_bottom
        if y2 == y1:
            return x1
        return x1 + (y - y1) * (x2 - x1) / (y2 - y1)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                detections = pickle.load(f)
            
            df_players = convert_to_df(detections)
            
            filtered_df = self.filter_player_tracks(df_players)
            # print(filtered_df.head(5), "\n\n")
            filtered_df.to_csv("miscellaneous/player_detections.csv", index=False)

            detections = convert_df_to_detections(filtered_df)
            
            return detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            detections.append(player_dict)
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(detections, f)
                
        df_players = convert_to_df(detections)
        
        filtered_df = self.filter_player_tracks(df_players)
        # print(filtered_df.head(5), "\n\n")
        filtered_df.to_csv("miscellaneous/player_detections.csv", index=False)
        
        
        
        detections = convert_df_to_detections(filtered_df)
        
        return detections   
    
    def detect_frame(self, frame):
        # results = self.model.track(frame, persist=True, tracker="custom_botsort.yaml")[0]
        results = self.model.track(frame, persist=True, classes=[0], tracker="custom_botsort.yaml")[0]
        player_dict = {}
        
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            # print("Track ID:", track_id)
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = results.names[object_cls_id]
            
            # print(f"Detected class name: '{object_cls_name}'")
            
            # if object_cls_name == "1" or object_cls_name == "player":
            #     player_dict[track_id] = result
            
            if result[3] > self.far_end_y_avg and result[3] < self.near_end_y_avg:
                if self.court_keypoints is not None:
                    y_bottom = result[3]
                    x_center = (result[0] + result[2]) / 2
                    
                    left_x_bound = self.get_x_for_y(self.p1, self.p11, y_bottom)
                    right_x_bound = self.get_x_for_y(self.p2, self.p12, y_bottom)
                    
                    buffer_x = 100
                    
                    if (left_x_bound - buffer_x) < x_center < (right_x_bound + buffer_x):
                        player_dict[track_id] = result
                else:
                    player_dict[track_id] = result
                # print(f"Player detected with ID: {track_id}")
        return player_dict
    
    
    def draw_bounding_boxes(self, frames, player_detections):
        output_frames = []
        
        for frame, players in zip(frames, player_detections):
            for track_id, (x1, y1, x2, y2) in players.items():
                if track_id == "player11" or track_id == "player12":
                    color = (0, 255, 0)  # green
                else:
                    color = (0, 0, 255)  # red
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'{track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            output_frames.append(frame)
        return output_frames 
            
            
    
    def filter_player_tracks(self, df):
        df = df.copy()
        top_four_ids = df['id'].value_counts().head(self.expected_players).index
        df.loc[~df['id'].isin(top_four_ids), 'id'] = pd.NA
        
        def bbox_center(row):
            return ((row["x1"] + row["x2"]) / 2, (row["y1"] + row["y2"]) / 2)

        nan_mask = df["id"].isna()

        for idx, row in df.loc[nan_mask].iterrows():
            prev_frame = row["frame"] - 1
            prev_rows = df[(df["frame"] == prev_frame) & df["id"].notna()]
            
            if prev_rows.empty:
                continue
            
            cx, cy = bbox_center(row)
            prev_centers = prev_rows.apply(bbox_center, axis=1)
            distances = ((prev_centers.apply(lambda p: p[0]) - cx) ** 2 + (prev_centers.apply(lambda p: p[1]) - cy) ** 2) ** 0.5
            
            closest_idx = distances.idxmin()
            df.at[idx, "id"] = df.at[closest_idx, "id"]
            
        avg_y_of_center_line_p6p7 = (self.court_keypoints['p6'][1] + self.court_keypoints['p7'][1]) / 2
            
        avg_y2_per_id = df.groupby('id')['y2'].mean()
        
        id_mapping = {}
        team1_count = 1
        team2_count = 1
        
        for unique_id, avg_y2 in avg_y2_per_id.items():
            if avg_y2 > avg_y_of_center_line_p6p7:
                id_mapping[unique_id] = f"player2{team2_count}"
                team2_count += 1
            else:
                id_mapping[unique_id] = f"player1{team1_count}"
                team1_count += 1
                
        # 4. Apply mapping to the dataframe
        df['id'] = df['id'].map(id_mapping)
            
        return df.sort_values(by=["frame", "id"]).reset_index(drop=True)
        