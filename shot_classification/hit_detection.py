import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def detect_hits(ball_detections, player_detections, player_vicinity_threshold=100):
    """
    Detects frames where a player hits the ball.
    
    :param ball_detections: List of dicts, e.g. [{1: (x, y)}, {1: (x, y)}, ...]
    :param player_detections: List of dicts, e.g. [{'player11': [x1,y1,x2,y2]}, ...]
    :param player_vicinity_threshold: Max distance in pixels between ball and player center to be considered a hit
    :return: List of events [{'frame': int, 'player_id': str, 'ball_pos': tuple}]
    """
    
    # 1. Extract ball positions into a DataFrame
    ball_data = []
    for frame_idx, b_dict in enumerate(ball_detections):
        if 1 in b_dict and pd.notna(b_dict[1][0]) and pd.notna(b_dict[1][1]):
            ball_data.append({'frame': frame_idx, 'x': b_dict[1][0], 'y': b_dict[1][1]})
        else:
            ball_data.append({'frame': frame_idx, 'x': np.nan, 'y': np.nan})
            
    df = pd.DataFrame(ball_data)
    
    # 2. Calculate Velocity (dx, dy) and Acceleration (ax, ay)
    # Using rolling windows to smooth out minor tracking jitters
    df['x_smooth'] = df['x'].rolling(window=8, min_periods=1, center=True).mean()
    df['y_smooth'] = df['y'].rolling(window=8, min_periods=1, center=True).mean()
    
    df['vx'] = df['x_smooth'].diff()
    df['vy'] = df['y_smooth'].diff()
    
    df['ax'] = df['vx'].diff()
    df['ay'] = df['vy'].diff()
    
    # Magnitude of acceleration (A sharp spike here represents a sudden change in direction/speed)
    df['a_mag'] = np.sqrt(df['ax']**2 + df['ay']**2)
    df['a_mag'] = df['a_mag'].fillna(0)
    
    # 3. Find peaks in the acceleration (these are candidate events: hits, wall bounces, floor bounces)
    # distance=15 means we expect at least 15 frames between hits
    # prominence/threshold might need tuning based on video resolution
    peaks, _ = find_peaks(df['a_mag'], distance=15, prominence=8)
    
    hit_events = []
    
    # 4. Filter candidate peaks: Is a player nearby?
    for peak_frame in peaks:
        ball_x = df.loc[peak_frame, 'x']
        ball_y = df.loc[peak_frame, 'y']
        
        # If tracking lost the ball right at the hit, skip
        if pd.isna(ball_x) or pd.isna(ball_y):
            continue
            
        closest_player = None
        min_dist = float('inf')
        
        # Check all players generated in this frame
        p_dict = player_detections[peak_frame]
        for p_id, p_bbox in p_dict.items():
            px1, py1, px2, py2 = p_bbox
            
            # Expand the bounding box to account for racket reach (approx 40% of the player's max dimension)
            box_width = px2 - px1
            box_height = py2 - py1
            expanded_dist = max(box_width, box_height) * 0.4
            
            # Simple check: Is the ball inside the player's extended bounding box?
            if (px1 - expanded_dist <= ball_x <= px2 + expanded_dist) and \
               (py1 - expanded_dist <= ball_y <= py2 + expanded_dist):
                closest_player = p_id
                min_dist = 0 # It's inside the bbox
                break 
            
            # Otherwise check distance to center of the player
            px_center = (px1 + px2) / 2
            py_center = (py1 + py2) / 2
            dist = np.hypot(ball_x - px_center, ball_y - py_center)
            
            if dist < min_dist:
                min_dist = dist
                closest_player = p_id
                
        # If the closest player is within our proximity threshold, we register it as a HIT
        if closest_player is not None and min_dist <= player_vicinity_threshold:
            hit_events.append({
                'frame': int(peak_frame),
                'player_id': closest_player,
                'ball_pos': (int(ball_x), int(ball_y))
            })
            
    df_hit_events = pd.DataFrame(hit_events)
    print(df_hit_events.head(10))
    return hit_events