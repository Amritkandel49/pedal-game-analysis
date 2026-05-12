import pandas as pd

def convert_to_df(detections):
    rows = []
    for frame_idx, frame_data in enumerate(detections):
        for obj_id, bbox in frame_data.items():
            rows.append({
                'frame': frame_idx,
                'id': obj_id,
                'x1': bbox[0],
                'y1': bbox[1],
                'x2': bbox[2],
                'y2': bbox[3]
            })
    df = pd.DataFrame(rows)
    return df


def convert_df_to_detections(df):
    if df.empty:
        return []
        
    max_frame = int(df['frame'].max())
    recovered_detections = [{} for _ in range(max_frame + 1)]
    
    for _, row in df.iterrows():
        frame_idx = int(row['frame'])
        obj_id = row['id']
        bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
        recovered_detections[frame_idx][obj_id] = bbox
    return recovered_detections