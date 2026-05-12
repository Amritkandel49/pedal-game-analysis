import csv
import json


def export_frame_data(ball_detections, player_detections, predictions, output_csv="output/frame_data.csv", output_json="output/frame_data.json"):
    shot_lookup = {}
    for pred in predictions:
        frame = pred['hit_frame']
        player_id = pred['player_id']
        shot_type = pred['shot_type']
        if frame not in shot_lookup:
            shot_lookup[frame] = {}
        shot_lookup[frame][player_id] = shot_type

    total_frames = max(len(ball_detections), len(player_detections))

    all_rows = []

    for frame_idx in range(total_frames):

        ball_x = None
        ball_y = None
        if frame_idx < len(ball_detections):
            b_dict = ball_detections[frame_idx]
            if 1 in b_dict and b_dict[1][0] is not None and b_dict[1][1] is not None:
                ball_x = b_dict[1][0]
                ball_y = b_dict[1][1]

        players_in_frame = {}
        if frame_idx < len(player_detections):
            p_dict = player_detections[frame_idx]
            for player_id, bbox in p_dict.items():
                x1, y1, x2, y2 = bbox
                players_in_frame[str(player_id)] = {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1)
                }

        shot_made = frame_idx in shot_lookup
        shot_player = None
        shot_type = None
        if shot_made:
            for p_id, s_type in shot_lookup[frame_idx].items():
                shot_player = str(p_id)
                shot_type = s_type

        row = {
            "frame": frame_idx,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "players": players_in_frame,
            "shot_made": shot_made,
            "shot_player": shot_player,
            "shot_type": shot_type
        }
        all_rows.append(row)

    with open(output_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Saved JSON to {output_json}")

    all_player_ids = set()
    for row in all_rows:
        all_player_ids.update(row["players"].keys())
    all_player_ids = sorted(all_player_ids)

    csv_headers = ["frame", "ball_x", "ball_y"]
    for pid in all_player_ids:
        csv_headers += [f"{pid}_x1", f"{pid}_y1", f"{pid}_x2", f"{pid}_y2"]
    csv_headers += ["shot_made", "shot_player", "shot_type"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

        for row in all_rows:
            flat = {
                "frame": row["frame"],
                "ball_x": row["ball_x"],
                "ball_y": row["ball_y"],
                "shot_made": row["shot_made"],
                "shot_player": row["shot_player"],
                "shot_type": row["shot_type"]
            }
            for pid in all_player_ids:
                if pid in row["players"]:
                    flat[f"{pid}_x1"] = row["players"][pid]["x1"]
                    flat[f"{pid}_y1"] = row["players"][pid]["y1"]
                    flat[f"{pid}_x2"] = row["players"][pid]["x2"]
                    flat[f"{pid}_y2"] = row["players"][pid]["y2"]
                else:
                    flat[f"{pid}_x1"] = None
                    flat[f"{pid}_y1"] = None
                    flat[f"{pid}_x2"] = None
                    flat[f"{pid}_y2"] = None
            writer.writerow(flat)

    print(f"Saved CSV to {output_csv}")
