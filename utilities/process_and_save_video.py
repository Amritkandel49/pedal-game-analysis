import cv2
import pandas as pd
import numpy as np


def _build_shot_lookup(predictions):
    lookup = {}
    for pred in sorted(predictions, key=lambda p: p['hit_frame']):
        lookup[pred['hit_frame']] = {
            'player_id': str(pred['player_id']),
            'shot_type': pred['shot_type']
        }
    return lookup


def _draw_panel(h, panel_width, sorted_preds, frame_idx, fps):
    """Build the dark side panel with the shot log table."""
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    cv2.putText(panel, "SHOT LOG", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(panel, (10, 42), (panel_width - 10, 42), (100, 100, 100), 1)

    col_time   = 10
    col_player = 80
    col_shot   = 185

    cv2.putText(panel, "Time",   (col_time,   62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(panel, "Player", (col_player, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(panel, "Shot",   (col_shot,   62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.line(panel, (10, 68), (panel_width - 10, 68), (70, 70, 70), 1)

    # Most recent hit up to current frame
    recent_hit_frame = None
    for pred in sorted_preds:
        if pred['hit_frame'] <= frame_idx:
            recent_hit_frame = pred['hit_frame']

    visible_preds = [p for p in sorted_preds if p['hit_frame'] <= frame_idx]
    visible_preds = visible_preds[-12:]

    row_y = 85
    row_h = 24

    for pred in visible_preds:
        if row_y > h - 20:
            break

        hit_frame  = pred['hit_frame']
        player_id  = str(pred['player_id'])
        shot_type  = pred['shot_type'].upper()

        total_sec = hit_frame / fps
        mins = int(total_sec // 60)
        secs = int(total_sec % 60)
        timestamp = f"{mins:02d}:{secs:02d}"

        is_current = (hit_frame == recent_hit_frame) and (frame_idx < hit_frame + int(fps * 2))

        if is_current:
            cv2.rectangle(panel,
                          (5, row_y - 14),
                          (panel_width - 5, row_y + 8),
                          (0, 80, 0), -1)
            text_color = (0, 255, 100)
        else:
            text_color = (210, 210, 210)

        cv2.putText(panel, timestamp, (col_time,   row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)
        cv2.putText(panel, player_id, (col_player, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)
        cv2.putText(panel, shot_type, (col_shot,   row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)

        row_y += row_h

    return panel, recent_hit_frame


def _draw_on_frame(frame, frame_idx, player_detections, ball_detections, sorted_preds, shot_lookup, fps, panel_width=340):

    if frame_idx < len(player_detections):
        for track_id, bbox in player_detections[frame_idx].items():
            x1, y1, x2, y2 = bbox
            if str(track_id).startswith("player1"):
                color = (0, 255, 0)   # green: team 1
            else:
                color = (0, 0, 255)   # red: team 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{track_id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if frame_idx < len(ball_detections):
        for track_id, (x_center, y_center) in ball_detections[frame_idx].items():
            if pd.notna(x_center) and pd.notna(y_center):
                cv2.circle(frame, (int(x_center), int(y_center)), 10, (0, 255, 255), 2)

    recent_hit_frame = None
    for pred in sorted_preds:
        if pred['hit_frame'] <= frame_idx:
            recent_hit_frame = pred['hit_frame']

    if recent_hit_frame is not None:
        frames_since = frame_idx - recent_hit_frame
        if 0 <= frames_since < int(fps * 2):
            info = shot_lookup.get(recent_hit_frame)
            if info:
                label = f"{info['player_id'].upper()} : {info['shot_type'].upper()}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                bx1, by1 = 20, 20
                bx2 = bx1 + tw + 20
                by2 = by1 + th + 16
                overlay = frame.copy()
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                cv2.putText(frame, label, (bx1 + 10, by2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)

    h = frame.shape[0]
    panel, _ = _draw_panel(h, panel_width, sorted_preds, frame_idx, fps)

    return np.hstack([frame, panel])


def process_and_save_video(video_path, output_path, player_detections, ball_detections, predictions, panel_width=340):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = w + panel_width
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, h))

    sorted_preds = sorted(predictions, key=lambda p: p['hit_frame'])
    shot_lookup  = _build_shot_lookup(predictions)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        combined = _draw_on_frame(frame, frame_idx, player_detections, ball_detections,
                                  sorted_preds, shot_lookup, fps, panel_width)
        out.write(combined)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved output video to {output_path}  ({frame_idx} frames)")
