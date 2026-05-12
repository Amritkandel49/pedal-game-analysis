import cv2
import numpy as np


def draw_shot_overlay(frames, predictions, fps=24, panel_width=340):

    output_frames = []

    sorted_preds = sorted(predictions, key=lambda p: p['hit_frame'])

    for frame_idx, frame in enumerate(frames):
        h, w = frame.shape[:2]

        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # dark grey background

        cv2.putText(panel, "SHOT LOG", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.line(panel, (10, 42), (panel_width - 10, 42), (100, 100, 100), 1)

        col_time  = 10
        col_player = 80
        col_shot  = 185

        cv2.putText(panel, "Time", (col_time, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.putText(panel, "Player", (col_player, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.putText(panel, "Shot", (col_shot, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        cv2.line(panel, (10, 68), (panel_width - 10, 68), (70, 70, 70), 1)

        recent_hit_frame = None
        for pred in sorted_preds:
            if pred['hit_frame'] <= frame_idx:
                recent_hit_frame = pred['hit_frame']

        row_y_start = 85
        row_height  = 24

        visible_preds = [p for p in sorted_preds if p['hit_frame'] <= frame_idx]

        visible_preds = visible_preds[-12:]

        for pred in visible_preds:
            if row_y_start > h - 20:
                break

            hit_frame   = pred['hit_frame']
            player_id   = str(pred['player_id'])
            shot_type   = pred['shot_type'].upper()

            # Convert frame number to timestamp mm:ss
            total_seconds = hit_frame / fps
            mins  = int(total_seconds // 60)
            secs  = int(total_seconds % 60)
            timestamp = f"{mins:02d}:{secs:02d}"

            is_current = (hit_frame == recent_hit_frame) and (hit_frame == frame_idx or
                          (frame_idx >= hit_frame and frame_idx < hit_frame + int(fps * 2)))

            if is_current:
                cv2.rectangle(panel,
                              (5, row_y_start - 14),
                              (panel_width - 5, row_y_start + 8),
                              (0, 80, 0), -1)
                text_color = (0, 255, 100)
            else:
                text_color = (210, 210, 210)

            cv2.putText(panel, timestamp,  (col_time,   row_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)
            cv2.putText(panel, player_id,  (col_player, row_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)
            cv2.putText(panel, shot_type,  (col_shot,   row_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)

            row_y_start += row_height

        if recent_hit_frame is not None:
            frames_since_hit = frame_idx - recent_hit_frame
            if 0 <= frames_since_hit < int(fps * 2):
                current_pred = next((p for p in sorted_preds if p['hit_frame'] == recent_hit_frame), None)
                if current_pred:
                    label = f"{current_pred['player_id'].upper()} : {current_pred['shot_type'].upper()}"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    box_x1 = 20
                    box_y1 = 20
                    box_x2 = box_x1 + tw + 20
                    box_y2 = box_y1 + th + 16

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

                    cv2.putText(frame, label,
                                (box_x1 + 10, box_y2 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)

        combined = np.hstack([frame, panel])
        output_frames.append(combined)

    return output_frames
