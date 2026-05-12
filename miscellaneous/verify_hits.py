import cv2

VIDEO_PATH = "input/input_video_shortest.mp4"
hit_detected_frames = [85, 146, 171, 213, 233, 256]

cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 1000, 800)

window_size = 5  # frames before and after

hit_idx = 0
total_hits = len(hit_detected_frames)

while hit_idx >= 0 and hit_idx < total_hits:
    center_hit_frame = hit_detected_frames[hit_idx]
    
    start_frame = max(0, center_hit_frame - window_size)
    end_frame = center_hit_frame + window_size
    
    sequence_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for f in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
            
        color = (0, 0, 255) if f == center_hit_frame else (0, 255, 0)
        text = f"HIT FRAME: {f}" if f == center_hit_frame else f"Frame: {f}"
            
        cv2.putText(frame, f"Hit Event {hit_idx + 1}/{total_hits}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        sequence_frames.append(frame)

    if not sequence_frames:
        print("Couldn't read frames. Breaking.")
        break

    print(f"Showing hit event {hit_idx + 1} at frame {center_hit_frame}. Use A/D or Left/Right to switch. Q to quit.")
    
    seq_idx = 0
    while True:
        cv2.imshow("Frame", sequence_frames[seq_idx])
        key = cv2.waitKeyEx(100)
        
        if key != -1:
            break
            
        seq_idx = (seq_idx + 1) % len(sequence_frames)

    if key & 0xFF == ord('q'):
        break
    elif key in (83, 65363) or key & 0xFF == ord('d'):
        hit_idx += 1
        if hit_idx >= total_hits:
            hit_idx = total_hits - 1
    elif key in (81, 65361) or key & 0xFF == ord('a'):
        hit_idx -= 1
        if hit_idx < 0:
            hit_idx = 0

cap.release()
cv2.destroyAllWindows()
