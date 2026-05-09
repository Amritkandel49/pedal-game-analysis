import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TrackNet'))

import torch
from TrackNet.model import BallTrackerNet
from TrackNet.infer_on_video import infer_model, remove_outliers, split_track, interpolation, read_video


def track_ball(video_path: str, model_path: str, interpolate: bool = True) -> tuple:
    """
    Run ball tracking on a video using the pretrained TrackNet model.

    :param video_path:  path to input video file
    :param model_path:  path to pretrained TrackNet weights (.pt)
    :param interpolate: whether to fill missing detections (recommended for poor quality video)

    :return: (ball_track, fps)
        ball_track : list of dicts, one per frame:
                     {"frame": int, "timestamp": float, "x": float|None, "y": float|None}
        fps        : int, frames per second of the video
    """
    # 1. Read video using the existing repo function
    frames, fps = read_video(video_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BallTrackerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    ball_track, dists = infer_model(frames, model)

    ball_track = remove_outliers(ball_track, dists)

    if interpolate:
        subtracks = split_track(ball_track)
        for start, end in subtracks:
            ball_track[start:end] = interpolation(ball_track[start:end])

    result = []
    for frame_idx, (x, y) in enumerate(ball_track):
        result.append({
            "frame"    : frame_idx,
            "timestamp": round(frame_idx / fps, 4),
            "x"        : float(x) if x is not None else None,
            "y"        : float(y) if y is not None else None,
        })

    return result, fps



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    VIDEO_PATH = "input/input_video_shortest.mp4"
    TRACKNET_WEIGHTS = "model_weights/tracknet_model.pt"
    
    ball_track, fps = track_ball(
        video_path=VIDEO_PATH,
        model_path=TRACKNET_WEIGHTS,
        interpolate=True
    )
    
    print(f"Processed {len(ball_track)} frames at {fps} FPS")
    print("Sample output:", ball_track[10])
