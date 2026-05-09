from ultralytics import YOLO
import cv2 as cv
import numpy as np  


model = YOLO('model_weights/yolo_ball_track_best.pt')
result = model.track(source='input/input_video_shortest.mp4', show=True, save=True, conf=0.3, iou=0.5)



print(result.boxes.xyxy)