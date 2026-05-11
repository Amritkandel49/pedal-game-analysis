import cv2
import os

COURT_KEYPOINTS_EXAMPLE_PATH = 'examples/court_keypoint_example.png'

class KeypointCollector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.keypoints = {}
        self.points_list = []
        self.max_points = 12
        self.example_image_path = COURT_KEYPOINTS_EXAMPLE_PATH
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points_list) < self.max_points:
            self.points_list.append((x, y))
            cv2.circle(self.display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(self.display_frame, f"p{len(self.points_list)}", (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Keypoint Collector", self.display_frame)
            
    def collect_keypoints(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read video frame.")
            return {}
            
        self.display_frame = frame.copy()
        
        # Load and overlay example image if it exists
        if os.path.exists(self.example_image_path):
            example_img = cv2.imread(self.example_image_path)
            if example_img is not None:
                # Resize example image to 25% of the frame width
                ex_h, ex_w = example_img.shape[:2]
                target_w = int(self.display_frame.shape[1] * 0.25)
                target_h = int(ex_h * (target_w / ex_w))
                example_img_resized = cv2.resize(example_img, (target_w, target_h))
                
                # Overlay on top right
                self.display_frame[0:target_h, -target_w:] = example_img_resized
                
        cv2.putText(self.display_frame, f"Click {self.max_points} points in order", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
        cv2.namedWindow("Keypoint Collector")
        cv2.setMouseCallback("Keypoint Collector", self._mouse_callback)
        
        print(f"Please click the {self.max_points} keypoints in order. Press 'q' or 'ESC' to exit early.")
        cv2.imshow("Keypoint Collector", self.display_frame)
        
        while len(self.points_list) < self.max_points:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
                
        cv2.destroyAllWindows()
        
        # Format mapping to p1, p2... p12
        for i, point in enumerate(self.points_list):
            self.keypoints[f'p{i+1}'] = point
            
        return self.keypoints
    
    def get_keypoints(self):
        return self.keypoints
    
    
if __name__ == "__main__":
    video_path = "input/input_video_shortest.mp4"
    collector = KeypointCollector(video_path)
    keypoints = collector.collect_keypoints()
    print("Collected Keypoints:", keypoints)