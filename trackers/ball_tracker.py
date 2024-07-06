from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_detections):
        ball_detections = [x.get(1, []) for x in ball_detections]
        df_ball_positions = pd.DataFrame(ball_detections, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_detections = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
    
        return ball_detections
    
    def get_ball_shot_frames(self, ball_positions):
        minimum_change_frames = 25

        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window = 5, min_periods = 1, center = False).mean()
        df_ball_positions['dela_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames * 1.2)):
            negative_change = df_ball_positions['dela_y'].iloc[i] > 0 and df_ball_positions['dela_y'].iloc[i + 1] < 0
            positive_change = df_ball_positions['dela_y'].iloc[i] < 0 and df_ball_positions['dela_y'].iloc[i + 1] > 0
        
            if negative_change or positive_change:
                change = 0

                for change in range(i + 1, i + int(minimum_change_frames * 1.2) + 1):
                    negative_change_next_frame = df_ball_positions['dela_y'].iloc[i] > 0 and df_ball_positions['dela_y'].iloc[change] < 0
                    positive_change_next_frame = df_ball_positions['dela_y'].iloc[i] < 0 and df_ball_positions['dela_y'].iloc[change] > 0
            
                    if negative_change and negative_change_next_frame:
                        change += 1
                    elif positive_change and positive_change_next_frame:
                        change += 1

                if change > minimum_change_frames - 1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.to_list()

        return frame_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detection = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detection = pickle.load(f)
            return ball_detection

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detection.append(ball_dict)

        if not read_from_stub:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detection, f)

        return ball_detection

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.15)[0]

        ball_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]

            ball_dict[1] = result

        return ball_dict
    
    def draw_bbox(self, video_frames, ball_detection):
        output_frames = []

        for frame, ball_dict in zip(video_frames, ball_detection):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox

                cv2.putText(frame, f"Ball", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #   cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2

                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (50, 0, 255), 2)
            output_frames.append(frame)
        
        return output_frames