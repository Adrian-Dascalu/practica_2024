from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8l.pt')

    model.track(source='input_videos/input_video.mp4', save = True, device = '0', conf = 0.4, show = True, line_width = 2)