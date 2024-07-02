from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train7/weights/best.pt')

    model.predict(source='input_videos/input_video.mp4',
                conf = 0.25,
                show = True,
                save = True,
                line_width = 1,
                device = '0'
                )