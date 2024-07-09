from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8l.pt')

    model.predict(source='input_videos/input_video_sh.mp4',
                conf = 0.20,
                #show = True,
                save = True,
                line_width = 1,
                device = '0'
                )