from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')

    model.train(data="C:/Users/adiad/OneDrive/Desktop/practica 2024/training/tennis-ball-detection-6/data.yaml", epochs=2, imgsz=640, device='cuda', batch = 10, workers = 16)

    #freeze_support()