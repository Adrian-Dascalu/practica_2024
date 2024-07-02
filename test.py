from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train6/weights/best.pt')

    model.train(data="C:/Users/adiad/OneDrive/Desktop/practica 2024/training/tennis-ball-detection-6/data.yaml",
                epochs=100,
                imgsz=640,
                device='0',
                batch = 1,
                workers = 1,
                amp = False,
                cache=True
                )

    #freeze_support()