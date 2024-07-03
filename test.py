from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train8/weights/best.pt')

    model.train(data="C:/Users/adiad/OneDrive/Desktop/practica 2024/ball-detection-fac/data.yaml",
                epochs=50,
                imgsz=640,
                device='0',
                batch = 1,
                workers = 1,
                amp = False,
                cache=True
                )

    #freeze_support()

