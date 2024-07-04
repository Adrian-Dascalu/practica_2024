import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtKeypointsDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights = None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img_rgp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgp).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)

        keypoints = output.squeeze().numpy()

        original_height, original_width = img_rgp.shape[:2]

        keypoints[::2] *= original_width / 224
        keypoints[1::2] *= original_height / 224

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])

            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_frames = []

        for frame in video_frames: 
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)

        return output_frames