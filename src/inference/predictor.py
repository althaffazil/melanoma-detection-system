import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.unet import UNet
from src.models.classifier import MelanomaClassifier
from src.config import DEVICE, IMAGE_SIZE, SEGMENTATION_THRESHOLD


class SkinLesionPredictor:

    def __init__(
        self,
        classifier_path: str,
        segmentation_path: str
    ):
        self.device = DEVICE

        # Load segmentation model
        self.seg_model = UNet().to(self.device)
        self.seg_model.load_state_dict(
            torch.load(segmentation_path, map_location=self.device)
        )
        self.seg_model.eval()

        # Load classification model
        self.clf_model = MelanomaClassifier().to(self.device)
        self.clf_model.load_state_dict(
            torch.load(classifier_path, map_location=self.device)
        )
        self.clf_model.eval()

        # Classification transform
        self.clf_transform = A.Compose([
            A.Resize(300, 300),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

    def predict_classification(self, image: np.ndarray) -> float:
        augmented = self.clf_transform(image=image)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.clf_model(tensor)
            prob = torch.sigmoid(output).item()

        return prob

    def predict_segmentation(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        tensor = torch.tensor(
            img,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.seg_model(tensor)
            pred = torch.sigmoid(output)
            pred = (pred > SEGMENTATION_THRESHOLD).float()

        return pred.squeeze().cpu().numpy()