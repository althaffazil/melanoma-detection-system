import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import IMAGE_SIZE


class ISICSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_list: list
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = image_list

        self.transform = A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):

        img_name = self.images[idx]
        mask_name = img_name.replace(".jpg", "_segmentation.png")

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {mask_path}")

        mask = np.expand_dims(mask, axis=-1)

        augmented = self.transform(image=image, mask=mask)

        image = augmented["image"]
        mask = augmented["mask"].float() / 255.0

        return image, mask