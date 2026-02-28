import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MelanomaDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        csv_file: str,
        image_list: list,
        train: bool = True
    ):
        self.image_dir = image_dir

        df = pd.read_csv(csv_file)

        # Remove .jpg extension for filtering
        image_ids = [img.replace(".jpg", "") for img in image_list]
        self.df = df[df["image"].isin(image_ids)].reset_index(drop=True)

        if train:
            self.transform = A.Compose([
                A.Resize(300, 300),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=30,
                    p=0.5
                ),

                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),

                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),

                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(300, 300),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image"]
        label = float(row["MEL"])

        img_path = os.path.join(self.image_dir, image_id + ".jpg")

        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.float32)