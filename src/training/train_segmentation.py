import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.datasets.segmentation_dataset import ISICSegmentationDataset
from src.models.unet import UNet
from src.utils.utils import dice_score, iou_score, DiceLoss
from src.config import (
    DEVICE,
    DATA_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    NUM_WORKERS,
    PIN_MEMORY,
    SEGMENTATION_SAVE_PATH
)


def train():

    image_dir = os.path.join(DATA_DIR, "images")
    mask_dir = os.path.join(DATA_DIR, "masks")

    all_images = os.listdir(image_dir)

    train_images, val_images = train_test_split(
        all_images,
        test_size=0.2,
        random_state=42
    )

    train_dataset = ISICSegmentationDataset(image_dir, mask_dir, train_images)
    val_dataset = ISICSegmentationDataset(image_dir, mask_dir, val_images)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = UNet().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5
    )

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    def combined_loss(pred, target):
        return 0.5 * bce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    use_amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = np.inf
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 7

    for epoch in range(NUM_EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader):

            images = images.to(DEVICE)
            masks = masks.permute(0, 3, 1, 2).to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = combined_loss(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        dice_total = 0.0
        iou_total = 0.0

        with torch.no_grad():
            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.permute(0, 3, 1, 2).to(DEVICE)

                outputs = model(images)
                loss = combined_loss(outputs, masks)

                val_loss += loss.item()
                dice_total += dice_score(outputs, masks).item()
                iou_total += iou_score(outputs, masks).item()

        val_loss /= len(val_loader)
        dice_avg = dice_total / len(val_loader)
        iou_avg = iou_total / len(val_loader)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Dice: {dice_avg:.4f}")
        print(f"IoU: {iou_avg:.4f}")

        # ---------------- CHECKPOINT ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(SEGMENTATION_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SEGMENTATION_SAVE_PATH)
            patience_counter = 0
            print("Saved Best Model")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    train()