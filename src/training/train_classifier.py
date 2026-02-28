import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.models.classifier import MelanomaClassifier
from src.datasets.classification_dataset import MelanomaDataset
from src.config import (
    DEVICE,
    DATA_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    NUM_WORKERS,
    PIN_MEMORY,
    CLASSIFIER_SAVE_PATH
)


def train():

    image_dir = os.path.join(DATA_DIR, "images")
    csv_file = os.path.join(DATA_DIR, "GroundTruth.csv")

    df = pd.read_csv(csv_file)
    image_names = [img + ".jpg" for img in df["image"]]

    train_images, val_images = train_test_split(
        image_names,
        test_size=0.2,
        stratify=df["MEL"],
        random_state=42
    )

    train_dataset = MelanomaDataset(image_dir, csv_file, train_images, train=True)
    val_dataset = MelanomaDataset(image_dir, csv_file, val_images, train=False)

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

    model = MelanomaClassifier().to(DEVICE)

    # Compute class imbalance dynamically
    train_labels = df[df["image"].isin([img.replace(".jpg", "") for img in train_images])]["MEL"]
    pos_count = train_labels.sum()
    neg_count = len(train_labels) - pos_count

    pos_weight = torch.tensor([neg_count / pos_count]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    use_amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader):

            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()

        # ---------------- VALIDATION ----------------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.unsqueeze(1).to(DEVICE)

                outputs = model(images)
                probs = torch.sigmoid(outputs)

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        auc = roc_auc_score(all_labels, all_preds)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(CLASSIFIER_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), CLASSIFIER_SAVE_PATH)
            print("Best model saved.")

    print("\nTraining complete.")
    print("Best AUC:", best_auc)


if __name__ == "__main__":
    train()