import os
import torch


# -------------------------------------------------
# Device Configuration
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Image / Data Configuration
# -------------------------------------------------
IMAGE_SIZE = 256
DATA_DIR = os.getenv("DATA_DIR", "data")


# -------------------------------------------------
# Training Hyperparameters
# -------------------------------------------------
BATCH_SIZE = 8        # Safe for RTX 3050 (4GB)
NUM_EPOCHS = 30
LR = 1e-3

NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()


# -------------------------------------------------
# Model Save Paths
# -------------------------------------------------
CLASSIFIER_SAVE_PATH = "checkpoints/classifier_best.pth"
SEGMENTATION_SAVE_PATH = "checkpoints/segmentation_best.pth"


# -------------------------------------------------
# Thresholds
# -------------------------------------------------
CLASSIFICATION_THRESHOLD = 0.2527
SEGMENTATION_THRESHOLD = 0.6