import torch
import torch.nn as nn


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    """
    Computes batch-averaged Dice score.
    """

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    """
    Computes batch-averaged IoU score.
    """

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Computes soft Dice loss (no thresholding).
        """

        pred = torch.sigmoid(pred)
        target = target.float()

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1.0 - dice

        return loss.mean()