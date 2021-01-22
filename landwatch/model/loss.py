import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth: float):
        super().__init__()

        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        numerator = (2.0 * intersection + self.smooth)
        denominator = (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)

        return (1 - (numerator/denominator)).mean()
