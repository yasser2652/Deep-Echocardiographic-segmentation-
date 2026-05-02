from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F


def one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long().clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5, include_background: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = one_hot(target, self.num_classes).to(logits.device)
        if not self.include_background:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        denominator = torch.sum(probs + target_oh, dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int = 4, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target.long())


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long(), reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha * loss
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, num_classes: int = 4, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = one_hot(target, self.num_classes).to(logits.device)
        dims = (0, 2, 3)
        tp = torch.sum(probs * target_oh, dims)
        fp = torch.sum(probs * (1 - target_oh), dims)
        fn = torch.sum((1 - probs) * target_oh, dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - score.mean()


class BoundaryLoss(nn.Module):
    """Simple multiclass boundary loss using target signed distance maps."""

    def __init__(self, num_classes: int = 4, include_background: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background

    def _distance_maps(self, target: torch.Tensor) -> torch.Tensor:
        target_np = target.detach().cpu().numpy().astype(np.int64)
        maps = []
        start = 0 if self.include_background else 1
        for mask in target_np:
            per_class = []
            for cls in range(start, self.num_classes):
                fg = mask == cls
                if fg.any():
                    dist_out = ndimage.distance_transform_edt(~fg)
                    dist_in = ndimage.distance_transform_edt(fg)
                    dist = dist_out - dist_in
                else:
                    dist = np.zeros_like(mask, dtype=np.float32)
                per_class.append(dist.astype(np.float32))
            maps.append(np.stack(per_class, axis=0))
        return torch.from_numpy(np.stack(maps, axis=0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        if not self.include_background:
            probs = probs[:, 1:]
        dist_maps = self._distance_maps(target).to(logits.device)
        return torch.mean(probs * dist_maps)


class TemporalSmoothnessLoss(nn.Module):
    def forward(self, sequence_logits: torch.Tensor) -> torch.Tensor:
        if sequence_logits.ndim != 5 or sequence_logits.shape[1] < 2:
            return sequence_logits.new_tensor(0.0)
        probs = torch.softmax(sequence_logits, dim=2)
        return torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))


def build_loss(name: str, num_classes: int = 4, **kwargs: Any) -> nn.Module:
    key = name.lower()
    if key in {"dice", "dice_loss"}:
        return DiceLoss(num_classes=num_classes, **kwargs)
    if key in {"ce", "cross_entropy"}:
        return nn.CrossEntropyLoss()
    if key in {"dice_ce", "dice_cross_entropy"}:
        return DiceCrossEntropyLoss(num_classes=num_classes)
    if key == "focal":
        return FocalLoss()
    if key == "tversky":
        return TverskyLoss(num_classes=num_classes)
    if key == "boundary":
        return BoundaryLoss(num_classes=num_classes)
    raise ValueError(f"Unknown loss '{name}'.")

