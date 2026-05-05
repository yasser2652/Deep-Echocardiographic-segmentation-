from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F


def one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long().clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-5,
        include_background: bool = True,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights, persistent=False)

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
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            weights = weights[: self.num_classes]
            if not self.include_background:
                weights = weights[1:]
            dice = dice * weights / weights.clamp_min(1e-8).mean()
        return 1.0 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.dice = DiceLoss(num_classes=num_classes, class_weights=weights)
        self.ce = nn.CrossEntropyLoss(weight=weights)
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
    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = one_hot(target, self.num_classes).to(logits.device)
        dims = (0, 2, 3)
        tp = torch.sum(probs * target_oh, dims)
        fp = torch.sum(probs * (1 - target_oh), dims)
        fn = torch.sum((1 - probs) * target_oh, dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)[: self.num_classes]
            score = score * weights / weights.clamp_min(1e-8).mean()
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


class CombinedClinicalSegmentationLoss(nn.Module):
    """Weighted Dice + CE + Tversky with optional boundary regularization.

    The name is "clinical" because the defaults emphasize foreground cardiac
    structures, especially LV cavity and myocardium, while still optimizing a
    standard segmentation target. It does not directly optimize LV volumes.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        tversky_weight: float = 0.5,
        boundary_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ) -> None:
        super().__init__()
        if class_weights is None:
            class_weights = [0.25, 1.0, 1.0, 1.0][:num_classes]
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.numel() != num_classes:
            raise ValueError(f"class_weights must contain {num_classes} values, got {weights.numel()}.")
        self.dice = DiceLoss(num_classes=num_classes, class_weights=weights)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.tversky = TverskyLoss(num_classes=num_classes, alpha=tversky_alpha, beta=tversky_beta, class_weights=weights)
        self.boundary = BoundaryLoss(num_classes=num_classes) if boundary_weight > 0 else None
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.tversky_weight = float(tversky_weight)
        self.boundary_weight = float(boundary_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (
            self.dice_weight * self.dice(logits, target)
            + self.ce_weight * self.ce(logits, target.long())
            + self.tversky_weight * self.tversky(logits, target)
        )
        if self.boundary is not None:
            loss = loss + self.boundary_weight * self.boundary(logits, target)
        return loss


def parse_class_weights(value: Any, num_classes: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    weights = [float(v) for v in value]
    if len(weights) != num_classes:
        raise ValueError(f"Expected {num_classes} class weights, got {len(weights)}.")
    return weights


def build_loss(name: str, num_classes: int = 4, **kwargs: Any) -> nn.Module:
    key = name.lower()
    if "class_weights" in kwargs:
        kwargs["class_weights"] = parse_class_weights(kwargs["class_weights"], num_classes)
    if key in {"dice", "dice_loss"}:
        allowed = {k: kwargs[k] for k in ("smooth", "include_background", "class_weights") if k in kwargs}
        return DiceLoss(num_classes=num_classes, **allowed)
    if key in {"ce", "cross_entropy"}:
        weights = None if kwargs.get("class_weights") is None else torch.as_tensor(kwargs["class_weights"], dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)
    if key in {"dice_ce", "dice_cross_entropy"}:
        return DiceCrossEntropyLoss(
            num_classes=num_classes,
            dice_weight=float(kwargs.get("dice_weight", 1.0)),
            ce_weight=float(kwargs.get("ce_weight", 1.0)),
            class_weights=kwargs.get("class_weights"),
        )
    if key == "focal":
        return FocalLoss()
    if key == "tversky":
        return TverskyLoss(
            num_classes=num_classes,
            alpha=float(kwargs.get("alpha", kwargs.get("tversky_alpha", 0.3))),
            beta=float(kwargs.get("beta", kwargs.get("tversky_beta", 0.7))),
            class_weights=kwargs.get("class_weights"),
        )
    if key == "boundary":
        return BoundaryLoss(num_classes=num_classes)
    if key in {"combined_clinical", "clinical", "combined"}:
        return CombinedClinicalSegmentationLoss(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown loss '{name}'.")
