from __future__ import annotations

import numpy as np
import torch


def to_label_map(prediction: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu()
        if prediction.ndim == 4:
            prediction = prediction.argmax(dim=1)
        return prediction.numpy().astype(np.int64)
    prediction = np.asarray(prediction)
    if prediction.ndim == 4:
        prediction = prediction.argmax(axis=1)
    return prediction.astype(np.int64)


def dice_per_class(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    pred = to_label_map(prediction)
    tgt = to_label_map(target)
    scores = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        p = pred == cls
        t = tgt == cls
        denom = p.sum() + t.sum()
        if denom == 0:
            scores[cls] = 1.0
        else:
            scores[cls] = 2.0 * np.logical_and(p, t).sum() / denom
    return scores


def iou_per_class(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    pred = to_label_map(prediction)
    tgt = to_label_map(target)
    scores = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        p = pred == cls
        t = tgt == cls
        union = np.logical_or(p, t).sum()
        if union == 0:
            scores[cls] = 1.0
        else:
            scores[cls] = np.logical_and(p, t).sum() / union
    return scores


def mean_foreground(score: np.ndarray) -> float:
    if len(score) <= 1:
        return float(np.mean(score))
    return float(np.mean(score[1:]))


def batch_segmentation_metrics(
    logits_or_pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    num_classes: int = 4,
) -> dict[str, float]:
    dice = dice_per_class(logits_or_pred, target, num_classes=num_classes)
    iou = iou_per_class(logits_or_pred, target, num_classes=num_classes)
    metrics = {f"dice_class_{i}": float(v) for i, v in enumerate(dice)}
    metrics.update({f"iou_class_{i}": float(v) for i, v in enumerate(iou)})
    metrics["mean_dice"] = mean_foreground(dice)
    metrics["mean_iou"] = mean_foreground(iou)
    return metrics


class MetricAccumulator:
    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.dice: list[np.ndarray] = []
        self.iou: list[np.ndarray] = []

    def update(self, logits_or_pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        self.dice.append(dice_per_class(logits_or_pred, target, self.num_classes))
        self.iou.append(iou_per_class(logits_or_pred, target, self.num_classes))

    def compute(self) -> dict[str, float]:
        dice = np.mean(np.stack(self.dice, axis=0), axis=0) if self.dice else np.ones(self.num_classes)
        iou = np.mean(np.stack(self.iou, axis=0), axis=0) if self.iou else np.ones(self.num_classes)
        out = {f"dice_class_{i}": float(v) for i, v in enumerate(dice)}
        out.update({f"iou_class_{i}": float(v) for i, v in enumerate(iou)})
        out["mean_dice"] = mean_foreground(dice)
        out["mean_iou"] = mean_foreground(iou)
        return out

