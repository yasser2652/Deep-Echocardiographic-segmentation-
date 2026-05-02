import numpy as np

from src.metrics import batch_segmentation_metrics, dice_per_class, iou_per_class
from src.postprocess import postprocess_mask


def test_dice_and_iou_perfect_match():
    target = np.array([[[0, 1], [2, 3]]])
    pred = target.copy()
    dice = dice_per_class(pred, target, num_classes=4)
    iou = iou_per_class(pred, target, num_classes=4)
    assert np.allclose(dice, 1.0)
    assert np.allclose(iou, 1.0)
    metrics = batch_segmentation_metrics(pred, target, num_classes=4)
    assert metrics["mean_dice"] == 1.0


def test_postprocessing_removes_tiny_islands():
    mask = np.zeros((32, 32), dtype=np.int64)
    mask[8:20, 8:20] = 1
    mask[0, 0] = 1
    processed = postprocess_mask(mask, num_classes=4, config={"min_region_size": 8, "keep_largest_component": True})
    assert processed[10, 10] == 1
    assert processed[0, 0] == 0

