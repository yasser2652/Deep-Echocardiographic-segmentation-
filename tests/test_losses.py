import torch

from src.losses import BoundaryLoss, CombinedClinicalSegmentationLoss, DiceCrossEntropyLoss, DiceLoss, FocalLoss, TverskyLoss, build_loss


def test_losses_forward_pass():
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randint(0, 4, (2, 32, 32))
    for loss_fn in [
        DiceLoss(num_classes=4),
        DiceCrossEntropyLoss(num_classes=4),
        FocalLoss(),
        TverskyLoss(num_classes=4),
        BoundaryLoss(num_classes=4),
        CombinedClinicalSegmentationLoss(num_classes=4, class_weights=[0.2, 1.0, 1.2, 1.0]),
        build_loss("combined_clinical", num_classes=4, class_weights="0.2,1.0,1.2,1.0"),
    ]:
        loss = loss_fn(logits, target)
        assert torch.isfinite(loss)
