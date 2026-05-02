import torch

from src.losses import BoundaryLoss, DiceCrossEntropyLoss, DiceLoss, FocalLoss, TverskyLoss


def test_losses_forward_pass():
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randint(0, 4, (2, 32, 32))
    for loss_fn in [
        DiceLoss(num_classes=4),
        DiceCrossEntropyLoss(num_classes=4),
        FocalLoss(),
        TverskyLoss(num_classes=4),
        BoundaryLoss(num_classes=4),
    ]:
        loss = loss_fn(logits, target)
        assert torch.isfinite(loss)

