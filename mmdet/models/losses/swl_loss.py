import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def smooth_wasserstein_loss(pred, target, thres=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff > thres,
        diff - 0.5 * thres,
        0.5 * (diff ** 2) / thres
    )
    return loss

@MODELS.register_module()
class SmoothWassersteinLoss(nn.Module):
    def __init__(self,
                 thres=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.thres = thres
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction = (
            reduction_override if reduction_override else self.reduction
        )
        loss = self.loss_weight * smooth_wasserstein_loss(
            pred,
            target,
            thres=self.thres,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss
