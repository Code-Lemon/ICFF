import math
import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class RNWDLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, anchor_area_thresh=1024):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.anchor_area_thresh = anchor_area_thresh
        self.constant = 12.8

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        # pred, target: (N, 4)
        pred_cx = (pred[..., 0] + pred[..., 2]) / 2
        pred_cy = (pred[..., 1] + pred[..., 3]) / 2
        pred_w = (pred[..., 2] - pred[..., 0]).clamp(min=1e-6)
        pred_h = (pred[..., 3] - pred[..., 1]).clamp(min=1e-6)

        target_cx = (target[..., 0] + target[..., 2]) / 2
        target_cy = (target[..., 1] + target[..., 3]) / 2
        target_w = (target[..., 2] - target[..., 0]).clamp(min=1e-6)
        target_h = (target[..., 3] - target[..., 1]).clamp(min=1e-6)

        dist = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2
        size = (pred_w/2 - target_w/2)**2 + (pred_h/2 - target_h/2)**2
        wasserstein_distance = dist + size

        reg_term = 0.1 * (pred_w.abs() + pred_h.abs() + target_w.abs() + target_h.abs()).mean()
        loss = torch.exp(-(wasserstein_distance + reg_term).sqrt() / self.constant)
        final_loss = 1 - loss

        return self.loss_weight * final_loss.mean()