import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class GaussianReassignLoss(nn.Module):
    def __init__(self, loss_weight=1.0, delta=1, alpha=0.25, gamma=2.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.delta = delta  # 控制 soft label 范围
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_scores, target_onehot, iou_scores, weight=None, reduction='mean', avg_factor=None):
        # pred_scores: [N, C], target_onehot: [N, C], iou_scores: [N]
        assert pred_scores.shape == target_onehot.shape, 'shape mismatch'
        assert iou_scores.ndim == 1 and iou_scores.shape[0] == pred_scores.shape[0], 'IoU shape mismatch'

        N, C = pred_scores.shape

        # 1. soft label based on IoU
        gaussian_weights = torch.exp(-(1 - iou_scores.unsqueeze(1))**2 / (2 * self.delta ** 2))  # [N,1]
        soft_targets = target_onehot * gaussian_weights

        # 2. sigmoid and pt
        pred_sigmoid = pred_scores.sigmoid()
        pt = (1 - pred_sigmoid) * soft_targets + pred_sigmoid * (1 - soft_targets)

        # 3. focal weight
        focal_weight = (self.alpha * soft_targets + (1 - self.alpha) * (1 - soft_targets)) * pt.pow(self.gamma)

        # 4. loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_scores, soft_targets, reduction='none')
        loss = bce_loss * focal_weight

        if weight is not None:
            if weight.shape != loss.shape:
                weight = weight.view(-1, 1)
            assert weight.shape == loss.shape
            loss = loss * weight

        if reduction == 'mean':
            loss = loss.sum() / (avg_factor if avg_factor else max(loss.shape[0], 1))
        elif reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss