import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps


@MODELS.register_module()
class LossInteraction(nn.Module):
    def __init__(self, loss_weight=0.1, num_classes=3, max_samples=500):
        super(LossInteraction, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.max_samples = max_samples

    def forward(self, pred_box, gt_box, labels):
        # 只保留正样本（背景类别用 num_classes 编码）
        pos_inds = (labels >= 0) & (labels < self.num_classes)
        pred_box = pred_box[pos_inds]
        gt_box = gt_box[pos_inds]

        # 没有正样本时返回 0
        if pred_box.size(0) <= 1:
            return pred_box.sum() * 0.0

        # 限制最大计算数量，避免内存爆炸
        if pred_box.size(0) > self.max_samples:
            perm = torch.randperm(pred_box.size(0), device=pred_box.device)
            pred_box = pred_box[perm[:self.max_samples]]
            gt_box = gt_box[perm[:self.max_samples]]

        # 计算预测框之间的 pairwise L2 距离
        diff = pred_box.unsqueeze(1) - pred_box.unsqueeze(0)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-6)  # [N, N]
        temperature = 100.0  # 新增温度控制项
        repulsion = torch.exp(-dist / temperature)
        # 可选抑制函数：指数抑制 repulsion 越近越大

        # 去除自身对自身影响（主对角线）
        repulsion = repulsion - torch.diag_embed(torch.diag(repulsion))

        # 平均损失
        loss = repulsion.mean()
        return self.loss_weight * loss
