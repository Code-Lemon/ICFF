import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class CategorySeparationLoss(nn.Module):
    """CAM 模块中的分类感知监督损失，对 Fcls 与 GT 类别响应图进行监督。

    Args:
        use_sigmoid (bool): 是否使用 Sigmoid（默认 False → softmax）
        loss_weight (float): 损失权重
        reduction (str): 损失聚合方式，支持 'mean', 'sum', 'none'
    """

    def __init__(self,
                 use_sigmoid: bool = False,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.reduction = reduction
        if use_sigmoid:
            self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self,
                pred: torch.Tensor,   # shape: [B, C, H, W]
                target: torch.Tensor, # shape: [B, H, W] or [B, C, H, W] if sigmoid
                weight: torch.Tensor = None,
                avg_factor: float = None,
                reduction_override: str = None,
                **kwargs):
        """计算 CAM 的 Lcfs 分类损失。

        Args:
            pred: 预测响应图 Fcls
            target: 每像素对应的类别索引 (long) 或 one-hot 编码
        """
        reduction = reduction_override if reduction_override else self.reduction

        if self.use_sigmoid:
            loss = self.criterion(pred, target.float())
        else:
            if pred.shape[2:] != target.shape[1:]:
                # resize target if needed
                target = F.interpolate(target.unsqueeze(1).float(),
                                       size=pred.shape[2:],
                                       mode='nearest').squeeze(1).long()
            loss = self.criterion(pred, target)

        return self.loss_weight * loss
