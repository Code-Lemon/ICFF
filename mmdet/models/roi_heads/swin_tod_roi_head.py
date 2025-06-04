import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi

@MODELS.register_module()
class INENRoIHead(StandardRoIHead):
    def __init__(self,
                 *args,
                 inen_head=None,
                 alpha=0.7,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.inen_head = MODELS.build(inen_head) if inen_head else None

    def loss(self,
             x: Tuple[Tensor],
             rpn_results_list: List[InstanceData],
             batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Compute losses from bbox/mask heads and INEN head."""
        # 1. 原始 bbox loss
        losses = super().loss(x, rpn_results_list, batch_data_samples, **kwargs)

        # 2. INEN forward + loss
        if self.inen_head is not None:
            # 2.1 从 RPN 结果中提取 proposals
            proposals_list = [res.priors for res in rpn_results_list]
            rois = bbox2roi(proposals_list)  # [N, 5] with batch indices
            # proposals = [res.priors  for res in rpn_results_list]  # List[T, 4]

            # 2.2 提取 roi_feats → proposal_feats
            roi_feats = self.bbox_roi_extractor(x, rois)  # [N, C, 7, 7]
            proposal_feats = roi_feats.mean(dim=[2, 3])         # [N, C]

            # 2.3 构造 edge_index（全连接图）
            N = proposal_feats.size(0)
            if N == 0:
                return losses  # 跳过空 proposals 的情况
            row_idx = torch.arange(N, device=proposal_feats.device).repeat_interleave(N)
            col_idx = torch.arange(N, device=proposal_feats.device).repeat(N)
            edge_index = torch.stack([row_idx, col_idx], dim=0)  # [2, N*N]

            # 2.4 构造标签（真实任务应从采样器或 assigner 获取）
            num_classes = self.inen_head.cls.out_features
            pseudo_labels = torch.randint(0, num_classes, (N,), device=proposal_feats.device)

            # 2.5 计算 INEN 分类 loss
            logits = self.inen_head(proposal_feats, edge_index)
            loss_inen = self.inen_head.loss(logits, pseudo_labels)

            # 2.6 融合分类损失（与 RoIHead 的 loss_cls）
            if 'loss_cls' in losses:
                loss_cls_combined = self.alpha * losses['loss_cls'] + (1 - self.alpha) * loss_inen
                losses['loss_cls'] = loss_cls_combined
            else:
                losses['loss_cls_inen'] = loss_inen

        return losses
