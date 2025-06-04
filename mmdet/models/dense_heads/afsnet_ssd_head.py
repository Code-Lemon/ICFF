import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from ..utils import multi_apply
from mmdet.models.dense_heads.ssd_head import SSDHead
from mmdet.registry import MODELS

@MODELS.register_module()
class AFSNetSSDHead(SSDHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 anchor_generator,
                 bbox_coder,
                 use_depthwise=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 loss_weights=dict(cls=1.0, bbox=1.0, cam=1.0),
                 train_cfg=None,
                 test_cfg=None):

        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            use_depthwise=use_depthwise,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

        self.loss_weights = loss_weights
        if train_cfg is not None:
            self.loss_cls = MODELS.build(train_cfg.get('loss_cls'))
            self.loss_bbox = MODELS.build(train_cfg.get('loss_bbox'))
            self.loss_cam = MODELS.build(train_cfg.get('loss_cam')) if 'loss_cam' in train_cfg else None

        self.cls_out_channels = num_classes + 1  # add background class
        self.num_anchors_per_loc = [4, 6, 6, 6, 4, 4]
        self.cls_outs = nn.ModuleList()
        self.reg_outs = nn.ModuleList()
        self.cam_outs = nn.ModuleList()

        for i, in_ch in enumerate(in_channels):
            na = self.num_anchors_per_loc[i]  # 每层 anchor 数
            self.cls_outs.append(nn.Conv2d(in_ch, na * self.cls_out_channels, kernel_size=3, padding=1))
            self.reg_outs.append(nn.Conv2d(in_ch, na * 4, kernel_size=3, padding=1))
            if self.loss_cam is not None:
                self.cam_outs.append(nn.Conv2d(in_ch, num_classes, kernel_size=1))


    def _generate_cam_targets(self, cam_preds, batch_gt_instances):
        """生成伪 CAM 分类 supervision 标签图（每像素类别ID）。

        可替换为真实 per-pixel 分割图。
        """
        B, _, H, W = cam_preds[0].shape
        device = cam_preds[0].device
        targets = torch.zeros((B, H, W), dtype=torch.long, device=device)

        # 示例：将每张图的第一个目标类别作为整图标签（简化演示）
        for i, gt in enumerate(batch_gt_instances):
            if len(gt.labels) > 0:
                targets[i] = gt.labels[0]
        return targets

    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     cam_preds: List[torch.Tensor],     # 新增输入：CAM 分支输出
                     anchors: List[List[torch.Tensor]],
                     batch_gt_instances: List,
                     batch_img_metas: List,
                     batch_gt_instances_ignore=None):

        # 1. 获取原始分类和回归损失
        losses = super().loss_by_feat(cls_scores, bbox_preds,
                                      batch_gt_instances, batch_img_metas,
                                      batch_gt_instances_ignore)

        # 2. 构造 CAM 监督目标
        if self.loss_cam is not None and cam_preds is not None:
            # 示例：使用 GT 实例的目标框生成伪 mask（可换成真实类别 heatmap）
            cam_targets = self._generate_cam_targets(cam_preds, batch_gt_instances)

            cam_loss = self.loss_cam(
                pred=torch.cat(cam_preds, dim=1),  # List[B, C, H, W] → B, C, H, W
                target=cam_targets
            )
            losses['loss_cam'] = cam_loss * self.loss_weights.get('cam', 1.0)

        # 3. 加权组合总损失（可由外部控制各项比例）
        losses['loss_cls'] = [loss * self.loss_weights.get('cls', 1.0) for loss in losses['loss_cls']]
        losses['loss_bbox'] = [loss * self.loss_weights.get('cls', 1.0) for loss in losses['loss_bbox']]
        return losses

    def forward(self, feats):
        cls_scores, bbox_preds, cam_preds = [], [], []

        for i, feat in enumerate(feats):
            cls_scores.append(self.cls_outs[i](feat))
            bbox_preds.append(self.reg_outs[i](feat))
            if self.loss_cam is not None:
                cam_preds.append(self.cam_outs[i](feat))

        target_size = max(feat.shape[-2] for feat in feats)  # Get the largest spatial size
        for i in range(len(cam_preds)):
            cam_preds[i] = F.interpolate(cam_preds[i], size=(target_size, target_size), mode='bilinear', align_corners=False)

        return cls_scores, bbox_preds, cam_preds

    def loss(self, feats, batch_data_samples, **kwargs):
        """Forward and compute losses."""
        cls_scores, bbox_preds, cam_preds = self.forward(feats)

        featmap_sizes = [feat.shape[-2:] for feat in cls_scores]
        device = cls_scores[0].device
        # 通过 .metainfo 获取 pad_shape
        batch_img_metas = [s.metainfo for s in batch_data_samples]

        anchor_list, _ = self.get_anchors(
            featmap_sizes=featmap_sizes,
            batch_img_metas=batch_img_metas,
            device=device
        )

        batch_gt_instances = [s.gt_instances for s in batch_data_samples]

        losses = self.loss_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            cam_preds=cam_preds,
            anchors=anchor_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=None
        )
        return losses
