# 修改后的 FCOSHead（集成 RSDS 损失：GRL、RNWD、Lossinter）

from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from mmdet.structures.bbox import bbox_overlaps
import torch
import torch.nn.functional as F

@MODELS.register_module()
class FCOSHeadRSDS(FCOSHead):
    def __init__(self,
                 loss_inter=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_inter = MODELS.build(loss_inter) if loss_inter else None

    def loss_by_feat(self,
                     cls_scores,
                     bbox_preds,
                     centernesses,
                     batch_gt_instances,
                     batch_img_metas,
                     batch_gt_instances_ignore=None):

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        labels, bbox_targets = self.get_targets(all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat([p.repeat(num_imgs, 1) for p in all_level_points])

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        # 初始化默认损失值
        loss_cls = flatten_cls_scores.sum() * 0
        loss_bbox = flatten_bbox_preds.sum() * 0
        loss_centerness = flatten_centerness.sum() * 0

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]

            # 解码预测框和目标框用于 IoU 计算
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)

            # 计算 IoU
            iou_scores = bbox_overlaps(pos_decoded_bbox_preds, pos_decoded_target_preds, is_aligned=True).clamp(min=1e-6)

            # 构造 one-hot 标签（包含 num_classes + 1）
            target_onehot = F.one_hot(flatten_labels, num_classes=self.num_classes + 1).to(flatten_cls_scores.dtype)

            # 筛选出正样本用于分类损失
            pos_cls_scores = flatten_cls_scores[pos_inds]
            pos_targets = target_onehot[pos_inds]

            # 分类损失：Gaussian Reassignment Loss
            # loss_cls = self.loss_cls(pos_cls_scores, pos_targets, iou_scores)
            loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels, avg_factor=num_pos)
            # interaction loss (repulsion loss): 用于增加不同目标的表征间隔
            if self.loss_inter is not None:
                # 限制正样本数，避免爆显存
                max_num = 500
                if len(pos_inds) > max_num:
                    perm = torch.randperm(len(pos_inds), device=pos_inds.device)[:max_num]
                    sampled_inds = pos_inds[perm]
                else:
                    sampled_inds = pos_inds

                sampled_pred_boxes = pos_decoded_bbox_preds[perm]
                sampled_target_boxes = pos_decoded_target_preds[perm]
                sampled_labels = flatten_labels[sampled_inds]

                loss_inter = self.loss_inter(sampled_pred_boxes, sampled_target_boxes, sampled_labels)
            else:
                loss_inter = flatten_cls_scores.sum() * 0

            # 回归损失
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)

            # 中心度损失
            loss_centerness = self.loss_centerness(
                pos_centerness,
                pos_centerness_targets,
                avg_factor=num_pos)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_inter=loss_inter)