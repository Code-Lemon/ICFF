# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers import NormedConv2d
from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from ..utils import multi_apply, AFF
from .fcos_head import FCOSHead
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)
from mmdet.structures import SampleList

INF = 1e8


@MODELS.register_module()
class FCOSHead_FeatureLoss(FCOSHead):
    def __init__(self,
                 loss_feature_contrast=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_feature_contrast = MODELS.build(loss_feature_contrast)

    def loss_by_feat(
        self,
        features: List[Tensor],
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points,
                                                batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        losses = dict()

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        if getattr(self.loss_cls, 'custom_accuracy', False):
            acc = self.loss_cls.get_accuracy(flatten_cls_scores,
                                             flatten_labels)
            losses.update(acc)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        losses['loss_cls'] = loss_cls
        losses['loss_bbox'] = loss_bbox
        losses['loss_centerness'] = loss_centerness
        losses['loss_feature_contrast'] = self.feature_loss(features, batch_gt_instances)
        return losses

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        # loss_inputs = feature + outs + (batch_gt_instances, batch_img_metas,
        #                                   batch_gt_instances_ignore)
        loss_inputs = tuple([x]) + outs + (batch_gt_instances, batch_img_metas,
                                          batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def all_point_centerness_target(self,all_bbox_targets: Tensor, all_labels: Tensor) -> Tensor:
        """Compute centerness targets.

        Args:
            all_bbox_targets (Tensor): BBox targets of all bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = all_bbox_targets[:, [0, 2]]
        top_bottom = all_bbox_targets[:, [1, 3]]

        # 有负值表示该点为负样本，中心度设为0
        valid_mask = all_labels != self.num_classes

        centerness_targets = torch.zeros_like(left_right[:, 0])
        if valid_mask.any():
            l_min, l_max = left_right[valid_mask].min(dim=-1)[0], left_right[valid_mask].max(dim=-1)[0]
            t_min, t_max = top_bottom[valid_mask].min(dim=-1)[0], top_bottom[valid_mask].max(dim=-1)[0]
            centerness_targets[valid_mask] = torch.sqrt((l_min / l_max) * (t_min / t_max))
        return centerness_targets


    def feature_loss(self, features, batch_gt_instances):
        featmap_sizes = [featmap.size()[-2:] for featmap in features]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=features[0].dtype,
            device=features[0].device)
        labels, bbox_targets = self.get_targets(all_level_points,
                                                batch_gt_instances)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_feature =[
            feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
            for feature in features
        ]
        flatten_feature = torch.cat(flatten_feature)
        flatten_activations = flatten_feature.norm(p=2, dim=1)
        flatten_centerness_targets = self.all_point_centerness_target(flatten_bbox_targets, flatten_labels)
        # loss_feature_contrast
        loss_feature_contrast = self.loss_feature_contrast(
            flatten_activations, flatten_centerness_targets, flatten_labels
        )
        return loss_feature_contrast