# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from ..task_modules.prior_generators import MlvlPointGenerator
from ..task_modules.samplers import PseudoSampler
from ..utils import multi_apply
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)
from mmdet.structures import SampleList
from .vfnet_head import VFNetHead

INF = 1e8


@MODELS.register_module()
class VFNetHead_FeatureLoss(VFNetHead):
    def __init__(self,
                 loss_feature_contrast=None,
                 strides=(4, 8, 16, 32, 64),
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_feature_contrast = MODELS.build(loss_feature_contrast)
        self.regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))
        self.strides = strides
        self.prior_generator_ancherfree = MlvlPointGenerator(strides)


    def loss_by_feat(
            self,
            features: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            bbox_preds_refine: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.fcos_prior_generator.grid_priors(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, label_weights, bbox_targets, bbox_weights = self.get_targets(
            cls_scores,
            all_level_points,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3,
                              1).reshape(-1,
                                         self.cls_out_channels).contiguous()
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_refine = [
            bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred_refine in bbox_preds_refine
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.where(
            ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]

        pos_decoded_bbox_preds = self.bbox_coder.decode(
            pos_points, pos_bbox_preds)
        pos_decoded_target_preds = self.bbox_coder.decode(
            pos_points, pos_bbox_targets)
        iou_targets_ini = bbox_overlaps(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        bbox_avg_factor_ini = reduce_mean(
            bbox_weights_ini.sum()).clamp_(min=1).item()

        pos_decoded_bbox_preds_refine = \
            self.bbox_coder.decode(pos_points, pos_bbox_preds_refine)
        iou_targets_rf = bbox_overlaps(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        bbox_avg_factor_rf = reduce_mean(
            bbox_weights_rf.sum()).clamp_(min=1).item()

        if num_pos > 0:
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_ini,
                avg_factor=bbox_avg_factor_ini)

            loss_bbox_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_rf,
                avg_factor=bbox_avg_factor_rf)

            # build IoU-aware cls_score targets
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)

        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_rf=loss_bbox_refine,
            loss_feature_contrast=self.feature_loss(features, batch_gt_instances))

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = tuple([x]) + outs + (batch_gt_instances, batch_img_metas,
                                          batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def feature_loss(self, features, batch_gt_instances):
        featmap_sizes = [featmap.size()[-2:] for featmap in features]
        all_level_points = self.prior_generator_ancherfree.grid_priors(
            featmap_sizes,
            dtype=features[0].dtype,
            device=features[0].device)
        labels, bbox_targets = self.get_targets_ancherfree(all_level_points,
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


    def get_targets_ancherfree(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor]]:
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_targets_single_ancherfree,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            # bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_targets_single_ancherfree(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets