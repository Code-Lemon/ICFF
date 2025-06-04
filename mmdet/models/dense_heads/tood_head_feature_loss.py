# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import deform_conv2d
from mmengine import MessageHub
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import anchor_inside_flags, MlvlPointGenerator
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply,
                     sigmoid_geometric_mean, unmap, unpack_gt_instances)
from .tood_head import TOODHead

INF = 1e8

@MODELS.register_module()
class TOODHead_FeatureLoss(TOODHead):

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
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0]
            for bbox_pred, stride in zip(bbox_preds,
                                         self.prior_generator.strides)
        ], 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox, \
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls,
                    loss_bbox=losses_bbox,
                    loss_feature_contrast=self.feature_loss(features, batch_gt_instances))

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