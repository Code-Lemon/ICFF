import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from mmdet.registry import MODELS


@MODELS.register_module()
class Feature_Contrast_LOSS(nn.Module):
    def __init__(self,
                 bg_class_ind,
                 pos_loss_weight=1.0,
                 neg_loss_weight=0.1,
                 loss_weight= 1.0,
                 ):
        super().__init__()
        self.bg_class_ind = bg_class_ind
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.loss_weight = loss_weight

    def forward(self,
                activations,
                centerness_targets,
                labels,
                ) -> Tensor:
        norm_activations = F.normalize(activations, p=1, dim=0)
        pos_indexs = torch.where((labels >= 0) & (labels < self.bg_class_ind))
        neg_indexs = torch.where(labels == self.bg_class_ind)
        pos_activations = activations[pos_indexs]
        pos_activations_norm = (pos_activations - pos_activations.min()) / (pos_activations.max() - pos_activations.min() + 1e-8)
        pos_centerness_targets = centerness_targets[pos_indexs]
        neg_activations_norm = norm_activations[neg_indexs]
        loss_neg_activations = torch.sum(neg_activations_norm)
        loss_pos_activations = F.mse_loss(pos_activations_norm, pos_centerness_targets)
        return self.loss_weight * (self.neg_loss_weight * loss_neg_activations + self.pos_loss_weight * loss_pos_activations)


# @MODELS.register_module()
# class Feature_Consistency_LOSS(nn.Module):
#     def __init__(self,
#                  num_class,
#                  threshold,
#                  intra_loss_weight=1.0,
#                  # inter_loss_weight=1.0,
#                  loss_weight= 1.0,
#                  ):
#         super().__init__()
#         self.num_class = num_class
#         self.intra_loss_weight = intra_loss_weight
#         # self.inter_loss_weight = inter_loss_weight
#         self.threshold = threshold
#         self.loss_weight = loss_weight
#
#     def forward(self,
#                 features,
#                 centerness,
#                 labels,
#                 ) -> Tensor:
#         pos_indexs = ((labels >= 0) & (labels < self.num_class)) & (centerness > self.threshold)
#         centers = []
#         loss_intra = torch.tensor(0.0, device=features.device)
#         for i in range(self.num_class):
#             class_feature = features[pos_indexs & (labels == i)]
#             class_feature = F.normalize(class_feature, p=2, dim=1)
#             if class_feature.shape[0] == 0:
#                 continue
#             center = class_feature.mean(dim=0, keepdim=True)
#             centers.append(center)
#             sim = torch.matmul(class_feature, center.T)
#             loss_intra += 1 - sim.mean()  # 类内越相似越好
#         if len(centers) > 0:
#             loss_intra = loss_intra / len(centers)
#         # n = len(centers)
#         # if n in [0, 1]:
#         #     loss_intra = torch.tensor(0.0)
#         #     loss_inter = torch.tensor(0.0)
#         # else:
#         #     loss_intra = loss_intra / n
#         #     centers = torch.cat(centers, dim=0)  # [C, D]
#         #     inter_sim = torch.matmul(centers, centers.T)  # [C, C]
#         #     inter_mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
#         #     loss_inter = inter_sim[inter_mask].mean()  # 类间尽量不相似
#         # return self.loss_weight * (self.intra_loss_weight * loss_intra + self.inter_loss_weight * loss_inter)
#         return self.loss_weight * (self.intra_loss_weight * loss_intra)



# @MODELS.register_module()
# class FCLOSS(nn.Module):
#     def __init__(self,
#                  strides,
#                  num_classes=3,
#                  multi_scale_feature_alignment_loss_weight=1.0,
#                  target_saliency_enhancement_loss_weight=1.0,
#                  loss_weight= 1.0,
#                  ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.loss_weight = loss_weight
#         self.strides = strides
#         self.multi_scale_feature_alignment_loss_weight = multi_scale_feature_alignment_loss_weight
#         self.target_saliency_enhancement_loss_weight = target_saliency_enhancement_loss_weight
#
#     def forward(self,
#                 feature,
#                 batch_gt_instances
#                 ) -> Tensor:
#         num_sample = len(batch_gt_instances)
#         for i in range(num_sample):
#             if i == 0:
#                 loss = self.item_loss([feature[j][i] for j in range(len(feature))], batch_gt_instances[i])
#             else:
#                 loss += self.item_loss([feature[j][i] for j in range(len(feature))], batch_gt_instances[i])
#         return self.loss_weight * loss / num_sample
#
#
#     def item_loss(self, feature, instance):
#         class_pos_features = [[] for i in range(self.num_classes)]
#         layer_neg_mask = [torch.ones(feature[i].shape[1:], dtype=torch.bool) for i in range(len(self.strides))]
#         neg_feature = []
#         bboxes = instance.get('bboxes')
#         labels = instance.get('labels')
#         for i in range(len(bboxes)):
#             bbox = bboxes[i]
#             label = int(labels[i])
#             for j in range(len(self.strides)):
#                 feature_bbox = bbox // self.strides[j]
#                 center = (int((feature_bbox[3] + feature_bbox[1]) / 2), int((feature_bbox[2] + feature_bbox[0]) / 2))
#                 class_pos_features[label].append(feature[j][:, center[0], center[1]])
#                 layer_neg_mask[j][int(feature_bbox[1]): int(feature_bbox[3]), int(feature_bbox[0]): int(feature_bbox[2])] = False
#         for i in range(len(self.strides)):
#             neg_feature.append(torch.abs(feature[i][:,  layer_neg_mask[i]]).mean(1))
#         multi_scale_feature_alignment_loss = self.Multi_Scale_Feature_Alignment_Loss(class_pos_features)
#         target_saliency_enhancement_loss = self.Target_Saliency_Enhancement_Loss(neg_feature, class_pos_features)
#         return self.loss_weight * (multi_scale_feature_alignment_loss + target_saliency_enhancement_loss)
#
#     def Multi_Scale_Feature_Alignment_Loss(self, class_pos_features):
#         count = 0
#         for i in range(len(class_pos_features)):
#             if class_pos_features[i] == []:
#                 continue
#             features = torch.vstack(class_pos_features[i])
#             mean = features.mean(dim=0, keepdim=True)
#             if count == 0:
#                 var_loss = torch.sum((features - mean) ** 2) / features.shape[0]
#             else:
#                 var_loss = var_loss + torch.sum((features - mean) ** 2) / features.shape[0]
#             count += 1
#         return self.multi_scale_feature_alignment_loss_weight * var_loss / count
#
#     def Target_Saliency_Enhancement_Loss(self, neg_feature, class_pos_features):
#         class_pos_features = [item for item in class_pos_features if item]
#         neg_features = torch.vstack(neg_feature)
#         # pos_features = torch.vstack([torch.vstack(class_pos_features[i]) for i in range(len(class_pos_features))])
#         neg_loss = torch.sum(neg_features ** 2) / neg_features.shape[0]
#         # pos_loss = -(pos_features ** 2).mean()
#         # return self.target_saliency_enhancement_loss_weight * (neg_loss + pos_loss)
#         return self.target_saliency_enhancement_loss_weight * neg_loss



