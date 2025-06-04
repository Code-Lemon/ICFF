from  .two_stage import TwoStageDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from mmdet.registry import MODELS
from  mmdet.models.losses.detail_loss import DetailAggregateLoss
from  mmdet.models.model_utils import segmenthead
import matplotlib.pyplot as plt
import numpy as np
import copy

@MODELS.register_module()
class CascadeRCNN_BAF(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor = None,
                 init_cfg=None):
        super(CascadeRCNN_BAF, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.boundary_loss_func = DetailAggregateLoss()
        self.rgb_global = Pred_Layer(256)
        self.dsam = DSAM(256)
        self.seghead = segmenthead(256,256,1)

    def extract_feat(self, img):
        #imgpath = img_metas[0]['filename']
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        x1 = list(x)
        #e3, p3 = self.rgb_global(x1[3])
        e4, p4 = self.rgb_global(x1[4])
        # [_, _, H, W] = p3.size()
        # p = F.interpolate(p4,
        #                   size=(H, W),
        #                   mode='bilinear',
        #                   align_corners=True) + p3
        ef, _p = self.dsam(x1[0], p4)
        #ef, _p = self.dsam(x1[0], _p)
        x1[0] = ef + x1[0]
        return x1

    def loss(self, batch_inputs,
             batch_data_samples) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        x_b0 = self.seghead(x[0])
        x_b3 = self.seghead(x[4])
        losses = dict()
        lb = torch.squeeze(batch_inputs, 1)
        transform = transforms.Grayscale()
        lb = transform(lb)
        boundery_loss = self.boundary_loss_func(x_b0, lb) + self.boundary_loss_func(x_b3, lb)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        losses['loss_bounder'] = boundery_loss
        # print(losses)
        return losses

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        x_b0 = self.seghead(x[0])
        x_b3 = self.seghead(x[4])
        losses = dict()
        lb = torch.squeeze(img, 1)
        transform = transforms.Grayscale()
        lb = transform(lb)
        boundery_loss = self.boundary_loss_func(x_b0, lb) + self.boundary_loss_func(x_b3, lb)
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        #
        losses.update(roi_losses)
        losses['loss_bounder'] = boundery_loss
        return losses
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img,img_metas)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN_BAF, self).show_result(data, result, **kwargs)

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.bn.train(True)
        self.bn.track_running_stats = False
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Pred_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1
# FF
class FF(nn.Module):
    def __init__(self, in_c):
        super(FF, self).__init__()
        self.reduce = nn.Conv2d(in_c, 32, 1)
        self.ff_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        enhanced_feat, new_pred = self.rgbd_pred_layer(ff_feat)
        return enhanced_feat, new_pred


# BF
class BF(nn.Module):
    def __init__(self, in_c):
        super(BF, self).__init__()
        self.reduce = nn.Conv2d(in_c * 2, 32, 1)
        self.bf_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(bf_feat)
        return enhanced_feat, new_pred


# ASPP for DSAM
class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c , 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class DSAM(nn.Module):
    def  __init__(self, in_c):
        super(DSAM, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgbd_pred_layer = Pred_Layer(256 * 8)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        return enhanced_feat, new_pred

